"""
LeanAI · Phase 2 Engine
New capabilities over Phase 1:
  - Vector episodic memory (ChromaDB semantic search)
  - Causal world model (persistent entity + user profile tracking)
  - Memory-first answering (answers from memory before touching model)
  - /remember command for explicit fact storage
  - /profile command to see what AI knows about you
"""

import time
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from core.router import TaskRouter, Tier
from core.watchdog import MetaCognitiveWatchdog
from core.confidence import ConfidenceScoringEngine
from tools.z3_verifier import Z3Verifier, Verdict
from memory.hierarchy_v2 import HierarchicalMemoryV2
from training.self_improve import TrainingDataStore, SelfPlayEngine, FeedbackSignal


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1


@dataclass
class LeanAIResponse:
    text: str
    pair_id: str
    confidence: float
    confidence_label: str
    confidence_bar: str
    tier_used: str
    latency_ms: float
    verified: bool
    verification_summary: str
    claims_checked: int
    claims_correct: int
    memory_context_used: bool
    answered_from_memory: bool
    quality_score: float
    warning: Optional[str] = None
    corrected: bool = False


DEFAULT_MODEL_PATH = Path.home() / ".leanai" / "models" / "phi3-mini-q4.gguf"


class LeanAIEngineV2:
    """Phase 2 engine — vector memory + world model + memory-first answering."""

    def __init__(self, model_path: Optional[str] = None, verbose: bool = False):
        self.model_path = model_path or str(DEFAULT_MODEL_PATH)
        self.verbose    = verbose

        self.router     = TaskRouter()
        self.watchdog   = MetaCognitiveWatchdog()
        self.confidence = ConfidenceScoringEngine()
        self.verifier   = Z3Verifier()
        self.memory     = HierarchicalMemoryV2()
        self.store      = TrainingDataStore()
        self.self_play  = SelfPlayEngine()

        self._model        = None
        self._model_loaded = False

        print("[LeanAI v2] Engine initialized.")
        print(f"[LeanAI v2] Memory backend: {self.memory.episodic.backend}")
        print(f"[LeanAI v2] World model: {self.memory.world.stats()['entities']} entities")
        print(f"[LeanAI v2] Verifier: {self.verifier.status}")

    def generate(self, query: str,
                 config: Optional[GenerationConfig] = None) -> LeanAIResponse:
        config = config or GenerationConfig()
        start  = time.time()

        # ── 1. Route ──────────────────────────────────────────────────
        decision = self.router.route(query, self.memory.working.current_tokens)

        # ── 2. Tiny tier ──────────────────────────────────────────────
        if decision.tier == Tier.TINY:
            text = self._handle_tiny(query)
            latency = (time.time() - start) * 1000
            pair = self.store.add_pair(
                instruction=query, response=text,
                feedback=FeedbackSignal.EXCELLENT,
                confidence=0.99, verified=False,
                latency_ms=latency, tier_used="tiny",
            )
            self.memory.record_exchange(query, text)
            return self._build_response(text, pair.id, 0.99, "High",
                                        "████████████████████",
                                        "tiny", latency, False, False,
                                        "Not needed.", 0, 0, False, True, 1.0)

        # ── 3. Memory-first answering ─────────────────────────────────
        # Try to answer from world model before touching the neural model
        memory_answer = self.memory.answer_from_memory(query)
        if memory_answer:
            latency = (time.time() - start) * 1000
            self.memory.record_exchange(query, memory_answer)
            pair = self.store.add_pair(
                instruction=query, response=memory_answer,
                feedback=FeedbackSignal.EXCELLENT,
                confidence=0.95, verified=True,
                latency_ms=latency, tier_used="memory",
            )
            return self._build_response(
                memory_answer, pair.id, 0.95, "High",
                "███████████████████░", "memory (instant)",
                latency, True, False,
                "Answered from world model.", 0, 0, True, True, 1.0,
            )

        # ── 4. Memory context ─────────────────────────────────────────
        mem_ctx  = self.memory.prepare_context(query)
        mem_used = bool(mem_ctx)

        # ── 5. Build prompt ───────────────────────────────────────────
        prompt = self._build_prompt(
            query, mem_ctx,
            self.memory.working.get_context_window(max_tokens=1024),
        )

        # ── 6. Generate ───────────────────────────────────────────────
        response_text = self._generate_with_model(prompt, config)

        # ── 7. Confidence ─────────────────────────────────────────────
        conf_score = self.confidence.score_from_text(response_text)

        # ── 8. Verification ───────────────────────────────────────────
        verified = False
        corrected = False
        verify_summary = "Not checked."
        claims_checked = claims_correct = 0

        if decision.requires_verifier or conf_score.needs_verification:
            report = self.verifier.verify_text(response_text, query)
            claims_checked = report.claims_found
            claims_correct = report.claims_verified
            verified  = report.overall_verdict == Verdict.TRUE
            corrected = report.overall_verdict == Verdict.FALSE
            verify_summary = report.summary
            if report.corrected_text:
                response_text = report.corrected_text
            conf_score = self.confidence.combine_with_verification(
                conf_score, verified, corrected)

        # ── 9. Memory storage ─────────────────────────────────────────
        self.memory.record_exchange(query, response_text)

        latency = (time.time() - start) * 1000

        # ── 10. Training ──────────────────────────────────────────────
        feedback = (FeedbackSignal.EXCELLENT if verified else
                    FeedbackSignal.GOOD if conf_score.overall > 0.7 else
                    FeedbackSignal.NEUTRAL)
        pair = self.store.add_pair(
            instruction=query, response=response_text,
            feedback=feedback, confidence=conf_score.overall,
            verified=verified, latency_ms=latency,
            tier_used=decision.tier.value,
        )

        return self._build_response(
            response_text, pair.id,
            conf_score.overall, conf_score.label, conf_score.bar,
            decision.tier.value, latency,
            verified, corrected, verify_summary,
            claims_checked, claims_correct,
            mem_used, False,
            round(pair.quality_score, 3),
            warning=conf_score.explanation if conf_score.needs_clarification else None,
        )

    def remember(self, fact: str):
        """Explicitly store a fact into long-term memory."""
        self.memory.remember_fact(fact)
        print(f"[LeanAI v2] Stored in memory: {fact}")

    def get_profile(self) -> dict:
        """Return what the AI knows about the user."""
        return self.memory.world.get_user_profile()

    def give_feedback(self, pair_id: str, good: bool):
        signal = FeedbackSignal.EXCELLENT if good else FeedbackSignal.WRONG
        self.store.update_feedback(pair_id, signal)

    def generate_self_play_batch(self, n: int = 10) -> int:
        pairs = self.self_play.generate_batch(n)
        for sp in pairs:
            self.store.add_pair(
                instruction=sp.problem, response=sp.solution,
                feedback=FeedbackSignal.EXCELLENT if sp.verified else FeedbackSignal.GOOD,
                confidence=0.95 if sp.verified else 0.7,
                verified=sp.verified, latency_ms=0,
                tier_used="self_play", tags=[sp.domain, "synthetic"],
            )
        return len(pairs)

    # ── Private ────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model_loaded:
            return
        if not Path(self.model_path).exists():
            print(f"[LeanAI v2] No model found — demo mode.")
            self._model_loaded = True
            return
        try:
            from llama_cpp import Llama
            print("[LeanAI v2] Loading model — optimized for i7-11800H...")
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=8,
                n_batch=512,
                n_gpu_layers=0,
                use_mmap=True,
                use_mlock=False,
                logits_all=False,
                verbose=self.verbose,
            )
            self._model_loaded = True
            print("[LeanAI v2] Model loaded. Ready.")
        except ImportError:
            print("[LeanAI v2] llama-cpp-python not installed.")
            self._model_loaded = True

    def _generate_with_model(self, prompt: str, config: GenerationConfig) -> str:
        self._load_model()
        if self._model is None:
            return "Demo mode — download a model with: python setup.py --download-model"
        result = self._model(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=["<|user|>", "<|end|>", "<|assistant|>", "\nYou:"],
            echo=False,
        )
        text = result["choices"][0]["text"].strip()
        for token in ["<|assistant|>", "<|user|>", "<|end|>", "<|system|>"]:
            text = text.split(token)[0].strip()
        return text

    def _build_prompt(self, query, memory_context, history) -> str:
        system = (
            "You are LeanAI — a fast, accurate, and honest AI assistant. "
            "You run locally on the user's device. "
            "Be concise and direct. Never repeat the question. "
            "Never add follow-up questions at the end of your answer."
        )
        parts = [f"<|system|>\n{system}<|end|>\n"]
        if memory_context:
            parts.append(f"<|system|>\nContext about this user:\n{memory_context}<|end|>\n")
        for msg in history[-6:]:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"<|user|>\n{content}<|end|>\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}<|end|>\n")
        parts.append(f"<|user|>\n{query}<|end|>\n<|assistant|>\n")
        return "".join(parts)

    def _handle_tiny(self, query: str) -> str:
        import re, math as m
        q = query.strip().lower()
        if re.match(r"^(hi|hello|hey)\b", q): return "Hello! How can I help?"
        if re.match(r"^(thanks|thank you)\b", q): return "You're welcome!"
        if re.match(r"^(bye|goodbye)\b", q): return "Goodbye!"
        try:
            expr = re.sub(r"[^0-9\+\-\*\/\(\)\.\s\^]", "", query).replace("^", "**")
            if expr.strip():
                result = eval(expr, {"__builtins__": {}, "sqrt": m.sqrt})
                return str(result)
        except Exception:
            pass
        return "Could you tell me more?"

    def _build_response(self, text, pair_id, confidence, label, bar,
                        tier, latency, verified, corrected, verify_summary,
                        claims_checked, claims_correct, mem_used,
                        answered_from_memory, quality, warning=None):
        return LeanAIResponse(
            text=text, pair_id=pair_id,
            confidence=confidence, confidence_label=label,
            confidence_bar=bar, tier_used=tier,
            latency_ms=round(latency, 1),
            verified=verified, corrected=corrected,
            verification_summary=verify_summary,
            claims_checked=claims_checked, claims_correct=claims_correct,
            memory_context_used=mem_used,
            answered_from_memory=answered_from_memory,
            quality_score=quality, warning=warning,
        )

    def status(self) -> dict:
        mem = self.memory.stats()
        return {
            "phase": 2,
            "model_loaded": self._model is not None,
            "verifier": self.verifier.status,
            "memory": mem,
            "training": self.store.stats(),
        }

"""
LeanAI · Phase 1 Engine
Upgraded from Phase 0 with:
  - Z3 formal verifier (replaces heuristic verifier)
  - Calibrated confidence scoring (replaces entropy heuristic)
  - Self-improvement loop (records every exchange as training data)
  - Richer response metadata
"""

import time
import os
from dataclasses import dataclass
from typing import Optional, Iterator
from pathlib import Path

from core.router import TaskRouter, Tier
from core.watchdog import MetaCognitiveWatchdog
from core.confidence import ConfidenceScoringEngine
from tools.z3_verifier import Z3Verifier, Verdict
from memory.hierarchy import HierarchicalMemory
from training.self_improve import TrainingDataStore, SelfPlayEngine, FeedbackSignal


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stream: bool = False


@dataclass
class LeanAIResponse:
    # Core output
    text: str
    pair_id: str              # training pair ID — use for feedback

    # Confidence
    confidence: float
    confidence_label: str
    confidence_bar: str
    confidence_explanation: str

    # Routing
    tier_used: str
    latency_ms: float

    # Verification
    verified: bool
    verification_summary: str
    claims_checked: int
    claims_correct: int

    # Memory
    memory_context_used: bool

    # Self-improvement
    quality_score: float

    # Optional
    warning: Optional[str] = None
    corrected: bool = False


DEFAULT_MODEL_PATH = Path.home() / ".leanai" / "models" / "phi3-mini-q4.gguf"


class LeanAIEngineV1:
    """Phase 1 engine — full pipeline with formal verification and confidence scoring."""

    def __init__(self, model_path: Optional[str] = None, verbose: bool = False):
        self.model_path = model_path or str(DEFAULT_MODEL_PATH)
        self.verbose = verbose

        # Subsystems
        self.router     = TaskRouter()
        self.watchdog   = MetaCognitiveWatchdog()
        self.confidence = ConfidenceScoringEngine()
        self.verifier   = Z3Verifier()
        self.memory     = HierarchicalMemory()
        self.store      = TrainingDataStore()
        self.self_play  = SelfPlayEngine()

        self._model = None
        self._model_loaded = False

        print("[LeanAI v1] Engine initialized.")
        print(f"[LeanAI v1] Verifier: {self.verifier.status}")
        print(f"[LeanAI v1] Training store: {self.store.stats()['total']} pairs")

    def generate(self, query: str, config: Optional[GenerationConfig] = None) -> LeanAIResponse:
        config = config or GenerationConfig()
        start  = time.time()

        # ── 1. Route ─────────────────────────────────────────────────
        decision = self.router.route(query, self.memory.working.current_tokens)
        if self.verbose:
            print(self.router.explain(decision))

        # ── 2. Tiny tier ─────────────────────────────────────────────
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
            return LeanAIResponse(
                text=text, pair_id=pair.id,
                confidence=0.99, confidence_label="High",
                confidence_bar="████████████████████",
                confidence_explanation="Rule-based response — deterministic.",
                tier_used="tiny", latency_ms=round(latency, 1),
                verified=False, verification_summary="Not needed.",
                claims_checked=0, claims_correct=0,
                memory_context_used=False, quality_score=1.0,
            )

        # ── 3. Memory context ─────────────────────────────────────────
        mem_ctx = self.memory.prepare_context(query)
        mem_used = bool(mem_ctx)

        # ── 4. Build prompt ───────────────────────────────────────────
        prompt = self._build_prompt(
            query, mem_ctx,
            self.memory.working.get_context_window(max_tokens=1024),
        )

        # ── 5. Generate ───────────────────────────────────────────────
        response_text = self._generate_with_model(prompt, config)

        # ── 6. Confidence scoring ─────────────────────────────────────
        conf_score = self.confidence.score_from_text(response_text)

        # ── 7. Formal verification ────────────────────────────────────
        verified = False
        corrected = False
        verify_summary = "Not checked."
        claims_checked = 0
        claims_correct = 0

        should_verify = decision.requires_verifier or conf_score.needs_verification
        if should_verify:
            report = self.verifier.verify_text(response_text, query)
            claims_checked = report.claims_found
            claims_correct = report.claims_verified
            verified  = report.overall_verdict == Verdict.TRUE
            corrected = report.overall_verdict == Verdict.FALSE

            verify_summary = report.summary

            if report.corrected_text:
                response_text = report.corrected_text

            # Update confidence with verification result
            conf_score = self.confidence.combine_with_verification(
                conf_score, verified, corrected
            )

        # ── 8. Memory ─────────────────────────────────────────────────
        self.memory.record_exchange(query, response_text)

        latency = (time.time() - start) * 1000

        # ── 9. Training store ─────────────────────────────────────────
        feedback = (
            FeedbackSignal.EXCELLENT if verified else
            FeedbackSignal.GOOD      if conf_score.overall > 0.7 else
            FeedbackSignal.NEUTRAL
        )
        pair = self.store.add_pair(
            instruction=query,
            response=response_text,
            feedback=feedback,
            confidence=conf_score.overall,
            verified=verified,
            latency_ms=latency,
            tier_used=decision.tier.value,
        )

        return LeanAIResponse(
            text=response_text,
            pair_id=pair.id,
            confidence=conf_score.overall,
            confidence_label=conf_score.label,
            confidence_bar=conf_score.bar,
            confidence_explanation=conf_score.explanation,
            tier_used=decision.tier.value,
            latency_ms=round(latency, 1),
            verified=verified,
            verification_summary=verify_summary,
            claims_checked=claims_checked,
            claims_correct=claims_correct,
            memory_context_used=mem_used,
            quality_score=round(pair.quality_score, 3),
            warning=conf_score.explanation if conf_score.needs_clarification else None,
            corrected=corrected,
        )

    def give_feedback(self, pair_id: str, good: bool):
        """User feedback — updates the training pair quality score."""
        signal = FeedbackSignal.EXCELLENT if good else FeedbackSignal.WRONG
        self.store.update_feedback(pair_id, signal)
        print(f"[LeanAI v1] Feedback recorded for pair {pair_id}: {'good' if good else 'wrong'}")

    def generate_self_play_batch(self, n: int = 10) -> int:
        """Generate synthetic training data via self-play. Returns count added."""
        pairs = self.self_play.generate_batch(n)
        for sp in pairs:
            self.store.add_pair(
                instruction=sp.problem,
                response=sp.solution,
                feedback=FeedbackSignal.EXCELLENT if sp.verified else FeedbackSignal.GOOD,
                confidence=0.95 if sp.verified else 0.7,
                verified=sp.verified,
                latency_ms=0,
                tier_used="self_play",
                tags=[sp.domain, "synthetic"],
            )
        return len(pairs)

    # ── Private ────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model_loaded:
            return
        if not Path(self.model_path).exists():
            print(f"[LeanAI v1] No model at {self.model_path} — demo mode.")
            self._model_loaded = True
            return
        try:
            from llama_cpp import Llama
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=os.cpu_count(),
                n_gpu_layers=0,
                logits_all=True,  # Phase 1: needed for real logprob scoring
                verbose=self.verbose,
            )
            self._model_loaded = True
            print("[LeanAI v1] Model loaded.")
        except ImportError:
            print("[LeanAI v1] llama-cpp-python not installed.")
            self._model_loaded = True

    def _generate_with_model(self, prompt: str, config: GenerationConfig) -> str:
        self._load_model()
        if self._model is None:
            return self._demo_response(prompt)
        result = self._model(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            echo=False,
        )
        return result["choices"][0]["text"].strip()

    def _build_prompt(self, query, memory_context, history) -> str:
        parts = [
            "<|system|>\nYou are LeanAI — fast, accurate, and honest. "
            "You run locally. You are concise and always acknowledge uncertainty.\n"
        ]
        if memory_context:
            parts.append(f"<|context|>\n{memory_context}\n")
        for msg in history[-6:]:
            parts.append(f"<|{msg['role']}|>\n{msg['content']}\n")
        parts.append(f"<|user|>\n{query}\n<|assistant|>\n")
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

    def _demo_response(self, prompt: str) -> str:
        return (
            "Demo mode — full pipeline running. "
            "Download a model with: python setup.py --download-model"
        )

    def status(self) -> dict:
        return {
            "phase": 1,
            "model_loaded": self._model is not None,
            "verifier": self.verifier.status,
            "memory": self.memory.stats(),
            "training": self.store.stats(),
        }

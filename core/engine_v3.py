"""
LeanAI · Phase 3 Engine
Adds to Phase 2:
  - Confidence calibration (fixes 50% showing on everything)
  - Background continual training loop
  - Enhanced self-play across math/code/reasoning
  - /train command to manually trigger training
  - /generate command to generate self-play data
"""

import time
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from core.router import TaskRouter, Tier
from core.watchdog import MetaCognitiveWatchdog
from core.confidence import ConfidenceScoringEngine
from core.calibrator import ConfidenceCalibrator
from tools.z3_verifier import Z3Verifier, Verdict
from memory.hierarchy_v2 import HierarchicalMemoryV2
from training.self_improve import TrainingDataStore, FeedbackSignal
from training.continual_trainer import ContinualTrainer, TrainingConfig
from training.self_play_v2 import EnhancedSelfPlayEngine


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
    confidence_method: str
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


class LeanAIEngineV3:
    """Phase 3 — Continual learning + calibrated confidence."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        verbose: bool = False,
        auto_train: bool = True,
    ):
        self.model_path = model_path or str(DEFAULT_MODEL_PATH)
        self.verbose    = verbose

        # Core subsystems
        self.router      = TaskRouter()
        self.watchdog    = MetaCognitiveWatchdog()
        self.scorer      = ConfidenceScoringEngine()
        self.calibrator  = ConfidenceCalibrator()
        self.verifier    = Z3Verifier()
        self.memory      = HierarchicalMemoryV2()
        self.store       = TrainingDataStore()
        self.self_play   = EnhancedSelfPlayEngine()

        # Phase 3: Continual trainer
        self.trainer = ContinualTrainer(
            store=self.store,
            config=TrainingConfig(
                check_interval_minutes=30,
                min_pairs_to_train=50,
                self_play_batch_size=20,
            ),
        )

        self._model        = None
        self._model_loaded = False

        # Start background trainer
        if auto_train:
            self.trainer.start()

        print("[LeanAI v3] Engine initialized.")
        print(f"[LeanAI v3] Memory: {self.memory.episodic.backend}")
        print(f"[LeanAI v3] Training: {self.store.stats()['total']} pairs")
        print(f"[LeanAI v3] Background trainer: {'running' if auto_train else 'off'}")

    def generate(self, query: str, config: Optional[GenerationConfig] = None) -> LeanAIResponse:
        config = config or GenerationConfig()
        start  = time.time()

        # ── 1. Route ──────────────────────────────────────────────────
        decision = self.router.route(query, self.memory.working.current_tokens)

        # ── 2. Tiny tier ──────────────────────────────────────────────
        if decision.tier == Tier.TINY:
            text = self._handle_tiny(query)
            latency = (time.time() - start) * 1000
            cal = self.calibrator.calibrate(0.99, text, "tiny", False, query)
            pair = self.store.add_pair(
                instruction=query, response=text,
                feedback=FeedbackSignal.EXCELLENT,
                confidence=cal.calibrated, verified=False,
                latency_ms=latency, tier_used="tiny",
            )
            self.memory.record_exchange(query, text)
            return self._wrap(text, pair.id, cal, "tiny",
                              latency, False, False,
                              "Not needed.", 0, 0, False, True, 1.0)

        # ── 3. Memory-first ───────────────────────────────────────────
        memory_answer = self.memory.answer_from_memory(query)
        if memory_answer:
            latency = (time.time() - start) * 1000
            cal = self.calibrator.calibrate(0.95, memory_answer, "memory", True, query)
            self.memory.record_exchange(query, memory_answer)
            pair = self.store.add_pair(
                instruction=query, response=memory_answer,
                feedback=FeedbackSignal.EXCELLENT,
                confidence=cal.calibrated, verified=True,
                latency_ms=latency, tier_used="memory",
            )
            return self._wrap(memory_answer, pair.id, cal,
                              "memory (instant)", latency,
                              True, False, "From world model.",
                              0, 0, True, True, 1.0)

        # ── 4. Memory context ─────────────────────────────────────────
        mem_ctx  = self.memory.prepare_context(query)
        mem_used = bool(mem_ctx)

        # ── 5. Prompt + generate ──────────────────────────────────────
        prompt = self._build_prompt(
            query, mem_ctx,
            self.memory.working.get_context_window(max_tokens=1024),
        )
        response_text = self._generate_with_model(prompt, config)

        # ── 6. Score + calibrate ──────────────────────────────────────
        raw_score  = self.scorer.score_from_text(response_text)
        cal = self.calibrator.calibrate(
            raw_score.overall, response_text,
            decision.tier.value, False, query,
        )

        # ── 7. Verify ─────────────────────────────────────────────────
        verified = corrected = False
        verify_summary = "Not checked."
        claims_checked = claims_correct = 0

        if decision.requires_verifier or raw_score.needs_verification:
            report = self.verifier.verify_text(response_text, query)
            claims_checked = report.claims_found
            claims_correct = report.claims_verified
            verified  = report.overall_verdict == Verdict.TRUE
            corrected = report.overall_verdict == Verdict.FALSE
            verify_summary = report.summary

            if report.corrected_text:
                response_text = report.corrected_text

            # Re-calibrate with verification result
            cal = self.calibrator.calibrate(
                raw_score.overall, response_text,
                decision.tier.value, verified, query,
            )

        # ── 8. Memory ─────────────────────────────────────────────────
        self.memory.record_exchange(query, response_text)

        latency = (time.time() - start) * 1000

        # ── 9. Training ───────────────────────────────────────────────
        feedback = (FeedbackSignal.EXCELLENT if verified else
                    FeedbackSignal.GOOD if cal.calibrated > 0.7 else
                    FeedbackSignal.NEUTRAL)
        pair = self.store.add_pair(
            instruction=query, response=response_text,
            feedback=feedback, confidence=cal.calibrated,
            verified=verified, latency_ms=latency,
            tier_used=decision.tier.value,
        )

        return self._wrap(
            response_text, pair.id, cal, decision.tier.value,
            latency, verified, corrected, verify_summary,
            claims_checked, claims_correct, mem_used, False,
            round(pair.quality_score, 3),
        )

    def give_feedback(self, pair_id: str, good: bool):
        signal = FeedbackSignal.EXCELLENT if good else FeedbackSignal.WRONG
        self.store.update_feedback(pair_id, signal)
        print(f"[LeanAI v3] Feedback: {'good' if good else 'wrong'}")

    def remember(self, fact: str):
        self.memory.remember_fact(fact)

    def get_profile(self) -> dict:
        return self.memory.world.get_user_profile()

    def trigger_training(self) -> dict:
        """Manually trigger a training cycle."""
        run = self.trainer.run_now()
        return {
            "status": run.status,
            "pairs_used": run.pairs_used,
            "notes": run.notes,
            "duration_s": round(
                (run.completed_at - run.started_at), 2
            ) if run.completed_at else 0,
        }

    def generate_training_data(self, n: int = 20) -> int:
        """Generate N self-play training pairs."""
        return self.trainer.generate_self_play(n)

    def training_status(self) -> dict:
        return self.trainer.status()

    # ── Private ────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model_loaded:
            return
        if not Path(self.model_path).exists():
            print("[LeanAI v3] No model — demo mode.")
            self._model_loaded = True
            return
        try:
            from llama_cpp import Llama
            print("[LeanAI v3] Loading model...")
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
            print("[LeanAI v3] Model loaded. Ready.")
        except ImportError:
            self._model_loaded = True

    def _generate_with_model(self, prompt: str, config: GenerationConfig) -> str:
        self._load_model()
        if self._model is None:
            return "Demo mode — run: python setup.py --download-model"
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
        # Build system message — include memory context inline, not as separate block
        system = (
            "You are LeanAI — a fast, accurate, and honest AI assistant. "
            "You run locally on the user's device. "
            "Be concise and direct. Answer only what was asked. "
            "Never repeat the question. Never add follow-up questions. "
            "Never generate new questions. Stop after answering."
        )
        if memory_context:
            # Trim context to essentials and embed in system message
            ctx_trimmed = memory_context[:300].replace("\n", " ")
            system += f" User context: {ctx_trimmed}"

        parts = [f"<|system|>\n{system}<|end|>\n"]
        for msg in history[-4:]:   # reduced to 4 turns — less context confusion
            role, content = msg["role"], msg["content"]
            # Truncate long history messages
            content = content[:300] if len(content) > 300 else content
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

    def _wrap(self, text, pair_id, cal, tier, latency,
              verified, corrected, verify_summary,
              claims_checked, claims_correct,
              mem_used, from_memory, quality,
              warning=None) -> LeanAIResponse:
        return LeanAIResponse(
            text=text, pair_id=pair_id,
            confidence=cal.calibrated,
            confidence_label=cal.label,
            confidence_bar=cal.bar,
            confidence_method=cal.method,
            tier_used=tier, latency_ms=round(latency, 1),
            verified=verified, corrected=corrected,
            verification_summary=verify_summary,
            claims_checked=claims_checked,
            claims_correct=claims_correct,
            memory_context_used=mem_used,
            answered_from_memory=from_memory,
            quality_score=quality,
            warning=warning,
        )

    def status(self) -> dict:
        mem = self.memory.stats()
        training = self.trainer.status()
        return {
            "phase": 3,
            "model_loaded": self._model is not None,
            "verifier": self.verifier.status,
            "memory": mem,
            "training": training,
        }

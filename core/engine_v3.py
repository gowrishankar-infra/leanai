"""
LeanAI Phase 3 Engine - Qwen2.5 Coder + Phi-3 support
Optimized for i7-11800H with 32GB RAM
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
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.05


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


def _get_active_model_path() -> str:
    config_file = Path.home() / ".leanai" / "active_model.txt"
    if config_file.exists():
        path = config_file.read_text().strip()
        if Path(path).exists():
            return path
    return str(Path.home() / ".leanai" / "models" / "phi3-mini-q4.gguf")


def _detect_prompt_format(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "qwen" in name:
        return "chatml"
    if "phi" in name:
        return "phi3"
    if "llama" in name:
        return "llama3"
    return "chatml"


def _optimal_threads(model_path: str) -> int:
    name = Path(model_path).name.lower()
    cpu_count = os.cpu_count() or 8
    if "7b" in name or "6b" in name or "8b" in name:
        return min(cpu_count, 16)
    if "33b" in name or "34b" in name:
        return min(cpu_count, 8)
    return min(cpu_count, 8)


class LeanAIEngineV3:

    def __init__(self, model_path=None, verbose=False, auto_train=True):
        self.model_path    = model_path or _get_active_model_path()
        self.prompt_format = _detect_prompt_format(self.model_path)
        self.n_threads     = _optimal_threads(self.model_path)
        self.verbose       = verbose
        self.router     = TaskRouter()
        self.watchdog   = MetaCognitiveWatchdog()
        self.scorer     = ConfidenceScoringEngine()
        self.calibrator = ConfidenceCalibrator()
        self.verifier   = Z3Verifier()
        self.memory     = HierarchicalMemoryV2()
        self.store      = TrainingDataStore()
        self.self_play  = EnhancedSelfPlayEngine()
        self.trainer    = ContinualTrainer(
            store=self.store,
            config=TrainingConfig(
                check_interval_minutes=30,
                min_pairs_to_train=50,
                self_play_batch_size=20,
            ),
        )
        self._model        = None
        self._model_loaded = False
        if auto_train:
            self.trainer.start()
        model_name = Path(self.model_path).name
        print("[LeanAI v3] Engine initialized.")
        print(f"[LeanAI v3] Model: {model_name}")
        print(f"[LeanAI v3] Format: {self.prompt_format}")
        print(f"[LeanAI v3] Threads: {self.n_threads}")
        print(f"[LeanAI v3] Memory: {self.memory.episodic.backend}")
        print(f"[LeanAI v3] Training: {self.store.stats()['total']} pairs")

    def generate(self, query, config=None):
        config = config or GenerationConfig()
        start  = time.time()
        decision = self.router.route(query, self.memory.working.current_tokens)

        if decision.tier == Tier.TINY:
            text = self._handle_tiny(query)
            latency = (time.time() - start) * 1000
            cal = self.calibrator.calibrate(0.99, text, "tiny", False, query)
            pair = self.store.add_pair(instruction=query, response=text,
                feedback=FeedbackSignal.EXCELLENT, confidence=cal.calibrated,
                verified=False, latency_ms=latency, tier_used="tiny")
            self.memory.record_exchange(query, text)
            return self._wrap(text, pair.id, cal, "tiny", latency,
                              False, False, "Not needed.", 0, 0, False, True, 1.0)

        memory_answer = self.memory.answer_from_memory(query)
        if memory_answer:
            latency = (time.time() - start) * 1000
            cal = self.calibrator.calibrate(0.95, memory_answer, "memory", True, query)
            self.memory.record_exchange(query, memory_answer)
            pair = self.store.add_pair(instruction=query, response=memory_answer,
                feedback=FeedbackSignal.EXCELLENT, confidence=cal.calibrated,
                verified=True, latency_ms=latency, tier_used="memory")
            return self._wrap(memory_answer, pair.id, cal, "memory (instant)", latency,
                              True, False, "From world model.", 0, 0, True, True, 1.0)

        mem_ctx  = self.memory.prepare_context(query)
        mem_used = bool(mem_ctx)
        prompt   = self._build_prompt(
            query, mem_ctx,
            self.memory.working.get_context_window(max_tokens=512),
        )
        response_text = self._generate_with_model(prompt, config)
        raw_score = self.scorer.score_from_text(response_text)
        cal = self.calibrator.calibrate(
            raw_score.overall, response_text, decision.tier.value, False, query)

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
            cal = self.calibrator.calibrate(
                raw_score.overall, response_text, decision.tier.value, verified, query)

        self.memory.record_exchange(query, response_text)
        latency = (time.time() - start) * 1000
        feedback = (FeedbackSignal.EXCELLENT if verified else
                    FeedbackSignal.GOOD if cal.calibrated > 0.7 else
                    FeedbackSignal.NEUTRAL)
        pair = self.store.add_pair(instruction=query, response=response_text,
            feedback=feedback, confidence=cal.calibrated, verified=verified,
            latency_ms=latency, tier_used=decision.tier.value)
        return self._wrap(response_text, pair.id, cal, decision.tier.value, latency,
            verified, corrected, verify_summary, claims_checked, claims_correct,
            mem_used, False, round(pair.quality_score, 3))

    def give_feedback(self, pair_id, good):
        self.store.update_feedback(
            pair_id, FeedbackSignal.EXCELLENT if good else FeedbackSignal.WRONG)

    def remember(self, fact):
        self.memory.remember_fact(fact)

    def get_profile(self):
        return self.memory.world.get_user_profile()

    def trigger_training(self):
        run = self.trainer.run_now()
        return {
            "status": run.status,
            "pairs_used": run.pairs_used,
            "notes": run.notes,
            "duration_s": round((run.completed_at - run.started_at), 2) if run.completed_at else 0,
        }

    def generate_training_data(self, n=20):
        return self.trainer.generate_self_play(n)

    def training_status(self):
        return self.trainer.status()

    def _load_model(self):
        if self._model_loaded:
            return
        if not Path(self.model_path).exists():
            print(f"[LeanAI v3] Model not found: {self.model_path}")
            self._model_loaded = True
            return
        try:
            from llama_cpp import Llama
            model_name = Path(self.model_path).name
            print(f"[LeanAI v3] Loading {model_name} ({self.n_threads} threads)...")
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=self.n_threads,
                n_batch=1024,
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

    def _generate_with_model(self, prompt, config):
        self._load_model()
        if self._model is None:
            return "Demo mode — run: python setup.py --download-model --model qwen25-coder"
        stop_tokens = [
            "<|im_end|>", "<|im_start|>",
            "<|user|>", "<|end|>", "<|assistant|>",
            "\nYou:", "\nHuman:", "\nUser:",
        ]
        result = self._model(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=stop_tokens,
            echo=False,
        )
        text = result["choices"][0]["text"].strip()
        for token in stop_tokens:
            text = text.split(token)[0].strip()
        return text

    def _build_prompt(self, query, memory_context, history):
        system = (
            "You are LeanAI — an expert coding AI assistant. "
            "You excel at Python, JavaScript, TypeScript, Go, Rust, SQL, bash, "
            "and all major frameworks. "
            "Be concise. Provide clean working code. "
            "Stop after answering. No follow-up questions."
        )
        if memory_context:
            ctx = memory_context[:200].replace("\n", " ")
            system = system + " Context: " + ctx

        if self.prompt_format == "chatml":
            parts = ["<|im_start|>system\n" + system + "<|im_end|>\n"]
            for msg in history[-4:]:
                role    = msg["role"]
                content = msg["content"]
                if len(content) > 400:
                    content = content[:400]
                if role == "user":
                    parts.append("<|im_start|>user\n" + content + "<|im_end|>\n")
                elif role == "assistant":
                    parts.append("<|im_start|>assistant\n" + content + "<|im_end|>\n")
            parts.append("<|im_start|>user\n" + query + "<|im_end|>\n<|im_start|>assistant\n")
        else:
            parts = ["<|system|>\n" + system + "<|end|>\n"]
            for msg in history[-4:]:
                role    = msg["role"]
                content = msg["content"]
                if len(content) > 300:
                    content = content[:300]
                if role == "user":
                    parts.append("<|user|>\n" + content + "<|end|>\n")
                elif role == "assistant":
                    parts.append("<|assistant|>\n" + content + "<|end|>\n")
            parts.append("<|user|>\n" + query + "<|end|>\n<|assistant|>\n")

        return "".join(parts)

    def _handle_tiny(self, query):
        import re
        import math as m
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

    def _wrap(self, text, pair_id, cal, tier, latency, verified, corrected,
              verify_summary, claims_checked, claims_correct, mem_used,
              from_memory, quality, warning=None):
        return LeanAIResponse(
            text=text, pair_id=pair_id,
            confidence=cal.calibrated, confidence_label=cal.label,
            confidence_bar=cal.bar, confidence_method=cal.method,
            tier_used=tier, latency_ms=round(latency, 1),
            verified=verified, corrected=corrected,
            verification_summary=verify_summary,
            claims_checked=claims_checked, claims_correct=claims_correct,
            memory_context_used=mem_used, answered_from_memory=from_memory,
            quality_score=quality, warning=warning,
        )

    def status(self):
        return {
            "phase": 3,
            "model": Path(self.model_path).name,
            "prompt_format": self.prompt_format,
            "threads": self.n_threads,
            "model_loaded": self._model is not None,
            "verifier": self.verifier.status,
            "memory": self.memory.stats(),
            "training": self.trainer.status(),
        }

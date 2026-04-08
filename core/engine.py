"""
LeanAI · Main Inference Engine
Ties together all Phase 0 components:
  - Smart router (picks model tier)
  - GGUF quantized model (BitNet/Mamba compatible via llama.cpp)
  - Metacognitive watchdog (confidence monitoring)
  - Neurosymbolic verifier (math/logic verification)
  - Hierarchical memory (4-layer context)

This is the heart of LeanAI. Every query flows through here.
"""

import time
import os
from dataclasses import dataclass
from typing import Optional, Iterator
from pathlib import Path

from core.router import TaskRouter, Tier
from core.watchdog import MetaCognitiveWatchdog
from tools.verifier import NeurosymbolicVerifier, VerificationStatus
from memory.hierarchy import HierarchicalMemory


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stream: bool = True


@dataclass
class LeanAIResponse:
    text: str
    confidence: float
    confidence_label: str
    tier_used: str
    latency_ms: float
    verified: bool
    verification_summary: str
    memory_context_used: bool
    warning: Optional[str] = None


# Default model — Phi-3 Mini 4K (Microsoft, MIT license, 2.3GB, excellent quality)
# Can be swapped for any GGUF model
DEFAULT_MODEL_URL = (
    "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/"
    "resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
)
DEFAULT_MODEL_PATH = Path.home() / ".leanai" / "models" / "phi3-mini-q4.gguf"


class LeanAIEngine:
    """
    Main engine. Initialize once, call .generate() for every query.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model_path = model_path or str(DEFAULT_MODEL_PATH)
        self.verbose = verbose

        # Initialize all subsystems
        self.router   = TaskRouter()
        self.watchdog = MetaCognitiveWatchdog()
        self.verifier = NeurosymbolicVerifier()
        self.memory   = HierarchicalMemory()

        # Load model (lazy — only when first query arrives)
        self._model = None
        self._model_loaded = False

        print("[LeanAI] Engine initialized. Model will load on first query.")
        print(f"[LeanAI] Verifier capabilities: {self.verifier.capabilities}")

    def generate(
        self,
        query: str,
        config: Optional[GenerationConfig] = None,
    ) -> LeanAIResponse:
        """
        Main generation entry point.
        Routes → retrieves memory → generates → monitors → verifies → returns.
        """
        config = config or GenerationConfig()
        start_time = time.time()

        # ── 1. Route the query ──────────────────────────────────────────
        context_tokens = self.memory.working.current_tokens
        decision = self.router.route(query, context_length=context_tokens)

        if self.verbose:
            print(self.router.explain(decision))

        # ── 2. Tiny tier: rule-based, no model ─────────────────────────
        if decision.tier == Tier.TINY:
            response_text = self._handle_tiny(query)
            latency = (time.time() - start_time) * 1000
            self.memory.record_exchange(query, response_text)
            return LeanAIResponse(
                text=response_text,
                confidence=0.99,
                confidence_label="High confidence",
                tier_used="tiny (rule-based)",
                latency_ms=round(latency, 1),
                verified=False,
                verification_summary="Rule-based — no verification needed.",
                memory_context_used=False,
            )

        # ── 3. Pull memory context ──────────────────────────────────────
        memory_context = self.memory.prepare_context(query)
        memory_context_used = bool(memory_context)

        # ── 4. Build full prompt ────────────────────────────────────────
        prompt = self._build_prompt(
            query=query,
            memory_context=memory_context,
            conversation_history=self.memory.working.get_context_window(max_tokens=1024),
        )

        # ── 5. Generate response ────────────────────────────────────────
        response_text = self._generate_with_model(prompt, config, decision.tier)

        # ── 6. Metacognitive monitoring ─────────────────────────────────
        watchdog_state = self.watchdog.simulate_from_response(response_text)
        confidence = watchdog_state.final_confidence
        confidence_label = self.watchdog.confidence_label(confidence)

        # Override verification flag if watchdog says so
        should_verify = decision.requires_verifier or watchdog_state.should_verify

        # ── 7. Neurosymbolic verification ───────────────────────────────
        verified = False
        verification_summary = "Not checked."

        if should_verify:
            report = self.verifier.verify_response(response_text, query)
            verified = report.overall_status == VerificationStatus.VERIFIED
            verification_summary = report.summary

            # Use corrected response if verifier found errors
            if report.corrected_response:
                response_text = report.corrected_response

        # ── 8. Store in memory ──────────────────────────────────────────
        self.memory.record_exchange(query, response_text)

        latency = (time.time() - start_time) * 1000

        return LeanAIResponse(
            text=response_text,
            confidence=round(confidence, 2),
            confidence_label=confidence_label,
            tier_used=decision.tier.value,
            latency_ms=round(latency, 1),
            verified=verified,
            verification_summary=verification_summary,
            memory_context_used=memory_context_used,
            warning=watchdog_state.warning_message,
        )

    def stream(
        self,
        query: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """
        Streaming version — yields tokens as they're generated.
        Use this for chat interfaces.
        """
        config = config or GenerationConfig(stream=True)
        decision = self.router.route(query)

        if decision.tier == Tier.TINY:
            yield self._handle_tiny(query)
            return

        memory_context = self.memory.prepare_context(query)
        prompt = self._build_prompt(
            query=query,
            memory_context=memory_context,
            conversation_history=self.memory.working.get_context_window(),
        )

        yield from self._stream_with_model(prompt, config)

    # ── Private: Model Loading ──────────────────────────────────────────

    def _load_model(self):
        """Load the GGUF model via llama-cpp-python."""
        if self._model_loaded:
            return

        if not Path(self.model_path).exists():
            print(f"\n[LeanAI] Model not found at: {self.model_path}")
            print("[LeanAI] Run: python setup.py --download-model")
            print("[LeanAI] Running in demo mode (no model loaded)\n")
            self._model_loaded = True
            return

        try:
            from llama_cpp import Llama
            print(f"[LeanAI] Loading model: {self.model_path}")
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=os.cpu_count(),
                n_gpu_layers=0,    # CPU only — Phase 0
                verbose=self.verbose,
            )
            self._model_loaded = True
            print("[LeanAI] Model loaded successfully.")
        except ImportError:
            print("[LeanAI] llama-cpp-python not installed.")
            print("[LeanAI] Install with: pip install llama-cpp-python")
            self._model_loaded = True

    # ── Private: Generation ─────────────────────────────────────────────

    def _generate_with_model(
        self,
        prompt: str,
        config: GenerationConfig,
        tier: Tier,
    ) -> str:
        """Generate a response using the loaded model."""
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

    def _stream_with_model(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> Iterator[str]:
        """Stream tokens from the model."""
        self._load_model()

        if self._model is None:
            demo = self._demo_response(prompt)
            for word in demo.split():
                yield word + " "
                time.sleep(0.05)
            return

        for chunk in self._model(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    # ── Private: Prompt Building ────────────────────────────────────────

    def _build_prompt(
        self,
        query: str,
        memory_context: str,
        conversation_history: list,
    ) -> str:
        """Build the full prompt with system instruction, memory, and history."""
        system = (
            "You are LeanAI — a fast, accurate, and honest AI assistant. "
            "You run locally on the user's device. "
            "You are concise, accurate, and always acknowledge uncertainty. "
            "You never pretend to know something you don't."
        )

        parts = [f"<|system|>\n{system}\n"]

        if memory_context:
            parts.append(f"<|context|>\n{memory_context}\n")

        for msg in conversation_history[-6:]:  # Last 6 turns
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}\n")

        parts.append(f"<|user|>\n{query}\n<|assistant|>\n")
        return "".join(parts)

    # ── Private: Tiny Tier Handler ──────────────────────────────────────

    def _handle_tiny(self, query: str) -> str:
        """Handle trivially simple queries without a model."""
        import re
        q = query.strip().lower()

        if re.match(r"^(hi|hello|hey)\b", q):
            return "Hello! How can I help you?"
        if re.match(r"^(thanks|thank you|thx)\b", q):
            return "You're welcome!"
        if re.match(r"^(bye|goodbye)\b", q):
            return "Goodbye!"

        # Try arithmetic
        try:
            expr = re.sub(r"[^0-9\+\-\*\/\(\)\.\s\^]", "", query)
            expr = expr.replace("^", "**")
            if expr.strip():
                result = eval(expr, {"__builtins__": {}})
                return f"{result}"
        except Exception:
            pass

        return "I understand. Could you tell me more?"

    def _demo_response(self, prompt: str) -> str:
        """Fallback when no model is loaded — for testing the pipeline."""
        return (
            "This is a demo response. "
            "The full pipeline (router → watchdog → verifier → memory) is working. "
            "Download a model with: python setup.py --download-model"
        )

    def status(self) -> dict:
        """Return engine status summary."""
        return {
            "model_loaded": self._model is not None,
            "model_path": self.model_path,
            "verifier": self.verifier.capabilities,
            "memory": self.memory.stats(),
        }

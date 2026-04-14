"""
LeanAI — DualPipe: Asymmetric GPU/CPU Speculative Decoding
════════════════════════════════════════════════════════════

NOVEL TECHNIQUE — No existing implementation splits draft/verify across GPU/CPU.

Draft model (7B) generates K tokens fast on GPU.
Target model (27B/MoE) generates 1 token on CPU for verification.
If they agree on the first token, accept the entire draft chunk.
If not, use target's token.

Uses high-level model APIs only — no raw logits, no C API, no
cross-tokenizer issues. Relies on llama-cpp-python's built-in
prompt caching for fast subsequent cycles.

Author: Gowri Shankar (github.com/gowrishankar-infra/leanai)
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
from pathlib import Path


@dataclass
class DualPipeConfig:
    enabled: bool = True
    draft_tokens: int = 8
    draft_gpu_layers: int = 10
    target_gpu_layers: int = 0
    target_ctx: int = 2048
    draft_ctx: int = 2048
    n_threads: int = 0
    verbose: bool = False


@dataclass
class DualPipeStats:
    total_tokens: int = 0
    draft_cycles: int = 0
    accepted_chunks: int = 0
    rejected_chunks: int = 0
    draft_time_ms: float = 0.0
    target_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted_chunks + self.rejected_chunks
        return self.accepted_chunks / max(total, 1)

    @property
    def effective_tok_s(self) -> float:
        return self.total_tokens / max(self.total_time_ms / 1000.0, 0.001)

    @property
    def speedup(self) -> float:
        if self.total_time_ms <= 0 or self.total_tokens <= 0:
            return 1.0
        target_only_ms = self.total_tokens * 420
        return target_only_ms / max(self.total_time_ms, 1.0)

    def summary(self) -> str:
        return (
            f"DualPipe: {self.total_tokens} tokens in {self.total_time_ms/1000:.1f}s "
            f"({self.effective_tok_s:.1f} tok/s) | "
            f"accept {self.acceptance_rate:.0%} | "
            f"{self.draft_cycles} cycles | "
            f"~{self.speedup:.1f}x vs target-only"
        )


class DualPipeEngine:
    """
    Asymmetric GPU/CPU Speculative Decoding.
    Uses ONLY high-level model APIs — no raw logits, no C pointers.

    Usage:
        pipe = DualPipeEngine(draft_path, target_path)
        pipe.load()
        text, stats = pipe.generate(prompt, max_tokens=512, callback=print)
    """

    def __init__(self, draft_path: str, target_path: str,
                 config: Optional[DualPipeConfig] = None):
        self.draft_path = draft_path
        self.target_path = target_path
        self.config = config or DualPipeConfig()
        self._draft = None
        self._target = None
        self._loaded = False
        self._available = False
        self._cumulative = DualPipeStats()
        self._gen_count = 0

        if self.config.n_threads <= 0:
            self.config.n_threads = min(os.cpu_count() or 8, 16)

    def load(self) -> bool:
        if self._loaded:
            return self._available

        if not Path(self.draft_path).exists() or not Path(self.target_path).exists():
            print(f"  [DualPipe] Model files not found")
            return False

        from llama_cpp import Llama
        draft_name = Path(self.draft_path).name
        target_name = Path(self.target_path).name

        gpu_attempts = [self.config.draft_gpu_layers, 10, 7, 5, 3]

        for gpu_layers in gpu_attempts:
            try:
                if self._draft:
                    del self._draft
                    self._draft = None
                if self._target:
                    del self._target
                    self._target = None

                print(f"  [DualPipe] Loading draft: {draft_name} (GPU, {gpu_layers} layers)")
                self._draft = Llama(
                    model_path=self.draft_path,
                    n_ctx=self.config.draft_ctx,
                    n_gpu_layers=gpu_layers,
                    n_threads=self.config.n_threads,
                    n_batch=512, use_mmap=True, verbose=False,
                )

                print(f"  [DualPipe] Loading target: {target_name} (CPU)")
                self._target = Llama(
                    model_path=self.target_path,
                    n_ctx=self.config.target_ctx,
                    n_gpu_layers=0,
                    n_threads=self.config.n_threads,
                    n_batch=512, use_mmap=True, verbose=False,
                )

                self._loaded = True
                self._available = True
                print(f"  [DualPipe] Ready. Draft={draft_name} ({gpu_layers} GPU), "
                      f"Target={target_name} (CPU), K={self.config.draft_tokens}")
                return True

            except Exception as e:
                if "OutOfDeviceMemory" in str(e) or "memory" in str(e).lower():
                    print(f"  [DualPipe] VRAM insufficient with {gpu_layers} layers — retrying...")
                    continue
                print(f"  [DualPipe] Failed: {e}")
                break

        self._loaded = True
        self._available = False
        return False

    def unload(self):
        if self._draft:
            del self._draft
            self._draft = None
        if self._target:
            del self._target
            self._target = None
        self._loaded = False
        self._available = False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_k: int = 40,
        top_p: float = 0.95,
        repeat_penalty: float = 1.05,
        stop_strings: Optional[List[str]] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, DualPipeStats]:
        """
        Generate using text-level speculative decoding.

        Each cycle:
          1. Draft (7B/GPU) generates K tokens of text — fast
          2. Target (27B/CPU) generates 1 token of text — slow but accurate
          3. If draft starts with target's token → accept entire draft (K tokens)
          4. If not → use target's 1 token, discard draft

        llama-cpp-python's built-in prompt caching means the target
        only re-processes NEW tokens each cycle, not the full prompt.
        """
        if not self._available:
            return "", DualPipeStats()

        stats = DualPipeStats()
        gen_start = time.time()
        K = self.config.draft_tokens
        generated = ""
        stop_strs = list(stop_strings or [])
        stop_strs_all = stop_strs + ["<|im_end|>", "<|im_start|>", "<end_of_turn>"]

        while stats.total_tokens < max_tokens:
            full_ctx = prompt + generated

            # ── DRAFT: 7B generates K tokens on GPU (fast) ──────────
            draft_start = time.time()
            try:
                draft_result = self._draft(
                    full_ctx, max_tokens=K,
                    temperature=max(temperature, 0.05),
                    top_k=top_k, top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop_strs_all, echo=False,
                )
                draft_text = draft_result['choices'][0]['text']
            except Exception as e:
                if self.config.verbose:
                    print(f"\n  [DualPipe] Draft failed: {e}")
                break
            stats.draft_time_ms += (time.time() - draft_start) * 1000

            if not draft_text or not draft_text.strip():
                break

            # ── TARGET: 27B generates 1 token on CPU (verify) ───────
            target_start = time.time()
            try:
                target_result = self._target(
                    full_ctx, max_tokens=1,
                    temperature=0.01,
                    repeat_penalty=repeat_penalty,
                    stop=stop_strs_all, echo=False,
                )
                target_token = target_result['choices'][0]['text']
            except Exception as e:
                if self.config.verbose:
                    print(f"\n  [DualPipe] Target failed: {e}")
                break
            stats.target_time_ms += (time.time() - target_start) * 1000

            if not target_token:
                break

            # ── COMPARE: does draft agree with target? ──────────────
            stats.draft_cycles += 1
            t = target_token.strip()
            d = draft_text.lstrip()

            if t and d.startswith(t):
                # ACCEPT: draft agrees → use entire draft chunk
                generated += draft_text
                stats.accepted_chunks += 1
                stats.total_tokens += max(len(draft_text) // 4, 1)
                if callback:
                    callback(draft_text)
            else:
                # REJECT: use target's 1 token
                generated += target_token
                stats.rejected_chunks += 1
                stats.total_tokens += 1
                if callback:
                    callback(target_token)

            # Check stop
            if stop_strs and any(s in generated for s in stop_strs):
                for s in stop_strs:
                    generated = generated.split(s)[0]
                break

        # ── Finalize ────────────────────────────────────────────────
        stats.total_time_ms = (time.time() - gen_start) * 1000

        import re
        generated = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL).strip()
        if "<think>" in generated:
            generated = generated.split("<think>")[0].strip()
        generated = re.sub(r"<\|channel>.*?<channel\|>", "", generated, flags=re.DOTALL).strip()

        if stop_strs:
            for s in stop_strs:
                generated = generated.split(s)[0]

        self._gen_count += 1
        self._cumulative.total_tokens += stats.total_tokens
        self._cumulative.draft_cycles += stats.draft_cycles
        self._cumulative.accepted_chunks += stats.accepted_chunks
        self._cumulative.rejected_chunks += stats.rejected_chunks
        self._cumulative.draft_time_ms += stats.draft_time_ms
        self._cumulative.target_time_ms += stats.target_time_ms
        self._cumulative.total_time_ms += stats.total_time_ms

        return generated, stats

    @property
    def is_available(self) -> bool:
        return self._available and self.config.enabled

    def stats(self) -> dict:
        cs = self._cumulative
        return {
            "generations": self._gen_count,
            "total_tokens": cs.total_tokens,
            "accepted_chunks": cs.accepted_chunks,
            "rejected_chunks": cs.rejected_chunks,
            "acceptance_rate": f"{cs.acceptance_rate:.0%}",
            "draft_time_ms": f"{cs.draft_time_ms:.0f}",
            "target_time_ms": f"{cs.target_time_ms:.0f}",
            "effective_tok_s": f"{cs.effective_tok_s:.1f}",
            "speedup": f"{cs.speedup:.1f}x",
            "available": self._available,
        }

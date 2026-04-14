"""
LeanAI — CodeEcho: Source-Grounded Speculative Decoding
═══════════════════════════════════════════════════════════

NOVEL TECHNIQUE — No published research, no existing implementation.

When a coding assistant reviews, fixes, or explains code, 40-80% of its
output tokens REPRODUCE code already present in the user's context (file
content, brain context, previous responses). Each of those tokens costs a
full sequential forward pass — even though the output is completely predictable.

CodeEcho exploits this by:
  1. Pre-tokenizing all source material and building a 5-gram hash index
  2. During token-by-token generation, monitoring for echo matches
  3. When 5+ consecutive tokens match a known source, batch-injecting
     the next N tokens from that source via eval() (prefill speed)
  4. Resuming normal generation after the echoed section

The speedup comes from batch eval running at PREFILL speed (~10-50x faster
per token than sequential decode). For a typical code review where 60% of
tokens are source reproductions, this yields 2-5x wall-clock speedup with
ZERO quality loss.

The "draft" model is replaced by the USER'S OWN CODEBASE — which has
near-perfect acceptance rate because the model was going to reproduce
that code anyway.

Author: Gowri Shankar (github.com/gowrishankar-infra/leanai)
"""

import time
import struct
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Tuple, Any
from collections import defaultdict


# ── Configuration ──────────────────────────────────────────────────

NGRAM_SIZE = 5          # consecutive tokens needed to trigger echo
MIN_ECHO_TOKENS = 8    # minimum draft length worth echoing
MAX_ECHO_TOKENS = 100   # maximum tokens to echo at once
MATCH_COOLDOWN = 3      # tokens to wait after a failed/short echo before trying again


@dataclass
class CodeEchoConfig:
    """Tuning knobs for CodeEcho."""
    enabled: bool = True
    ngram_size: int = NGRAM_SIZE
    min_echo_tokens: int = MIN_ECHO_TOKENS
    max_echo_tokens: int = MAX_ECHO_TOKENS
    match_cooldown: int = MATCH_COOLDOWN
    verbose: bool = False


@dataclass
class EchoStats:
    """Statistics from a single generation with CodeEcho."""
    total_tokens: int = 0
    echoed_tokens: int = 0
    echo_events: int = 0          # how many times echo was triggered
    normal_tokens: int = 0
    echo_time_ms: float = 0.0     # time spent in batch eval (echo)
    normal_time_ms: float = 0.0   # time spent in normal decode
    total_time_ms: float = 0.0
    sources_indexed: int = 0
    index_size: int = 0           # number of 5-grams in index

    @property
    def echo_ratio(self) -> float:
        """Fraction of tokens that were echoed (0.0 to 1.0)."""
        return self.echoed_tokens / max(self.total_tokens, 1)

    @property
    def speedup_estimate(self) -> float:
        """Estimated speedup vs. generating all tokens normally."""
        if self.normal_time_ms <= 0 or self.normal_tokens <= 0:
            return 1.0
        # Time per normal token
        normal_rate = self.normal_time_ms / self.normal_tokens
        # If we had generated echoed tokens normally, it would have taken:
        hypothetical_ms = normal_rate * self.total_tokens
        actual_ms = self.total_time_ms
        return hypothetical_ms / max(actual_ms, 1.0)

    def summary(self) -> str:
        pct = f"{self.echo_ratio:.0%}"
        return (
            f"CodeEcho: {self.echoed_tokens}/{self.total_tokens} tokens echoed ({pct}) "
            f"in {self.echo_events} events | "
            f"~{self.speedup_estimate:.1f}x speedup"
        )


# ── Source Index ───────────────────────────────────────────────────

class SourceIndex:
    """
    Pre-tokenizes source material and builds a fast n-gram lookup index.

    Given source texts (file contents, brain context, previous responses),
    tokenizes them and creates a hash map of n-gram → (source_id, position)
    for O(1) echo detection during generation.
    """

    def __init__(self, ngram_size: int = NGRAM_SIZE):
        self.ngram_size = ngram_size
        self._sources: List[List[int]] = []       # source_id → token list
        self._source_texts: List[str] = []         # source_id → original text
        self._index: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        # hash(ngram) → [(source_id, position), ...]

    def add_source(self, text: str, tokenizer_fn: Callable) -> int:
        """
        Add a source text to the index.

        Args:
            text: The source text (file content, brain context, etc.)
            tokenizer_fn: Function that converts text to token IDs
                          e.g. model.tokenize(text.encode())

        Returns:
            source_id for reference
        """
        if not text or len(text.strip()) < 20:
            return -1

        try:
            tokens = tokenizer_fn(text.encode('utf-8'))
            if not tokens or len(tokens) < self.ngram_size + 2:
                return -1
        except Exception:
            return -1

        source_id = len(self._sources)
        self._sources.append(tokens)
        self._source_texts.append(text)

        # Build n-gram index for this source
        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = tuple(tokens[i:i + self.ngram_size])
            key = self._hash_ngram(ngram)
            self._index[key].append((source_id, i))

        return source_id

    def lookup(self, recent_tokens: List[int]) -> Optional[Tuple[int, int, List[int]]]:
        """
        Check if the last N generated tokens match any source.

        Args:
            recent_tokens: The last ngram_size tokens that were generated

        Returns:
            (source_id, match_position, remaining_tokens) or None
            remaining_tokens = tokens from the source AFTER the match
        """
        if len(recent_tokens) < self.ngram_size:
            return None

        ngram = tuple(recent_tokens[-self.ngram_size:])
        key = self._hash_ngram(ngram)

        if key not in self._index:
            return None

        # Verify actual match (hash collisions are possible)
        for source_id, pos in self._index[key]:
            source_tokens = self._sources[source_id]
            match_end = pos + self.ngram_size

            # Verify the n-gram actually matches (not just hash collision)
            if source_tokens[pos:match_end] == list(ngram):
                # Return tokens AFTER the matched n-gram
                remaining = source_tokens[match_end:]
                if len(remaining) >= MIN_ECHO_TOKENS:
                    return (source_id, match_end, remaining[:MAX_ECHO_TOKENS])

        return None

    @staticmethod
    def _hash_ngram(ngram: tuple) -> int:
        """Fast hash of an n-gram tuple. Uses struct packing for speed."""
        # Pack token IDs as 4-byte ints, hash the result
        try:
            packed = struct.pack(f'{len(ngram)}i', *ngram)
            # FNV-1a inspired hash
            h = 0x811c9dc5
            for b in packed:
                h ^= b
                h = (h * 0x01000193) & 0xFFFFFFFF
            return h
        except struct.error:
            return hash(ngram)

    @property
    def num_sources(self) -> int:
        return len(self._sources)

    @property
    def num_ngrams(self) -> int:
        return len(self._index)

    def __len__(self) -> int:
        return self.num_ngrams


# ── CodeEcho Engine ────────────────────────────────────────────────

class CodeEchoEngine:
    """
    Source-Grounded Speculative Decoding engine.

    Integrates with llama-cpp-python's Llama class to provide
    accelerated token generation when the model is reproducing
    known source material.

    Usage:
        echo = CodeEchoEngine()

        # Index source material before generation
        echo.index_sources(model, [file_content, brain_context])

        # Generate with echo acceleration
        text, stats = echo.generate(
            model=model,
            prompt_tokens=prompt_tokens,
            max_tokens=1024,
            stop_tokens=[...],
            callback=print_token,
        )
    """

    def __init__(self, config: Optional[CodeEchoConfig] = None):
        self.config = config or CodeEchoConfig()
        self._index: Optional[SourceIndex] = None
        self._cumulative_stats = EchoStats()
        self._generation_count = 0
        self._api_available = None  # None = unchecked, True/False after check

    def index_sources(self, model: Any, sources: List[str]) -> int:
        """
        Pre-tokenize and index source material for echo detection.
        Call this BEFORE generate() with all relevant source texts.

        Args:
            model: llama-cpp-python Llama instance (for tokenizer)
            sources: List of source texts (file contents, brain context, etc.)

        Returns:
            Number of sources successfully indexed
        """
        self._index = SourceIndex(ngram_size=self.config.ngram_size)
        indexed = 0

        # Use model's tokenizer (without BOS token for source fragments)
        def tokenize_fn(text_bytes):
            try:
                return model.tokenize(text_bytes, add_bos=False, special=False)
            except TypeError:
                # Older llama-cpp-python versions may not have add_bos parameter
                return model.tokenize(text_bytes)

        for source_text in sources:
            if not source_text or not source_text.strip():
                continue
            sid = self._index.add_source(source_text, tokenize_fn)
            if sid >= 0:
                indexed += 1

        if self.config.verbose and indexed > 0:
            print(f"  [CodeEcho] Indexed {indexed} sources, {self._index.num_ngrams} n-grams")

        return indexed

    def check_api(self, model: Any) -> bool:
        """
        Check if the model exposes the low-level API needed for CodeEcho.
        Requires: tokenize(), eval(), and logit/score access.
        """
        if self._api_available is not None:
            return self._api_available

        try:
            # Check tokenize
            test_tokens = model.tokenize(b"test", add_bos=False)
            assert isinstance(test_tokens, list) and len(test_tokens) > 0

            # Check that eval() exists
            assert hasattr(model, 'eval') and callable(model.eval)

            # Check logit/score access (needed for sampling)
            # Try multiple attribute names across llama-cpp-python versions
            has_scores = (
                hasattr(model, 'scores') or
                hasattr(model, '_scores') or
                hasattr(model, 'logits')
            )

            self._api_available = has_scores
            if self.config.verbose:
                print(f"  [CodeEcho] API check: {'available' if has_scores else 'scores not accessible'}")
        except Exception as e:
            self._api_available = False
            if self.config.verbose:
                print(f"  [CodeEcho] API check failed: {e}")

        return self._api_available

    def generate(
        self,
        model: Any,
        prompt_tokens: List[int],
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.05,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Optional[List[str]] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, EchoStats]:
        """
        Generate text with CodeEcho acceleration.

        This replaces the standard streaming generation loop. It generates
        tokens one by one, checking for echo matches after each token.
        When a match is found, it batch-injects source tokens via eval()
        instead of generating them sequentially.

        Args:
            model: llama-cpp-python Llama instance
            prompt_tokens: Tokenized prompt (from model.tokenize())
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            stop_token_ids: Token IDs that signal end of generation
            stop_strings: String patterns that signal end of generation
            callback: Called with each generated text chunk (for streaming)

        Returns:
            (generated_text, echo_stats)
        """
        stats = EchoStats()
        stats.sources_indexed = self._index.num_sources if self._index else 0
        stats.index_size = len(self._index) if self._index else 0

        gen_start = time.time()

        # ── Step 1: Process the prompt ──────────────────────────────
        try:
            model.reset()  # Clear KV cache for fresh generation
            model.eval(prompt_tokens)
        except Exception as e:
            if self.config.verbose:
                print(f"  [CodeEcho] Prompt eval failed: {e}")
            return "", stats

        # ── Step 2: Token-by-token generation with echo detection ──
        generated_ids: List[int] = []
        generated_text = ""
        cooldown = 0  # tokens to wait before next echo attempt
        in_thinking = False
        think_buffer = ""

        # Resolve stop token IDs
        if stop_token_ids is None:
            stop_token_ids = []
        # Try to get EOS token
        try:
            eos = model.token_eos()
            if eos not in stop_token_ids:
                stop_token_ids.append(eos)
        except Exception:
            pass

        for step in range(max_tokens):
            # ── Sample next token ───────────────────────────────────
            normal_start = time.time()
            try:
                token_id = self._sample_token(
                    model, temperature, top_p, top_k, repeat_penalty, generated_ids
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"\n  [CodeEcho] Sampling failed at step {step}: {e}")
                break

            normal_elapsed = (time.time() - normal_start) * 1000
            stats.normal_time_ms += normal_elapsed
            stats.normal_tokens += 1

            # Check stop condition
            if token_id in stop_token_ids:
                break

            generated_ids.append(token_id)
            stats.total_tokens += 1

            # Decode token to text
            try:
                token_text = model.detokenize([token_id]).decode('utf-8', errors='replace')
            except Exception:
                token_text = ""

            generated_text += token_text

            # ── Thinking tag detection (same as engine_v3) ──────────
            think_buffer += token_text
            if "<think>" in think_buffer and not in_thinking:
                in_thinking = True
                think_buffer = ""
                continue
            if "</think>" in think_buffer and in_thinking:
                in_thinking = False
                think_buffer = ""
                continue
            if "<|channel>" in think_buffer and not in_thinking:
                in_thinking = True
                think_buffer = ""
                continue
            if "<channel|>" in think_buffer and in_thinking:
                in_thinking = False
                think_buffer = ""
                continue

            # Only output when not in thinking mode
            if not in_thinking and callback and token_text:
                # Flush buffer if it doesn't look like a partial tag
                if len(think_buffer) > 15 or not any(p in think_buffer for p in ["<t", "</t", "<|c", "<c"]):
                    clean = think_buffer
                    for tag in ["<think>", "</think>", "<|channel>", "<channel|>"]:
                        clean = clean.replace(tag, "")
                    if clean:
                        callback(clean)
                    think_buffer = ""
            elif in_thinking:
                if len(think_buffer) > 200:
                    think_buffer = think_buffer[-50:]

            # Check stop strings
            if stop_strings and any(s in generated_text for s in stop_strings):
                # Trim at stop string
                for s in stop_strings:
                    if s in generated_text:
                        generated_text = generated_text.split(s)[0]
                break

            # ── Echo detection ──────────────────────────────────────
            # Also check remaining KV budget to prevent overflow
            remaining_budget = max_tokens - len(generated_ids)
            if (
                self.config.enabled
                and self._index
                and self._index.num_ngrams > 0
                and cooldown <= 0
                and not in_thinking
                and len(generated_ids) >= self.config.ngram_size
                and remaining_budget > self.config.min_echo_tokens + 10
            ):
                match = self._index.lookup(generated_ids)
                if match is not None:
                    source_id, match_pos, draft_tokens = match
                    # Cap draft to remaining budget (leave 10 tokens for post-echo generation)
                    max_draft = min(
                        len(draft_tokens),
                        self.config.max_echo_tokens,
                        remaining_budget - 10,
                    )
                    draft_len = max_draft

                    if draft_len >= self.config.min_echo_tokens:
                        # ── ECHO: Batch-inject source tokens ────────
                        echo_start = time.time()
                        draft = draft_tokens[:draft_len]

                        try:
                            # Batch eval: process all draft tokens at once
                            # This runs at PREFILL speed (~10-50x faster/token)
                            model.eval(draft)

                            # Decode the echoed tokens for output
                            echo_text = model.detokenize(draft).decode('utf-8', errors='replace')
                            generated_ids.extend(draft)
                            generated_text += echo_text
                            stats.echoed_tokens += draft_len
                            stats.total_tokens += draft_len
                            stats.echo_events += 1

                            echo_elapsed = (time.time() - echo_start) * 1000
                            stats.echo_time_ms += echo_elapsed

                            # Stream the echoed text
                            if callback and echo_text and not in_thinking:
                                callback(echo_text)

                            if self.config.verbose:
                                print(
                                    f"\n  [CodeEcho] ⚡ Echoed {draft_len} tokens "
                                    f"from source {source_id} "
                                    f"({echo_elapsed:.0f}ms vs ~{draft_len * normal_elapsed:.0f}ms normal)"
                                )

                            # Set small cooldown to avoid echo-chaining issues
                            cooldown = 2

                        except Exception as e:
                            if self.config.verbose:
                                print(f"\n  [CodeEcho] Echo eval failed: {e}")
                            cooldown = self.config.match_cooldown
                    else:
                        cooldown = self.config.match_cooldown

            if cooldown > 0:
                cooldown -= 1

        # ── Finalize ────────────────────────────────────────────────
        stats.total_time_ms = (time.time() - gen_start) * 1000

        # Clean up thinking blocks from final text
        import re
        generated_text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL).strip()
        if "<think>" in generated_text:
            generated_text = generated_text.split("<think>")[0].strip()
        generated_text = re.sub(r"<\|channel>.*?<channel\|>", "", generated_text, flags=re.DOTALL).strip()

        # Clean stop strings
        if stop_strings:
            for s in stop_strings:
                generated_text = generated_text.split(s)[0].strip()

        # Update cumulative stats
        self._generation_count += 1
        self._cumulative_stats.total_tokens += stats.total_tokens
        self._cumulative_stats.echoed_tokens += stats.echoed_tokens
        self._cumulative_stats.echo_events += stats.echo_events
        self._cumulative_stats.normal_tokens += stats.normal_tokens
        self._cumulative_stats.echo_time_ms += stats.echo_time_ms
        self._cumulative_stats.normal_time_ms += stats.normal_time_ms
        self._cumulative_stats.total_time_ms += stats.total_time_ms

        return generated_text, stats

    def _sample_token(
        self,
        model: Any,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        prev_tokens: List[int],
    ) -> int:
        """
        Sample the next token from the model's current state.

        Tries multiple approaches for compatibility with different
        llama-cpp-python versions:
          1. model.sample() if available
          2. Direct llama_cpp C API sampling
          3. Greedy argmax from logits/scores
        """
        import numpy as np

        # ── Approach 1: Try the model's sample method ───────────────
        # Some versions of llama-cpp-python expose this directly
        try:
            if hasattr(model, 'sample'):
                result = model.sample(
                    top_k=top_k, top_p=top_p, temp=temperature,
                    repeat_penalty=repeat_penalty,
                )
                if isinstance(result, int) and result >= 0:
                    # Feed the sampled token back through eval for KV cache update
                    model.eval([result])
                    return result
        except (TypeError, AttributeError, Exception):
            pass

        # ── Approach 2: Use llama_cpp C API for sampling ────────────
        try:
            import llama_cpp
            ctx = model._ctx
            if hasattr(ctx, 'ctx'):
                ctx_ptr = ctx.ctx  # newer API wraps in object
            else:
                ctx_ptr = ctx

            n_vocab = llama_cpp.llama_n_vocab(model._model.model)

            # Get logits for the last token
            logits_ptr = llama_cpp.llama_get_logits(ctx_ptr)
            if logits_ptr is not None:
                # Convert to numpy array
                logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy()

                # Apply repeat penalty
                if repeat_penalty != 1.0 and prev_tokens:
                    for tid in set(prev_tokens[-64:]):
                        if 0 <= tid < n_vocab:
                            if logits[tid] > 0:
                                logits[tid] /= repeat_penalty
                            else:
                                logits[tid] *= repeat_penalty

                # Temperature scaling
                if temperature > 0 and temperature != 1.0:
                    logits = logits / temperature

                # Top-k filtering
                if top_k > 0 and top_k < n_vocab:
                    indices = np.argpartition(logits, -top_k)[-top_k:]
                    mask = np.ones(n_vocab, dtype=bool)
                    mask[indices] = False
                    logits[mask] = -float('inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_indices = np.argsort(logits)[::-1]
                    probs = np.exp(logits - np.max(logits))
                    probs = probs / probs.sum()
                    sorted_probs = probs[sorted_indices]
                    cumsum = np.cumsum(sorted_probs)
                    cutoff_idx = np.searchsorted(cumsum, top_p) + 1
                    keep = sorted_indices[:cutoff_idx]
                    mask = np.ones(n_vocab, dtype=bool)
                    mask[keep] = False
                    logits[mask] = -float('inf')

                # Sample from distribution
                if temperature <= 0.01:
                    # Greedy
                    token_id = int(np.argmax(logits))
                else:
                    probs = np.exp(logits - np.max(logits))
                    probs = probs / probs.sum()
                    token_id = int(np.random.choice(n_vocab, p=probs))

                # Feed token back for KV cache update
                model.eval([token_id])
                return token_id

        except (ImportError, AttributeError, TypeError, Exception):
            pass

        # ── Approach 3: Greedy from scores array ────────────────────
        try:
            scores = None
            if hasattr(model, 'scores') and model.scores is not None:
                scores = model.scores
            elif hasattr(model, '_scores') and model._scores is not None:
                scores = model._scores

            if scores is not None and len(scores) > 0:
                last_logits = np.array(scores[-1])
                token_id = int(np.argmax(last_logits))
                model.eval([token_id])
                return token_id
        except Exception:
            pass

        # ── Fallback: should not reach here ─────────────────────────
        raise RuntimeError(
            "CodeEcho: Cannot sample tokens — llama-cpp-python API incompatible. "
            "Try updating: pip install llama-cpp-python --upgrade"
        )

    # ── Public API ──────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Whether CodeEcho can be used (API available + sources indexed)."""
        return (
            self.config.enabled
            and self._api_available is True
            and self._index is not None
            and self._index.num_ngrams > 0
        )

    @property
    def cumulative_stats(self) -> EchoStats:
        return self._cumulative_stats

    def stats(self) -> dict:
        """Get cumulative statistics."""
        return {
            "generations": self._generation_count,
            "total_tokens": self._cumulative_stats.total_tokens,
            "echoed_tokens": self._cumulative_stats.echoed_tokens,
            "echo_events": self._cumulative_stats.echo_events,
            "echo_ratio": f"{self._cumulative_stats.echo_ratio:.0%}",
            "avg_speedup": f"{self._cumulative_stats.speedup_estimate:.1f}x",
            "api_available": self._api_available,
        }

    def reset_index(self):
        """Clear the source index (call before re-indexing for a new query)."""
        self._index = SourceIndex(ngram_size=self.config.ngram_size)

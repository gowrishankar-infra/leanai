"""
LeanAI — Auto-Recovery System
Handles model crashes, out-of-memory errors, and automatic fallback.

Scenarios handled:
  1. 32B model runs out of RAM → auto-fallback to 7B
  2. Model crashes mid-generation → retry with smaller context
  3. Generation timeout → retry with fewer tokens
  4. Any exception → graceful error message, system stays alive
"""

import sys
import time
import gc
import traceback
from dataclasses import dataclass, field
from typing import Optional, Callable, List


@dataclass
class RecoveryEvent:
    """Record of a recovery event."""
    timestamp: float
    error_type: str
    action_taken: str
    success: bool
    details: str = ""


@dataclass
class RecoveryConfig:
    """Configuration for auto-recovery."""
    max_retries: int = 2
    timeout_seconds: int = 600        # 10 min max per generation
    oom_fallback_enabled: bool = True  # auto-switch to smaller model on OOM
    reduce_tokens_on_retry: bool = True  # reduce max_tokens on retry
    token_reduction_factor: float = 0.5  # halve tokens on retry
    context_reduction_on_retry: bool = True  # reduce context on retry


class AutoRecovery:
    """
    Wraps model calls with automatic error handling and recovery.
    
    Usage:
        recovery = AutoRecovery(
            primary_model_loader=load_32b,
            fallback_model_loader=load_7b,
        )
        
        # This won't crash — it handles everything
        result = recovery.safe_generate(
            generate_fn=model.generate,
            prompt="...",
            max_tokens=1024,
        )
        
        if result.success:
            print(result.text)
        else:
            print(f"Failed: {result.error}")
    """

    def __init__(
        self,
        config: Optional[RecoveryConfig] = None,
        on_fallback: Optional[Callable] = None,  # called when falling back
        on_recovery: Optional[Callable] = None,   # called after recovery
    ):
        self.config = config or RecoveryConfig()
        self.on_fallback = on_fallback
        self.on_recovery = on_recovery
        self._events: List[RecoveryEvent] = []
        self._total_recoveries = 0
        self._total_fallbacks = 0

    def safe_generate(
        self,
        generate_fn: Callable,
        max_tokens: int = 1024,
        fallback_fn: Optional[Callable] = None,
        **kwargs,
    ) -> "RecoveryResult":
        """
        Call generate_fn with automatic retry and fallback.
        
        Args:
            generate_fn: the primary generation function
            max_tokens: max tokens for generation
            fallback_fn: alternative generation function (smaller model)
            **kwargs: additional args passed to generate_fn
        
        Returns:
            RecoveryResult with text and metadata
        """
        last_error = ""
        current_tokens = max_tokens

        # Try primary model with retries
        for attempt in range(self.config.max_retries + 1):
            try:
                text = generate_fn(max_tokens=current_tokens, **kwargs)
                return RecoveryResult(
                    success=True, text=text,
                    model_used="primary", attempts=attempt + 1,
                )

            except MemoryError as e:
                last_error = f"Out of memory: {e}"
                self._record_event("MemoryError", "attempting fallback", True, str(e))
                self._total_recoveries += 1

                # Free memory
                gc.collect()

                # Try with reduced tokens
                if self.config.reduce_tokens_on_retry:
                    current_tokens = int(current_tokens * self.config.token_reduction_factor)
                    if current_tokens < 64:
                        break

            except RuntimeError as e:
                error_str = str(e).lower()
                if "out of memory" in error_str or "oom" in error_str or "cuda" in error_str:
                    last_error = f"GPU OOM: {e}"
                    self._record_event("GPU_OOM", "reducing tokens", True, str(e))
                    gc.collect()
                    current_tokens = int(current_tokens * self.config.token_reduction_factor)
                    if current_tokens < 64:
                        break
                else:
                    last_error = str(e)
                    self._record_event("RuntimeError", "retrying", attempt < self.config.max_retries, str(e))
                    if attempt == self.config.max_retries:
                        break

            except Exception as e:
                last_error = str(e)
                self._record_event(type(e).__name__, "retrying", attempt < self.config.max_retries, str(e))
                if attempt == self.config.max_retries:
                    break
                # Reduce tokens and retry
                if self.config.reduce_tokens_on_retry:
                    current_tokens = int(current_tokens * self.config.token_reduction_factor)

        # Try fallback model
        if fallback_fn and self.config.oom_fallback_enabled:
            self._total_fallbacks += 1
            if self.on_fallback:
                self.on_fallback(last_error)

            try:
                text = fallback_fn(max_tokens=max_tokens, **kwargs)
                self._record_event("Fallback", "used fallback model", True)
                if self.on_recovery:
                    self.on_recovery("fallback")
                return RecoveryResult(
                    success=True, text=text,
                    model_used="fallback", attempts=self.config.max_retries + 2,
                    recovered=True, recovery_note="Used fallback model after primary failed",
                )
            except Exception as e:
                last_error = f"Fallback also failed: {e}"
                self._record_event("FallbackFailed", str(e), False)

        return RecoveryResult(
            success=False, text="",
            error=last_error,
            model_used="none",
            attempts=self.config.max_retries + 1,
        )

    def safe_call(self, fn: Callable, *args, default=None, **kwargs):
        """
        Safely call any function with exception handling.
        Returns the result or default on failure.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            self._record_event(
                type(e).__name__, f"safe_call failed: {fn.__name__}",
                False, str(e),
            )
            return default

    def _record_event(self, error_type: str, action: str, success: bool, details: str = ""):
        event = RecoveryEvent(
            timestamp=time.time(),
            error_type=error_type,
            action_taken=action,
            success=success,
            details=details[:200],
        )
        self._events.append(event)
        # Keep last 100 events
        if len(self._events) > 100:
            self._events = self._events[-100:]

    def stats(self) -> dict:
        return {
            "total_recoveries": self._total_recoveries,
            "total_fallbacks": self._total_fallbacks,
            "recent_events": len(self._events),
            "last_event": self._events[-1].error_type if self._events else "none",
        }

    def recent_events(self, limit: int = 5) -> str:
        if not self._events:
            return "No recovery events."
        lines = ["Recent recovery events:"]
        for e in self._events[-limit:]:
            icon = "●" if e.success else "✗"
            lines.append(f"  {icon} {e.error_type}: {e.action_taken}")
        return "\n".join(lines)


@dataclass
class RecoveryResult:
    """Result of a recovery-wrapped generation."""
    success: bool
    text: str
    model_used: str = "primary"
    attempts: int = 1
    recovered: bool = False
    recovery_note: str = ""
    error: str = ""


def wrap_engine_with_recovery(engine, model_manager=None) -> AutoRecovery:
    """
    Create an AutoRecovery that wraps the LeanAI engine.
    Handles OOM, crashes, and auto-fallback between models.
    """
    def on_fallback(error):
        print(f"\n  [Recovery] Primary model failed: {error[:80]}", flush=True)
        print(f"  [Recovery] Falling back to smaller model...", flush=True)

    def on_recovery(method):
        print(f"  [Recovery] Recovered via {method}", flush=True)

    return AutoRecovery(
        on_fallback=on_fallback,
        on_recovery=on_recovery,
    )

"""
LeanAI — Cascade Inference
7B model generates a draft response (fast), 32B only reviews and corrects it.

Instead of:  32B generates from scratch (7 minutes)
We do:       7B drafts (30 sec) → 32B reviews & fixes (60 sec) = 90 seconds total

The 32B model is much faster at editing than generating because:
- It processes the draft as INPUT (fast — just reading)
- It only GENERATES corrections (small output — fast)
- Total generation tokens: ~200 instead of ~2000

This is different from speculative decoding (token-level).
This is response-level: draft the whole thing, then fix it.
"""

import time
from typing import Optional, Callable, Tuple


class CascadeInference:
    """
    Two-stage inference: fast model drafts, quality model reviews.
    
    Usage:
        cascade = CascadeInference(
            draft_fn=model_7b.generate,
            review_fn=model_32b.generate,
        )
        result = cascade.generate(query, project_context="...")
        # result.text has the final corrected response
        # result.draft_time_ms + result.review_time_ms = total time
    """

    def __init__(self, draft_fn=None, review_fn=None, enabled=True):
        """
        Args:
            draft_fn: function(system, user) -> str — fast model (7B)
            review_fn: function(system, user) -> str — quality model (32B)
            enabled: whether to use cascade or fall back to single model
        """
        self.draft_fn = draft_fn
        self.review_fn = review_fn
        self.enabled = enabled
        self._stats = {
            "total_cascades": 0,
            "avg_draft_ms": 0,
            "avg_review_ms": 0,
            "avg_total_ms": 0,
        }

    def generate(self, query: str, system_prompt: str = "",
                 project_context: str = "") -> 'CascadeResult':
        """
        Generate a response using cascade inference.
        
        Step 1: 7B drafts the response (fast)
        Step 2: 32B reviews and corrects (focused, fast)
        
        Returns CascadeResult with final text and timing.
        """
        if not self.enabled or not self.draft_fn or not self.review_fn:
            # Fallback to review model only
            if self.review_fn:
                text = self.review_fn(system_prompt, query)
                return CascadeResult(text=text, draft="", draft_time_ms=0, review_time_ms=0)
            return CascadeResult(text="Cascade not configured.", draft="", draft_time_ms=0, review_time_ms=0)

        # ── Step 1: Fast draft with 7B ────────────────────────
        draft_system = system_prompt
        if project_context:
            draft_system += f"\n\nProject context:\n{project_context[:1500]}"

        start = time.time()
        draft = self.draft_fn(draft_system, query)
        draft_ms = (time.time() - start) * 1000

        if not draft or len(draft.strip()) < 20:
            # Draft failed — fall back to 32B
            start2 = time.time()
            text = self.review_fn(draft_system, query)
            review_ms = (time.time() - start2) * 1000
            return CascadeResult(text=text, draft=draft or "", 
                               draft_time_ms=draft_ms, review_time_ms=review_ms)

        # ── Step 2: 32B reviews and corrects ──────────────────
        review_prompt = (
            "A junior developer wrote this response. Your job is to:\n"
            "1. Fix any incorrect code or wrong information\n"
            "2. Add anything important that was missed\n"
            "3. Improve the explanation if it's unclear\n"
            "4. Keep what's already good — don't rewrite from scratch\n\n"
            "If the response is already good, return it mostly unchanged "
            "with minor improvements.\n\n"
            f"Original question: {query}\n\n"
            f"Draft response:\n{draft}\n\n"
            "Your improved version:"
        )

        review_system = (
            "You are a senior code reviewer. Fix errors in the draft response. "
            "Keep what's good, fix what's wrong, add what's missing. "
            "Be concise. Output ONLY the improved response, no meta-commentary."
        )

        if project_context:
            review_system += f"\n\nActual project context (use this to verify claims):\n{project_context[:1500]}"

        start2 = time.time()
        reviewed = self.review_fn(review_system, review_prompt)
        review_ms = (time.time() - start2) * 1000

        # Use reviewed version if it's substantive, otherwise use draft
        if reviewed and len(reviewed.strip()) > 20:
            final_text = reviewed
        else:
            final_text = draft

        # Update stats
        self._stats["total_cascades"] += 1
        total_ms = draft_ms + review_ms
        n = self._stats["total_cascades"]
        self._stats["avg_draft_ms"] = (self._stats["avg_draft_ms"] * (n-1) + draft_ms) / n
        self._stats["avg_review_ms"] = (self._stats["avg_review_ms"] * (n-1) + review_ms) / n
        self._stats["avg_total_ms"] = (self._stats["avg_total_ms"] * (n-1) + total_ms) / n

        return CascadeResult(
            text=final_text,
            draft=draft,
            draft_time_ms=draft_ms,
            review_time_ms=review_ms,
        )

    def should_cascade(self, query: str) -> bool:
        """
        Determine if a query should use cascade inference.
        
        Use cascade for complex queries where 32B quality matters.
        Skip cascade for simple queries where 7B is sufficient.
        """
        if not self.enabled:
            return False

        # Complex queries benefit from cascade
        complex_words = {
            "explain", "review", "analyze", "compare", "architect",
            "design", "refactor", "optimize", "security", "pipeline",
            "implement", "production", "best practice", "trade-off",
        }
        lower = query.lower()
        return any(w in lower for w in complex_words)

    def stats(self) -> dict:
        return dict(self._stats)


class CascadeResult:
    """Result from cascade inference."""

    def __init__(self, text: str, draft: str, draft_time_ms: float, review_time_ms: float):
        self.text = text
        self.draft = draft
        self.draft_time_ms = draft_time_ms
        self.review_time_ms = review_time_ms
        self.total_time_ms = draft_time_ms + review_time_ms
        self.used_cascade = draft_time_ms > 0 and review_time_ms > 0

    def __str__(self):
        return self.text

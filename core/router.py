"""
LeanAI · Smart Task Router
Routes each query to the cheapest model tier that can handle it.
This is the key to being fast AND accurate — never use a big model for a small task.

Tiers:
  TINY   — rule-based / regex / lookup  (0ms, 0 RAM)
  SMALL  — 50M param model              (~10ms, ~100MB)
  MEDIUM — 300M param model             (~50ms, ~300MB)
  FULL   — 1B param model               (~200ms, ~512MB)
  HEAVY  — future: larger model or swarm (Phase 5)
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Tier(Enum):
    TINY   = "tiny"
    SMALL  = "small"
    MEDIUM = "medium"
    FULL   = "full"
    HEAVY  = "heavy"


@dataclass
class RouteDecision:
    tier: Tier
    reason: str
    confidence: float          # 0.0–1.0
    requires_verifier: bool    # should neurosymbolic verifier run?
    requires_tools: bool       # should code executor / calculator run?
    estimated_tokens: int      # rough output length estimate


# Patterns that never need a neural model
TINY_PATTERNS = [
    (r"^\s*\d[\d\s\+\-\*\/\^\(\)\.]+\=?\s*$", "arithmetic expression"),
    (r"^(hi|hello|hey|yo|sup|howdy)\b", "simple greeting"),
    (r"^(thanks|thank you|thx|ty)\b", "thanks"),
    (r"^(bye|goodbye|see you|cya)\b", "farewell"),
    (r"^what(?:'s| is) the time\??$", "time query"),
    (r"^what(?:'s| is) (\d+)\s*[\+\-\*\/]\s*(\d+)\??$", "simple math"),
]

# Keywords indicating hard reasoning needed
HEAVY_KEYWORDS = [
    "prove", "formal proof", "theorem", "hypothesis",
    "simulate", "generate dataset", "train a model",
    "write a full", "entire codebase", "complete system",
    "research paper", "literature review",
]

# Keywords indicating math/logic verification needed
VERIFY_KEYWORDS = [
    "calculate", "compute", "solve", "equation", "solution", "formula",
    "math", "algebra", "calculus", "statistics", "probability",
    "proof", "theorem", "logical", "verify", "is it true that",
    "does", "will", "if then", "implies",
]

# Keywords indicating code execution would help
TOOL_KEYWORDS = [
    "code", "script", "function", "class", "implement",
    "algorithm", "sort", "search", "parse", "regex",
    "api", "request", "fetch", "file", "read", "write",
    "execute", "run", "output", "result of",
]

# Complexity signals
COMPLEXITY_SIGNALS = [
    "explain", "describe", "what is", "how does", "why does",
    "compare", "difference between", "pros and cons",
    "step by step", "walk me through",
]


class TaskRouter:
    """
    Classifies incoming queries and routes them to the appropriate model tier.
    Uses heuristics in Phase 0 — will be replaced by a learned classifier in Phase 1.
    """

    def __init__(self):
        self._tiny_patterns = [(re.compile(p, re.IGNORECASE), label)
                                for p, label in TINY_PATTERNS]

    def route(self, query: str, context_length: int = 0) -> RouteDecision:
        """
        Analyze a query and return a routing decision.
        
        Args:
            query: The user's input text
            context_length: Number of tokens in current conversation history
            
        Returns:
            RouteDecision with tier, reason, and flags
        """
        q = query.strip().lower()

        # ── Tiny: pure rule-based, no model needed ──
        for pattern, label in self._tiny_patterns:
            if pattern.match(q):
                return RouteDecision(
                    tier=Tier.TINY,
                    reason=f"matched rule: {label}",
                    confidence=0.99,
                    requires_verifier=False,
                    requires_tools=False,
                    estimated_tokens=20,
                )

        # ── Heavy: complex multi-step tasks ──
        for kw in HEAVY_KEYWORDS:
            if kw in q:
                return RouteDecision(
                    tier=Tier.HEAVY,
                    reason=f"heavy keyword: '{kw}'",
                    confidence=0.85,
                    requires_verifier=self._needs_verifier(q),
                    requires_tools=True,
                    estimated_tokens=800,
                )

        # ── Score complexity ──
        complexity = self._complexity_score(q, context_length)

        needs_verify = self._needs_verifier(q)
        needs_tools  = self._needs_tools(q)

        # ── Route by complexity score ──
        if complexity < 0.2:
            return RouteDecision(
                tier=Tier.SMALL,
                reason=f"low complexity ({complexity:.2f})",
                confidence=0.8,
                requires_verifier=needs_verify,
                requires_tools=needs_tools,
                estimated_tokens=60,
            )
        elif complexity < 0.5:
            return RouteDecision(
                tier=Tier.MEDIUM,
                reason=f"medium complexity ({complexity:.2f})",
                confidence=0.8,
                requires_verifier=needs_verify,
                requires_tools=needs_tools,
                estimated_tokens=200,
            )
        else:
            return RouteDecision(
                tier=Tier.FULL,
                reason=f"high complexity ({complexity:.2f})",
                confidence=0.75,
                requires_verifier=needs_verify,
                requires_tools=needs_tools,
                estimated_tokens=400,
            )

    def _complexity_score(self, q: str, context_length: int) -> float:
        """Score 0.0 (trivial) to 1.0 (very complex)."""
        score = 0.0

        # Length signal
        words = len(q.split())
        score += min(words / 60.0, 0.3)

        # Complexity signal words
        for kw in COMPLEXITY_SIGNALS:
            if kw in q:
                score += 0.1
                break

        # Multi-sentence
        sentences = q.count(".") + q.count("?") + q.count("!")
        score += min(sentences * 0.05, 0.15)

        # Long conversation context
        score += min(context_length / 4000.0, 0.2)

        # Code/technical content
        if any(c in q for c in ["`", "def ", "class ", "import ", "```"]):
            score += 0.15

        return min(score, 1.0)

    def _needs_verifier(self, q: str) -> bool:
        return any(kw in q for kw in VERIFY_KEYWORDS)

    def _needs_tools(self, q: str) -> bool:
        return any(kw in q for kw in TOOL_KEYWORDS)

    def explain(self, decision: RouteDecision) -> str:
        """Human-readable explanation of routing decision."""
        tier_descriptions = {
            Tier.TINY:   "rule-based (instant, no model)",
            Tier.SMALL:  "small model (~50M params, ~10ms)",
            Tier.MEDIUM: "medium model (~300M params, ~50ms)",
            Tier.FULL:   "full model (~1B params, ~200ms)",
            Tier.HEAVY:  "full model + tools (complex task)",
        }
        extras = []
        if decision.requires_verifier:
            extras.append("+ neurosymbolic verifier")
        if decision.requires_tools:
            extras.append("+ code executor")

        return (
            f"[Router] → {tier_descriptions[decision.tier]} "
            f"{' '.join(extras)} | "
            f"reason: {decision.reason} | "
            f"confidence: {decision.confidence:.0%}"
        )

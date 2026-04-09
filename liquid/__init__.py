"""
LeanAI Phase 6b — Liquid Adaptive Router
A continuously adapting routing system inspired by Liquid Neural Networks.

Unlike the static rule-based router, this one LEARNS from every query:
  - Tracks which tier produced good results for which query types
  - Adapts routing weights in real-time using exponential moving averages
  - Predicts optimal tier before routing, improving over time
  
The "liquid" property: weights flow and adapt continuously, never fixed.
"""

import time
import math
import json
import os
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class RoutingFeatures:
    """Features extracted from a query for routing decisions."""
    word_count: int = 0
    has_code_keywords: bool = False
    has_math_keywords: bool = False
    has_question_mark: bool = False
    is_greeting: bool = False
    is_memory_query: bool = False
    avg_word_length: float = 0.0
    complexity_score: float = 0.0  # 0-1, estimated complexity

    def to_vector(self) -> list:
        """Convert to a numeric feature vector."""
        return [
            min(self.word_count / 50.0, 1.0),
            1.0 if self.has_code_keywords else 0.0,
            1.0 if self.has_math_keywords else 0.0,
            1.0 if self.has_question_mark else 0.0,
            1.0 if self.is_greeting else 0.0,
            1.0 if self.is_memory_query else 0.0,
            min(self.avg_word_length / 10.0, 1.0),
            self.complexity_score,
        ]


@dataclass
class TierPerformance:
    """Tracks how well a tier performs over time."""
    tier_name: str
    total_queries: int = 0
    total_good: int = 0        # queries where confidence > threshold
    total_fast: int = 0        # queries under latency threshold
    avg_confidence: float = 0.5
    avg_latency_ms: float = 1000.0
    ema_confidence: float = 0.5  # exponential moving average
    ema_latency: float = 1000.0
    # Per-feature-bin performance (learned routing)
    feature_scores: Dict[str, float] = field(default_factory=dict)


TIERS = ["tiny", "small", "medium", "swarm"]

CODE_KEYWORDS = {
    "def", "class", "import", "function", "code", "program", "script",
    "python", "javascript", "rust", "golang", "api", "bug", "debug",
    "error", "compile", "syntax", "algorithm", "loop", "array", "list",
    "dict", "sql", "database", "html", "css", "react", "docker",
    "kubernetes", "git", "build", "deploy", "test", "pytest",
}

MATH_KEYWORDS = {
    "calculate", "compute", "sum", "multiply", "divide", "add", "subtract",
    "integral", "derivative", "equation", "solve", "math", "algebra",
    "factorial", "prime", "sqrt", "root", "percentage", "average",
}

GREETING_PATTERNS = {
    "hello", "hi", "hey", "good morning", "good evening", "good afternoon",
    "howdy", "greetings", "what's up", "sup",
}

MEMORY_KEYWORDS = {
    "my name", "my job", "my work", "where do i", "where am i",
    "what do i", "who am i", "remember", "you know about me",
}


def extract_features(query: str) -> RoutingFeatures:
    """Extract routing features from a query."""
    words = query.lower().split()
    word_count = len(words)
    word_set = set(words)
    lower = query.lower().strip()

    has_code = bool(word_set & CODE_KEYWORDS)
    has_math = bool(word_set & MATH_KEYWORDS)
    has_question = "?" in query
    is_greeting = lower in GREETING_PATTERNS or (word_count <= 3 and word_set & {"hello", "hi", "hey"})
    is_memory = any(kw in lower for kw in MEMORY_KEYWORDS)

    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)

    # Complexity heuristic
    complexity = min(1.0, (
        word_count / 30.0 * 0.3 +
        (1.0 if has_code else 0.0) * 0.3 +
        avg_word_len / 8.0 * 0.2 +
        (0.0 if is_greeting else 0.2)
    ))

    return RoutingFeatures(
        word_count=word_count,
        has_code_keywords=has_code,
        has_math_keywords=has_math,
        has_question_mark=has_question,
        is_greeting=is_greeting,
        is_memory_query=is_memory,
        avg_word_length=avg_word_len,
        complexity_score=complexity,
    )


def feature_bin(features: RoutingFeatures) -> str:
    """Create a hashable bin key from features for the lookup table."""
    parts = [
        "code" if features.has_code_keywords else "nocode",
        "math" if features.has_math_keywords else "nomath",
        "greet" if features.is_greeting else "nogreet",
        "mem" if features.is_memory_query else "nomem",
        "short" if features.word_count < 5 else "med" if features.word_count < 20 else "long",
    ]
    return "|".join(parts)


class LiquidRouter:
    """
    Adaptive router that learns optimal tier selection in real-time.
    
    Usage:
        router = LiquidRouter()
        tier = router.route("explain quicksort")  # -> "medium"
        
        # After getting response, feed back the quality signal
        router.feedback("medium", confidence=0.85, latency_ms=5000)
        
        # Next time a similar query comes, routing is better informed
    """

    EMA_ALPHA = 0.1  # learning rate for exponential moving average

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        latency_threshold_ms: float = 5000,
        data_dir: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.data_dir = data_dir or str(Path.home() / ".leanai" / "liquid_router")
        os.makedirs(self.data_dir, exist_ok=True)

        # Performance tracking per tier
        self.tiers: Dict[str, TierPerformance] = {
            name: TierPerformance(tier_name=name) for name in TIERS
        }

        # Current query context (for feedback)
        self._current_features: Optional[RoutingFeatures] = None
        self._current_bin: str = ""
        self._queries_routed: int = 0

        self._load_state()

    def route(self, query: str) -> str:
        """
        Route a query to the optimal tier.
        Uses learned performance data + feature matching.
        """
        features = extract_features(query)
        self._current_features = features
        self._current_bin = feature_bin(features)
        self._queries_routed += 1

        # Rule-based fast paths (never change)
        if features.is_greeting:
            return "tiny"
        if features.is_memory_query:
            return "tiny"  # memory lookups are instant
        if features.has_math_keywords and features.word_count < 10:
            return "tiny"  # math verifier handles these

        # Learned routing: score each tier for this feature bin
        scores = {}
        for tier_name, perf in self.tiers.items():
            bin_key = self._current_bin
            base_score = perf.feature_scores.get(bin_key, 0.5)

            # Adjust by EMA confidence and latency
            conf_bonus = (perf.ema_confidence - 0.5) * 0.5  # [-0.25, +0.25]
            latency_penalty = max(0, (perf.ema_latency - self.latency_threshold_ms) / 10000) * 0.2

            scores[tier_name] = base_score + conf_bonus - latency_penalty

        # Complexity-based fallback
        if features.complexity_score > 0.7:
            scores["medium"] += 0.3
        elif features.complexity_score > 0.4:
            scores["small"] += 0.2
        else:
            scores["tiny"] += 0.2

        if features.has_code_keywords:
            scores["medium"] += 0.2

        # Pick best tier
        best = max(scores, key=scores.get)
        return best

    def feedback(self, tier_used: str, confidence: float, latency_ms: float):
        """
        Feed back the quality of a response to update routing weights.
        Call this after every query with the actual results.
        """
        if tier_used not in self.tiers:
            return

        perf = self.tiers[tier_used]
        perf.total_queries += 1

        # Normalize confidence to 0-1 if it's 0-100
        if confidence > 1.0:
            confidence = confidence / 100.0

        good = confidence >= self.confidence_threshold
        fast = latency_ms <= self.latency_threshold_ms

        if good:
            perf.total_good += 1
        if fast:
            perf.total_fast += 1

        # Update EMAs (liquid adaptation)
        alpha = self.EMA_ALPHA
        perf.ema_confidence = alpha * confidence + (1 - alpha) * perf.ema_confidence
        perf.ema_latency = alpha * latency_ms + (1 - alpha) * perf.ema_latency
        perf.avg_confidence = perf.total_good / max(perf.total_queries, 1)
        perf.avg_latency_ms = perf.ema_latency

        # Update feature-bin score
        bin_key = self._current_bin
        if bin_key:
            old_score = perf.feature_scores.get(bin_key, 0.5)
            reward = 1.0 if good and fast else 0.7 if good else 0.3 if fast else 0.1
            perf.feature_scores[bin_key] = alpha * reward + (1 - alpha) * old_score

        # Periodic save
        if perf.total_queries % 10 == 0:
            self._save_state()

    def _state_path(self) -> str:
        return os.path.join(self.data_dir, "liquid_router_state.json")

    def _save_state(self):
        """Persist learned routing weights."""
        state = {}
        for name, perf in self.tiers.items():
            state[name] = {
                "total_queries": perf.total_queries,
                "total_good": perf.total_good,
                "total_fast": perf.total_fast,
                "ema_confidence": perf.ema_confidence,
                "ema_latency": perf.ema_latency,
                "feature_scores": perf.feature_scores,
            }
        try:
            with open(self._state_path(), "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def _load_state(self):
        """Load learned routing weights from disk."""
        path = self._state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                state = json.load(f)
            for name, data in state.items():
                if name in self.tiers:
                    perf = self.tiers[name]
                    perf.total_queries = data.get("total_queries", 0)
                    perf.total_good = data.get("total_good", 0)
                    perf.total_fast = data.get("total_fast", 0)
                    perf.ema_confidence = data.get("ema_confidence", 0.5)
                    perf.ema_latency = data.get("ema_latency", 1000.0)
                    perf.feature_scores = data.get("feature_scores", {})
        except (json.JSONDecodeError, Exception):
            pass

    def stats(self) -> dict:
        return {
            "queries_routed": self._queries_routed,
            "tiers": {
                name: {
                    "queries": perf.total_queries,
                    "good_rate": perf.total_good / max(perf.total_queries, 1),
                    "ema_confidence": round(perf.ema_confidence, 3),
                    "ema_latency": round(perf.ema_latency, 1),
                    "learned_bins": len(perf.feature_scores),
                }
                for name, perf in self.tiers.items()
            },
        }

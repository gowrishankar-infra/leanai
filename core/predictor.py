"""
LeanAI — Predictive Pre-Generation
Predicts what you'll ask next and pre-generates the answer before you type.

How it works:
  1. After every query, analyze what you just asked
  2. Predict 2-3 likely follow-up questions
  3. Generate answers for them in a background thread
  4. When you ask, check if it matches a prediction → instant response

Example:
  You ask: "what does engine_v3.py do?"
  LeanAI predicts you might next ask:
    - "show me the generate method"
    - "what imports does it use"
    - "how does it connect to the router"
  Pre-generates answers for all three in the background.
  When you type "show me generate" → instant, already cached.

This is the only way to solve the speed problem without GPU hardware.
"""

import time
import threading
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple


@dataclass
class Prediction:
    """A predicted follow-up question with its pre-generated answer."""
    question: str
    answer: str = ""
    confidence: float = 0.0
    generated: bool = False
    generation_time_ms: float = 0.0
    timestamp: float = 0.0


# ── Follow-up prediction patterns ────────────────────────────

FOLLOW_UP_PATTERNS = {
    # If they asked about a file → likely ask about its functions/imports/deps
    "file_inquiry": {
        "triggers": ["what does", "describe", "explain", "about"],
        "follow_ups": [
            "show me the main functions in {topic}",
            "what imports does {topic} use",
            "what depends on {topic}",
        ],
    },
    # If they asked about a function → likely ask how to use it or fix it
    "function_inquiry": {
        "triggers": ["how does", "what does", "explain", "function"],
        "follow_ups": [
            "show me an example of using {topic}",
            "what are the parameters for {topic}",
            "how can I improve {topic}",
        ],
    },
    # If they asked about an error → likely ask how to fix it
    "error_inquiry": {
        "triggers": ["error", "bug", "fix", "broken", "crash", "fail"],
        "follow_ups": [
            "how do I fix {topic}",
            "what causes {topic}",
            "show me the correct way to do {topic}",
        ],
    },
    # If they asked about architecture → likely ask about specifics
    "architecture_inquiry": {
        "triggers": ["design", "architect", "structure", "how does the system"],
        "follow_ups": [
            "what are the trade-offs of {topic}",
            "how would you improve {topic}",
            "what alternatives exist for {topic}",
        ],
    },
    # If they wrote code → likely ask to test or improve it
    "code_written": {
        "triggers": ["implement", "write", "create", "build", "code"],
        "follow_ups": [
            "write tests for {topic}",
            "how can I optimize {topic}",
            "add error handling to {topic}",
        ],
    },
    # If they asked about git → likely want more git info
    "git_inquiry": {
        "triggers": ["/git", "commit", "change", "history", "branch"],
        "follow_ups": [
            "show me the most changed files",
            "what changed in the last week",
            "who made the most recent changes",
        ],
    },
}


def predict_follow_ups(query: str, response: str = "", max_predictions: int = 3) -> List[str]:
    """
    Predict likely follow-up questions based on the current query.
    Returns a list of predicted questions.
    """
    query_lower = query.lower()
    predictions = []

    # Extract the topic from the query
    topic = _extract_topic(query)

    # Match against patterns
    for pattern_name, pattern in FOLLOW_UP_PATTERNS.items():
        if any(trigger in query_lower for trigger in pattern["triggers"]):
            for follow_up in pattern["follow_ups"]:
                pred = follow_up.format(topic=topic)
                if pred not in predictions:
                    predictions.append(pred)

    # Generic follow-ups if no patterns matched
    if not predictions:
        predictions = [
            f"tell me more about {topic}",
            f"give me an example of {topic}",
        ]

    return predictions[:max_predictions]


def _extract_topic(query: str) -> str:
    """Extract the main topic from a query."""
    # Remove common question prefixes
    lower = query.lower().strip()
    prefixes = [
        "what does ", "how does ", "explain ", "describe ",
        "show me ", "tell me about ", "what is ", "how to ",
        "implement ", "write ", "create ", "fix ", "why does ",
    ]
    topic = lower
    for prefix in prefixes:
        if topic.startswith(prefix):
            topic = topic[len(prefix):]
            break

    # Remove trailing question marks and common suffixes
    topic = topic.rstrip("?").strip()
    suffixes = [" do", " work", " mean", " look like", " file"]
    for suffix in suffixes:
        if topic.endswith(suffix):
            topic = topic[:-len(suffix)].strip()

    return topic if topic else query[:50]


def _similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class PredictivePreGenerator:
    """
    Background pre-generation of predicted follow-up responses.
    
    Usage:
        predictor = PredictivePreGenerator(generate_fn=model.generate)
        
        # After each query, predict and pre-generate
        predictor.on_query_complete("what does engine.py do?", "It handles...")
        
        # Before generating, check if we have a pre-generated answer
        cached = predictor.check_prediction("show me the generate method")
        if cached:
            print(cached)  # instant!
    """

    def __init__(self, generate_fn: Optional[Callable] = None,
                 similarity_threshold: float = 0.4,
                 max_predictions: int = 3):
        self.generate_fn = generate_fn
        self.similarity_threshold = similarity_threshold
        self.max_predictions = max_predictions
        self._predictions: Dict[str, Prediction] = {}
        self._bg_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stats = {
            "predictions_made": 0,
            "predictions_hit": 0,
            "predictions_missed": 0,
            "time_saved_ms": 0,
        }

    def on_query_complete(self, query: str, response: str):
        """
        Called after each query completes.
        Predicts follow-ups and starts background generation.
        """
        follow_ups = predict_follow_ups(query, response, self.max_predictions)

        if not follow_ups or not self.generate_fn:
            return

        # Store predictions (without answers yet)
        with self._lock:
            self._predictions.clear()  # clear old predictions
            for q in follow_ups:
                key = hashlib.md5(q.lower().encode()).hexdigest()
                self._predictions[key] = Prediction(
                    question=q,
                    timestamp=time.time(),
                )
                self._stats["predictions_made"] += 1

        # Start background generation thread
        self._bg_thread = threading.Thread(
            target=self._generate_predictions,
            args=(follow_ups,),
            daemon=True,
        )
        self._bg_thread.start()

    def _generate_predictions(self, questions: List[str]):
        """Background thread that generates answers for predictions."""
        for question in questions:
            key = hashlib.md5(question.lower().encode()).hexdigest()

            try:
                start = time.time()
                answer = self.generate_fn(question)
                elapsed = (time.time() - start) * 1000

                with self._lock:
                    if key in self._predictions:
                        self._predictions[key].answer = answer
                        self._predictions[key].generated = True
                        self._predictions[key].generation_time_ms = elapsed
                        self._predictions[key].confidence = 0.7
            except Exception:
                pass  # silently skip failed predictions

    def check_prediction(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Check if we have a pre-generated answer for this query.
        Returns (answer, confidence) or None.
        """
        with self._lock:
            # Exact match first
            key = hashlib.md5(query.lower().encode()).hexdigest()
            if key in self._predictions:
                pred = self._predictions[key]
                if pred.generated and pred.answer:
                    self._stats["predictions_hit"] += 1
                    self._stats["time_saved_ms"] += pred.generation_time_ms
                    return pred.answer, pred.confidence

            # Fuzzy match — check similarity
            for pred_key, pred in self._predictions.items():
                if pred.generated and pred.answer:
                    sim = _similarity(query, pred.question)
                    if sim >= self.similarity_threshold:
                        self._stats["predictions_hit"] += 1
                        self._stats["time_saved_ms"] += pred.generation_time_ms
                        return pred.answer, pred.confidence * sim

            self._stats["predictions_missed"] += 1
            return None

    def get_pending_predictions(self) -> List[str]:
        """Get list of predicted questions (for display)."""
        with self._lock:
            return [p.question for p in self._predictions.values()]

    def stats(self) -> dict:
        return {
            **self._stats,
            "hit_rate": (
                f"{self._stats['predictions_hit'] / max(self._stats['predictions_hit'] + self._stats['predictions_missed'], 1):.0%}"
            ),
            "pending": len([p for p in self._predictions.values() if not p.generated]),
            "ready": len([p for p in self._predictions.values() if p.generated]),
        }

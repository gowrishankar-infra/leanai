"""
LeanAI · Phase 3 — Quality Filter
Decides which exchanges are worth training on.

The golden rule: only train on what you're sure is correct.
Bad training data is worse than no training data.

Scoring criteria:
  - Verified by Z3/SymPy         → +0.40 (strongest signal)
  - High confidence score        → +0.25
  - User gave positive feedback  → +0.20
  - Response was fast            → +0.05
  - Response was concise         → +0.05
  - From self-play (verified)    → +0.35

Threshold: 0.65 minimum quality to enter training pool
"""

from dataclasses import dataclass
from typing import Optional
from training.self_improve import TrainingPair, FeedbackSignal


@dataclass
class QualityAssessment:
    score: float                  # 0.0 – 1.0
    passes: bool                  # True if above threshold
    reasons: list                 # what contributed to score
    disqualifiers: list           # what lowered the score


THRESHOLD = 0.65


class QualityFilter:
    """
    Evaluates every training pair and decides if it's good enough to train on.
    This is the gatekeeper — nothing below threshold reaches the model.
    """

    def assess(self, pair: TrainingPair) -> QualityAssessment:
        score = 0.0
        reasons = []
        disqualifiers = []

        # ── Positive signals ──────────────────────────────────────────

        # Formal verification is the strongest signal
        if pair.verified:
            score += 0.40
            reasons.append("formally verified correct (+0.40)")

        # High confidence from scoring engine
        if pair.confidence >= 0.85:
            score += 0.25
            reasons.append(f"high confidence {pair.confidence:.0%} (+0.25)")
        elif pair.confidence >= 0.70:
            score += 0.15
            reasons.append(f"good confidence {pair.confidence:.0%} (+0.15)")

        # User explicitly said it was good
        if pair.feedback == FeedbackSignal.EXCELLENT:
            score += 0.20
            reasons.append("user marked excellent (+0.20)")
        elif pair.feedback == FeedbackSignal.GOOD:
            score += 0.10
            reasons.append("user marked good (+0.10)")

        # Self-play pairs are pre-verified
        if pair.tier_used == "self_play":
            score += 0.35
            reasons.append("self-play verified pair (+0.35)")

        # Fast responses tend to be more confident
        if pair.latency_ms < 3000 and pair.latency_ms > 0:
            score += 0.05
            reasons.append("fast response (+0.05)")

        # Concise responses (not too short, not too long)
        resp_len = len(pair.response)
        if 50 < resp_len < 800:
            score += 0.05
            reasons.append("good response length (+0.05)")

        # ── Negative signals ──────────────────────────────────────────

        # Explicit wrong feedback
        if pair.feedback == FeedbackSignal.WRONG:
            score -= 0.50
            disqualifiers.append("user marked wrong (-0.50)")

        # Very low confidence
        if pair.confidence < 0.35:
            score -= 0.20
            disqualifiers.append(f"very low confidence {pair.confidence:.0%} (-0.20)")

        # Response too short (likely unhelpful)
        if resp_len < 20:
            score -= 0.15
            disqualifiers.append("response too short (-0.15)")

        # Response too long (likely rambling)
        if resp_len > 2000:
            score -= 0.10
            disqualifiers.append("response too long (-0.10)")

        # Demo mode responses never train
        if "demo mode" in pair.response.lower():
            score = 0.0
            disqualifiers.append("demo mode response (score zeroed)")

        score = max(0.0, min(1.0, score))

        return QualityAssessment(
            score=round(score, 3),
            passes=score >= THRESHOLD,
            reasons=reasons,
            disqualifiers=disqualifiers,
        )

    def filter_batch(self, pairs: list) -> tuple:
        """
        Filter a list of TrainingPairs.
        Returns (accepted, rejected) tuple.
        """
        accepted = []
        rejected = []
        for pair in pairs:
            assessment = self.assess(pair)
            if assessment.passes:
                accepted.append(pair)
            else:
                rejected.append(pair)
        return accepted, rejected

    def stats(self, pairs: list) -> dict:
        """Return quality statistics for a batch of pairs."""
        if not pairs:
            return {"total": 0, "passed": 0, "pass_rate": 0.0, "avg_score": 0.0}

        assessments = [self.assess(p) for p in pairs]
        passed = sum(1 for a in assessments if a.passes)
        avg_score = sum(a.score for a in assessments) / len(assessments)

        return {
            "total": len(pairs),
            "passed": passed,
            "rejected": len(pairs) - passed,
            "pass_rate": round(passed / len(pairs), 3),
            "avg_score": round(avg_score, 3),
            "threshold": THRESHOLD,
        }

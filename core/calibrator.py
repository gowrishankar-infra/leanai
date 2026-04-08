"""
LeanAI · Phase 3 — Confidence Calibrator
Fixes the 50% confidence showing on everything.

The problem: Phi-3 Mini doesn't hedge like GPT-4.
It says "The capital of France is Paris" not "I think Paris is the capital".
So our linguistic scorer sees no hedges and gives 50% base score.

The fix: domain-aware calibration.
  - Factual lookups from memory → 95% (we verified it)
  - Math with verifier → 98%
  - General knowledge → calibrate by response characteristics
  - Uncertain/hedging → lower as before
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class CalibratedScore:
    raw: float           # original score from linguistic analysis
    calibrated: float    # adjusted score
    label: str
    bar: str
    method: str          # what calibration was applied
    explanation: str


class ConfidenceCalibrator:
    """
    Post-processes confidence scores to be better calibrated
    for Phi-3 Mini's output style.
    """

    # Phi-3 doesn't hedge unless it's genuinely uncertain
    PHI3_BASE_CONFIDENCE = 0.72

    # Domain boosts
    DOMAIN_BOOSTS = {
        "tiny":   0.99,   # rule-based, always correct
        "memory": 0.95,   # from world model, verified
        "self_play": 0.93, # self-generated and verified
    }

    # Response patterns that indicate genuine high confidence
    HIGH_CONF_PATTERNS = [
        r"^\w+\s+is\s+\w+\.",           # "X is Y." — direct factual
        r"^the\s+\w+\s+of\s+\w+\s+is",  # "The X of Y is Z"
        r"^yes[,\.]",                    # definitive yes
        r"^no[,\.]",                     # definitive no
        r"^\d+$",                        # pure number answer
        r"^the answer is",               # explicit answer
    ]

    # Response patterns that indicate uncertainty
    LOW_CONF_PATTERNS = [
        r"i('m| am) not (sure|certain)",
        r"i (think|believe|suppose)",
        r"it (might|could|may) be",
        r"possibly|probably|perhaps",
        r"i('m| am) unsure",
        r"you (should|may want to) verify",
    ]

    def calibrate(
        self,
        raw_score: float,
        response: str,
        tier: str,
        verified: bool,
        query: str = "",
    ) -> CalibratedScore:
        """
        Apply calibration to get a better confidence score.
        """
        # Tier-based overrides (most reliable)
        if tier in self.DOMAIN_BOOSTS:
            score = self.DOMAIN_BOOSTS[tier]
            return self._make_score(
                raw_score, score,
                f"tier override for '{tier}'",
            )

        # Verified by formal verifier
        if verified:
            score = 0.96
            return self._make_score(
                raw_score, score,
                "formally verified correct",
            )

        # Check response patterns
        resp_lower = response.lower().strip()

        # Explicit uncertainty signals
        uncertainty = sum(
            1 for p in self.LOW_CONF_PATTERNS
            if re.search(p, resp_lower)
        )
        if uncertainty > 0:
            score = max(0.25, raw_score - (uncertainty * 0.10))
            return self._make_score(
                raw_score, score,
                f"{uncertainty} uncertainty signal(s) detected",
            )

        # High confidence patterns
        high_conf = sum(
            1 for p in self.HIGH_CONF_PATTERNS
            if re.search(p, resp_lower)
        )
        if high_conf > 0:
            score = min(0.90, self.PHI3_BASE_CONFIDENCE + (high_conf * 0.06))
            return self._make_score(
                raw_score, score,
                f"{high_conf} high-confidence pattern(s) matched",
            )

        # Short definitive responses tend to be more confident
        words = len(response.split())
        if words < 15 and not any(c in resp_lower for c in ["think", "believe", "maybe"]):
            score = 0.78
            return self._make_score(raw_score, score, "short definitive response")

        # Default: apply Phi-3 base calibration
        # Phi-3's base is higher than our linguistic scorer assumes
        score = max(raw_score, self.PHI3_BASE_CONFIDENCE)
        return self._make_score(raw_score, score, "phi3 base calibration applied")

    def _make_score(self, raw: float, calibrated: float, method: str) -> CalibratedScore:
        calibrated = round(max(0.05, min(0.99, calibrated)), 3)
        pct = int(calibrated * 100)
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        label = self._label(calibrated)
        return CalibratedScore(
            raw=raw,
            calibrated=calibrated,
            label=label,
            bar=bar,
            method=method,
            explanation=f"{pct}% ({method})",
        )

    def _label(self, score: float) -> str:
        if score >= 0.90: return "High"
        if score >= 0.75: return "Good"
        if score >= 0.55: return "Moderate"
        if score >= 0.35: return "Low"
        return "Very low"

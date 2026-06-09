"""
core/abstention.py — calibrated abstention + self-consistency.

For a security tool, calibrated honesty IS capability: a confident wrong answer
is worse than "I'm not sure, here's why." Two reusable helpers that
disproportionately help a small local model:

  * self_consistency(generate_fn, n, parse_fn): sample the model n times and
    take the majority answer; the agreement fraction is a free confidence
    signal — if the model disagrees with itself, that's a flag.
  * should_abstain / CalibratedDecider: turn a confidence (or agreement) below
    a threshold into an explicit abstention instead of a guess.

Pure stdlib. Deterministic given a deterministic generate_fn. Never raises.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple


@dataclass
class ConsensusResult:
    answer: Optional[Any]          # majority answer (parsed), or None
    agreement: float               # fraction of samples agreeing with the winner
    n: int                         # number of samples actually collected
    samples: List[Any] = field(default_factory=list)
    distribution: dict = field(default_factory=dict)


def self_consistency(generate_fn: Callable[[], str], n: int = 3,
                     parse_fn: Optional[Callable[[str], Any]] = None
                     ) -> ConsensusResult:
    """Call generate_fn() n times, parse each, return the majority answer plus
    the agreement fraction. parse_fn maps raw text -> a hashable key (default:
    the stripped lowercased text). Samples that raise or parse to None are
    dropped from the vote but counted in n."""
    parse_fn = parse_fn or (lambda s: (s or "").strip().lower())
    raw_samples: List[Any] = []
    keys: List[Any] = []
    for _ in range(max(1, int(n))):
        try:
            raw = generate_fn()
        except Exception:
            raw = None
        raw_samples.append(raw)
        if raw is None:
            continue
        try:
            k = parse_fn(raw)
        except Exception:
            k = None
        if k is not None and k != "":
            keys.append(k)

    if not keys:
        return ConsensusResult(answer=None, agreement=0.0, n=len(raw_samples),
                               samples=raw_samples, distribution={})
    counts = Counter(keys)
    winner, top = counts.most_common(1)[0]
    agreement = top / len(raw_samples)
    return ConsensusResult(answer=winner, agreement=agreement,
                           n=len(raw_samples), samples=raw_samples,
                           distribution=dict(counts))


def should_abstain(confidence: float, threshold: float = 0.5) -> bool:
    """True if confidence is below the abstention threshold."""
    try:
        return float(confidence) < float(threshold)
    except (TypeError, ValueError):
        return True


@dataclass
class Decision:
    answer: Optional[Any]
    abstained: bool
    confidence: float
    reason: str = ""


class CalibratedDecider:
    """Combine a confidence signal (and optionally self-consistency agreement)
    into an answer-or-abstain decision."""

    def __init__(self, threshold: float = 0.5, abstain_message: str = (
            "Not confident enough to answer reliably — recommend manual review.")):
        self.threshold = float(threshold)
        self.abstain_message = abstain_message

    def decide(self, answer: Any, confidence: float) -> Decision:
        if should_abstain(confidence, self.threshold):
            return Decision(answer=None, abstained=True, confidence=confidence,
                            reason=self.abstain_message)
        return Decision(answer=answer, abstained=False, confidence=confidence)

    def decide_by_consensus(self, generate_fn: Callable[[], str], n: int = 3,
                            parse_fn: Optional[Callable[[str], Any]] = None
                            ) -> Decision:
        c = self_consistency(generate_fn, n=n, parse_fn=parse_fn)
        if c.answer is None or should_abstain(c.agreement, self.threshold):
            return Decision(answer=None, abstained=True, confidence=c.agreement,
                            reason=(f"{self.abstain_message} "
                                    f"(self-consistency {c.agreement:.0%} over {c.n})"))
        return Decision(answer=c.answer, abstained=False, confidence=c.agreement,
                        reason=f"self-consistency {c.agreement:.0%}")

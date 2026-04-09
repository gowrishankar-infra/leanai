"""
LeanAI Phase 5a — Swarm Consensus
Runs multiple inference passes with different parameters,
compares answers, and returns the one with highest agreement.

This genuinely improves accuracy: if 3/3 models agree, confidence is near-certain.
If they disagree, we know the answer is uncertain.
"""

import time
import difflib
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any


@dataclass
class SwarmCandidate:
    """One inference result from the swarm."""
    text: str
    temperature: float
    latency_ms: float
    agreement_score: float = 0.0  # how much this agrees with others (0-1)


@dataclass
class SwarmResult:
    """Result of a swarm consensus vote."""
    best_answer: str
    consensus_score: float       # 0-1, how much the candidates agreed
    confidence: float            # 0-100, boosted confidence
    candidates: List[SwarmCandidate] = field(default_factory=list)
    total_latency_ms: float = 0.0
    num_passes: int = 0
    unanimous: bool = False      # True if all candidates are nearly identical

    def summary(self) -> str:
        status = "UNANIMOUS" if self.unanimous else f"{self.consensus_score:.0%} agreement"
        lines = [
            f"Swarm: {self.num_passes} passes | {status}",
            f"Confidence: {self.confidence:.0f}% | Time: {self.total_latency_ms:.0f}ms",
        ]
        if len(self.candidates) > 1:
            lines.append("Candidates:")
            for i, c in enumerate(self.candidates):
                marker = "→" if c.text == self.best_answer else " "
                preview = c.text[:80].replace("\n", " ")
                lines.append(f"  {marker} [{i+1}] t={c.temperature} agree={c.agreement_score:.0%} | {preview}...")
        return "\n".join(lines)


def text_similarity(a: str, b: str) -> float:
    """Compute similarity between two texts (0-1).
    Uses SequenceMatcher for robust comparison."""
    if not a or not b:
        return 0.0
    # Normalize whitespace
    a_norm = " ".join(a.lower().split())
    b_norm = " ".join(b.lower().split())
    if a_norm == b_norm:
        return 1.0
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def extract_core_answer(text: str) -> str:
    """Extract the core factual answer from a response, ignoring filler.
    Useful for comparing answers that have different explanations but same conclusion."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return text
    first = lines[0]
    # Try to extract first sentence (core answer is usually in the first sentence)
    for delim in [". ", ".\n", "!"]:
        idx = first.find(delim)
        if idx != -1:
            return first[:idx + 1]
    # No sentence break found — return first line, truncated if needed
    if len(first) > 100:
        return first[:100]
    return first


class SwarmConsensus:
    """
    Runs multiple inference passes and picks the best answer via consensus.

    Usage:
        swarm = SwarmConsensus(model_fn=my_model_call)
        result = swarm.query("What is the capital of France?")
        print(result.best_answer)  # "The capital of France is Paris."
        print(result.confidence)    # 99.0
    """

    # Temperature presets for each pass
    DEFAULT_TEMPS = [0.1, 0.3, 0.5]

    def __init__(
        self,
        model_fn: Optional[Callable] = None,
        num_passes: int = 3,
        temperatures: Optional[List[float]] = None,
        similarity_threshold: float = 0.7,
        verbose: bool = False,
    ):
        """
        Args:
            model_fn: function(prompt: str, temperature: float) -> str
            num_passes: number of inference passes (default 3)
            temperatures: list of temperatures for each pass
            similarity_threshold: min similarity to count as "agreeing"
            verbose: print debug info
        """
        self.model_fn = model_fn
        self.num_passes = num_passes
        self.temperatures = temperatures or self.DEFAULT_TEMPS[:num_passes]
        self.similarity_threshold = similarity_threshold
        self.verbose = verbose

        # Ensure we have enough temperatures
        while len(self.temperatures) < self.num_passes:
            self.temperatures.append(self.temperatures[-1] + 0.1)

    def _run_single_pass(self, prompt: str, temperature: float) -> SwarmCandidate:
        """Run one inference pass and return the candidate."""
        start = time.time()
        text = self.model_fn(prompt, temperature)
        elapsed = (time.time() - start) * 1000
        return SwarmCandidate(
            text=text.strip(),
            temperature=temperature,
            latency_ms=elapsed,
        )

    def _compute_agreement(self, candidates: List[SwarmCandidate]) -> List[SwarmCandidate]:
        """Compute agreement scores: how much each candidate agrees with others."""
        n = len(candidates)
        if n <= 1:
            for c in candidates:
                c.agreement_score = 1.0
            return candidates

        for i, ci in enumerate(candidates):
            # Compare core answer against all others
            core_i = extract_core_answer(ci.text)
            similarities = []
            for j, cj in enumerate(candidates):
                if i == j:
                    continue
                core_j = extract_core_answer(cj.text)
                sim = text_similarity(core_i, core_j)
                similarities.append(sim)
            # Agreement = average similarity to all others
            ci.agreement_score = sum(similarities) / len(similarities) if similarities else 1.0

        return candidates

    def _pick_best(self, candidates: List[SwarmCandidate]) -> SwarmCandidate:
        """Pick the best candidate based on agreement score.
        If tied, prefer the lower-temperature (more precise) answer."""
        if not candidates:
            return SwarmCandidate(text="", temperature=0.0, latency_ms=0.0)
        return max(candidates, key=lambda c: (c.agreement_score, -c.temperature))

    def _compute_consensus(self, candidates: List[SwarmCandidate]) -> float:
        """Compute overall consensus score (0-1)."""
        if not candidates:
            return 0.0
        scores = [c.agreement_score for c in candidates]
        return sum(scores) / len(scores)

    def _is_unanimous(self, candidates: List[SwarmCandidate]) -> bool:
        """Check if all candidates essentially agree."""
        if len(candidates) <= 1:
            return True
        for c in candidates:
            if c.agreement_score < self.similarity_threshold:
                return False
        return True

    def query(self, prompt: str, base_confidence: float = 50.0) -> SwarmResult:
        """
        Run swarm consensus on a query.

        Args:
            prompt: the prompt to send to the model
            base_confidence: confidence from the original single-pass response

        Returns:
            SwarmResult with best answer, consensus score, and boosted confidence
        """
        if self.model_fn is None:
            raise RuntimeError("No model_fn provided to SwarmConsensus")

        total_start = time.time()
        candidates = []

        for i in range(self.num_passes):
            temp = self.temperatures[i]
            if self.verbose:
                print(f"  [Swarm] Pass {i+1}/{self.num_passes} (t={temp})...", flush=True)
            candidate = self._run_single_pass(prompt, temp)
            candidates.append(candidate)

        # Compute agreement
        candidates = self._compute_agreement(candidates)

        # Pick best
        best = self._pick_best(candidates)
        consensus = self._compute_consensus(candidates)
        unanimous = self._is_unanimous(candidates)

        # Boost confidence based on consensus
        # Formula: start from base confidence, boost proportional to agreement
        if unanimous:
            boosted = max(base_confidence, 95.0)
        elif consensus >= 0.8:
            boosted = max(base_confidence, 85.0)
        elif consensus >= 0.6:
            boosted = max(base_confidence, 75.0)
        else:
            # Low consensus = lower confidence (answers disagree)
            boosted = min(base_confidence, 60.0)

        total_ms = (time.time() - total_start) * 1000

        return SwarmResult(
            best_answer=best.text,
            consensus_score=consensus,
            confidence=boosted,
            candidates=candidates,
            total_latency_ms=total_ms,
            num_passes=self.num_passes,
            unanimous=unanimous,
        )

    def query_with_existing(self, first_answer: str, prompt: str,
                            base_confidence: float = 50.0) -> SwarmResult:
        """
        Use an existing first answer as pass 1, then run additional passes.
        This avoids re-running the first inference when /swarm is called after a response.

        Args:
            first_answer: the response already generated
            prompt: the prompt to use for additional passes
            base_confidence: confidence of the first response
        """
        if self.model_fn is None:
            raise RuntimeError("No model_fn provided to SwarmConsensus")

        total_start = time.time()

        # First candidate is the existing answer
        candidates = [
            SwarmCandidate(text=first_answer.strip(), temperature=self.temperatures[0], latency_ms=0)
        ]

        # Run remaining passes
        for i in range(1, self.num_passes):
            temp = self.temperatures[i]
            if self.verbose:
                print(f"  [Swarm] Pass {i+1}/{self.num_passes} (t={temp})...", flush=True)
            candidate = self._run_single_pass(prompt, temp)
            candidates.append(candidate)

        # Same consensus logic
        candidates = self._compute_agreement(candidates)
        best = self._pick_best(candidates)
        consensus = self._compute_consensus(candidates)
        unanimous = self._is_unanimous(candidates)

        if unanimous:
            boosted = max(base_confidence, 95.0)
        elif consensus >= 0.8:
            boosted = max(base_confidence, 85.0)
        elif consensus >= 0.6:
            boosted = max(base_confidence, 75.0)
        else:
            boosted = min(base_confidence, 60.0)

        total_ms = (time.time() - total_start) * 1000

        return SwarmResult(
            best_answer=best.text,
            consensus_score=consensus,
            confidence=boosted,
            candidates=candidates,
            total_latency_ms=total_ms,
            num_passes=self.num_passes,
            unanimous=unanimous,
        )

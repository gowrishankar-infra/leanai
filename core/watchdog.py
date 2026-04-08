"""
LeanAI · Metacognitive Watchdog
The most important innovation in Phase 0.

Monitors token-level confidence (entropy) during generation.
When entropy spikes — the model is uncertain — the watchdog:
  1. Flags the uncertain span
  2. Optionally triggers the verifier
  3. Can pause and request clarification
  4. Adjusts final confidence score

No other production AI does this. They output with equal confidence
whether right or wrong. Ours knows the difference.
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from collections import deque


@dataclass
class WatchdogState:
    """Live state tracked during a single generation."""
    token_count: int = 0
    entropy_history: list = field(default_factory=list)
    spike_positions: list = field(default_factory=list)
    max_entropy: float = 0.0
    mean_entropy: float = 0.0
    final_confidence: float = 1.0
    should_verify: bool = False
    should_pause: bool = False
    warning_message: Optional[str] = None


class MetaCognitiveWatchdog:
    """
    Monitors generation quality in real time by tracking token probability entropy.

    Entropy measures how "spread out" the model's probability distribution is:
      - Low entropy  → model is confident (probability mass on one token)
      - High entropy → model is uncertain (probability spread across many tokens)

    Thresholds (tunable):
      SPIKE_THRESHOLD  — single token entropy that triggers a spike flag
      MEAN_THRESHOLD   — average entropy that triggers overall low confidence
      VERIFY_THRESHOLD — entropy level that should trigger neurosymbolic verifier
    """

    SPIKE_THRESHOLD  = 3.5   # nats — high single-token uncertainty
    MEAN_THRESHOLD   = 2.5   # nats — sustained uncertainty
    VERIFY_THRESHOLD = 3.0   # nats — trigger formal verifier
    PAUSE_THRESHOLD  = 4.2   # nats — so uncertain, should ask user

    # Rolling window for smoothed entropy
    WINDOW_SIZE = 10

    def __init__(self):
        self._window = deque(maxlen=self.WINDOW_SIZE)

    def new_generation(self) -> WatchdogState:
        """Call this before starting each new generation."""
        self._window.clear()
        return WatchdogState()

    def observe_token(
        self,
        state: WatchdogState,
        token_id: int,
        logprobs: list[float],   # log probabilities for top-k tokens
        position: int,
    ) -> None:
        """
        Called for each generated token.
        
        Args:
            state: The current WatchdogState (mutated in place)
            token_id: The selected token
            logprobs: Log probabilities of top-k candidates
            position: Token position in output
        """
        entropy = self._entropy_from_logprobs(logprobs)
        state.token_count += 1
        state.entropy_history.append(entropy)
        self._window.append(entropy)

        # Track max
        if entropy > state.max_entropy:
            state.max_entropy = entropy

        # Spike detection
        if entropy > self.SPIKE_THRESHOLD:
            state.spike_positions.append(position)
            if entropy > self.VERIFY_THRESHOLD:
                state.should_verify = True
            if entropy > self.PAUSE_THRESHOLD:
                state.should_pause = True

    def finalize(self, state: WatchdogState) -> WatchdogState:
        """
        Call after generation completes.
        Computes final confidence score and sets warning if needed.
        """
        if not state.entropy_history:
            state.final_confidence = 1.0
            return state

        state.mean_entropy = sum(state.entropy_history) / len(state.entropy_history)

        # Convert mean entropy to confidence score (0.0–1.0)
        # A mean entropy of 0 → confidence 1.0
        # A mean entropy of 4+ → confidence ~0.1
        state.final_confidence = max(0.05, 1.0 - (state.mean_entropy / 5.0))

        # Set human-readable warning
        if state.should_pause:
            state.warning_message = (
                "I'm quite uncertain about parts of this answer. "
                "Please verify independently or ask me to clarify."
            )
        elif state.should_verify:
            state.warning_message = (
                "Some parts of this answer had high uncertainty — "
                "running formal verification."
            )
        elif state.mean_entropy > self.MEAN_THRESHOLD:
            state.warning_message = (
                "Moderate confidence. I recommend double-checking key facts."
            )

        return state

    def confidence_label(self, confidence: float) -> str:
        """Human-readable confidence label."""
        if confidence >= 0.85: return "High confidence"
        if confidence >= 0.65: return "Moderate confidence"
        if confidence >= 0.40: return "Low confidence"
        return "Very uncertain — please verify"

    def _entropy_from_logprobs(self, logprobs: list[float]) -> float:
        """
        Compute Shannon entropy from log probabilities.
        H = -∑ p(x) * log(p(x))
        """
        if not logprobs:
            return 0.0

        # Convert logprobs to probs and normalize
        max_lp = max(logprobs)
        probs = [math.exp(lp - max_lp) for lp in logprobs]
        total = sum(probs)
        if total == 0:
            return 0.0
        probs = [p / total for p in probs]

        # Shannon entropy in nats
        entropy = 0.0
        for p in probs:
            if p > 1e-10:
                entropy -= p * math.log(p)

        return entropy

    # ── Simulation mode (Phase 0: when we don't have real logprobs) ──

    def simulate_from_response(
        self,
        response: str,
        domain: str = "general",
    ) -> WatchdogState:
        """
        Phase 0 fallback: estimate confidence from response characteristics
        when real token logprobs aren't available.

        Uses heuristics:
        - Hedging language → lower confidence
        - Factual claims without qualifiers → medium confidence
        - Math/code with verifiable structure → high confidence
        """
        state = WatchdogState()
        state.token_count = len(response.split())

        # Hedging signals → high entropy simulation
        hedge_phrases = [
            "i think", "i believe", "i'm not sure", "might be",
            "could be", "possibly", "probably", "i'm not certain",
            "you should verify", "approximately", "roughly",
        ]
        hedge_count = sum(1 for p in hedge_phrases if p in response.lower())

        # Confident signals → low entropy simulation
        confident_signals = [
            "the answer is", "equals", "=", "therefore", "thus",
            "definitively", "confirmed", "proven",
        ]
        confident_count = sum(1 for s in confident_signals if s in response.lower())

        # Base entropy estimate
        base_entropy = 2.0
        base_entropy += hedge_count * 0.4
        base_entropy -= confident_count * 0.3
        base_entropy = max(0.5, min(4.5, base_entropy))

        # Simulate token-level entropy (approximate)
        words = response.split()
        for i, word in enumerate(words):
            # Words that follow hedges get higher entropy
            local_entropy = base_entropy + (
                0.5 if i > 0 and any(
                    p in " ".join(words[max(0,i-3):i]).lower()
                    for p in ["not sure", "think", "believe", "maybe"]
                ) else 0.0
            )
            self.observe_token(state, i, [-local_entropy, -local_entropy-1.0, -local_entropy-2.0], i)

        return self.finalize(state)

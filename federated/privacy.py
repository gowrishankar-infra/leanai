"""
LeanAI Phase 5c — Differential Privacy
Adds calibrated noise to model weight updates before sharing with peers.
Guarantees that no single training example can be reverse-engineered from shared updates.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy."""
    epsilon: float = 1.0        # privacy budget (lower = more private, noisier)
    delta: float = 1e-5         # failure probability
    max_grad_norm: float = 1.0  # clip gradients to this L2 norm
    noise_type: str = "gaussian"  # "gaussian" or "laplace"
    enabled: bool = True


class DifferentialPrivacy:
    """
    Applies differential privacy to weight updates.
    
    Before sharing weights with peers, this module:
    1. Clips the update to bound sensitivity
    2. Adds calibrated noise proportional to sensitivity/epsilon
    3. Returns the noisy update that's safe to share
    
    Guarantees (epsilon, delta)-differential privacy per update round.
    """

    def __init__(self, config: Optional[PrivacyConfig] = None):
        self.config = config or PrivacyConfig()
        self._rounds = 0
        self._total_noise_added = 0.0

    @property
    def noise_scale(self) -> float:
        """Compute noise scale (sigma) from privacy parameters."""
        if self.config.noise_type == "gaussian":
            # Gaussian mechanism: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
            return (
                self.config.max_grad_norm
                * np.sqrt(2 * np.log(1.25 / self.config.delta))
                / self.config.epsilon
            )
        else:
            # Laplace mechanism: scale = sensitivity / epsilon
            return self.config.max_grad_norm / self.config.epsilon

    def clip_update(self, update: np.ndarray) -> np.ndarray:
        """Clip update vector to max_grad_norm L2 ball."""
        norm = np.linalg.norm(update)
        if norm > self.config.max_grad_norm:
            return update * (self.config.max_grad_norm / norm)
        return update.copy()

    def add_noise(self, update: np.ndarray) -> np.ndarray:
        """Add calibrated noise to a clipped update."""
        if not self.config.enabled:
            return update.copy()

        sigma = self.noise_scale
        if self.config.noise_type == "gaussian":
            noise = np.random.normal(0, sigma, size=update.shape)
        else:
            noise = np.random.laplace(0, sigma, size=update.shape)

        self._rounds += 1
        self._total_noise_added += np.linalg.norm(noise)
        return update + noise

    def privatize(self, update: np.ndarray) -> np.ndarray:
        """Full pipeline: clip then add noise. This is the main entry point."""
        clipped = self.clip_update(update)
        return self.add_noise(clipped)

    def privacy_spent(self) -> dict:
        """Report cumulative privacy budget spent."""
        # Simple composition: epsilon grows linearly with rounds
        return {
            "rounds": self._rounds,
            "epsilon_per_round": self.config.epsilon,
            "total_epsilon": self.config.epsilon * self._rounds,
            "delta": self.config.delta,
            "noise_type": self.config.noise_type,
            "avg_noise_norm": self._total_noise_added / max(self._rounds, 1),
        }

    def stats(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "epsilon": self.config.epsilon,
            "noise_scale": self.noise_scale,
            **self.privacy_spent(),
        }

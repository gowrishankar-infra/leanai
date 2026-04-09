"""
LeanAI Phase 5c — Weight Aggregator
Aggregates model weight updates from multiple federated peers.

Strategies:
  - FedAvg: simple weighted average (McMahan et al., 2017)
  - Median: coordinate-wise median (Byzantine-robust)
  - Trimmed Mean: remove outliers then average
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class AggregationStrategy(str, Enum):
    FEDAVG = "fedavg"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"


@dataclass
class PeerUpdate:
    """A weight update received from a peer."""
    peer_id: str
    weights: np.ndarray          # the weight update vector
    num_samples: int = 1         # how many training samples produced this update
    round_number: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "weights": self.weights.tolist(),
            "num_samples": self.num_samples,
            "round_number": self.round_number,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PeerUpdate":
        return cls(
            peer_id=d["peer_id"],
            weights=np.array(d["weights"], dtype=np.float32),
            num_samples=d.get("num_samples", 1),
            round_number=d.get("round_number", 0),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class AggregationResult:
    """Result of aggregating peer updates."""
    aggregated_weights: np.ndarray
    num_peers: int
    strategy: str
    total_samples: int
    weight_divergence: float = 0.0  # how much peers disagreed

    def summary(self) -> str:
        return (
            f"Aggregated {self.num_peers} peers ({self.total_samples} samples) "
            f"via {self.strategy} | divergence={self.weight_divergence:.4f}"
        )


class WeightAggregator:
    """
    Aggregates weight updates from multiple federated peers.
    
    Usage:
        agg = WeightAggregator(strategy="fedavg")
        agg.submit(PeerUpdate(peer_id="node1", weights=w1, num_samples=100))
        agg.submit(PeerUpdate(peer_id="node2", weights=w2, num_samples=50))
        result = agg.aggregate()
        new_weights = result.aggregated_weights
    """

    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.FEDAVG,
        trim_fraction: float = 0.1,
        min_peers: int = 1,
    ):
        self.strategy = strategy
        self.trim_fraction = trim_fraction
        self.min_peers = min_peers
        self._updates: List[PeerUpdate] = []
        self._round = 0

    def submit(self, update: PeerUpdate):
        """Submit a weight update from a peer."""
        self._updates.append(update)

    def clear(self):
        """Clear all submitted updates."""
        self._updates = []

    @property
    def num_pending(self) -> int:
        return len(self._updates)

    def _fedavg(self, updates: List[PeerUpdate]) -> np.ndarray:
        """Federated Averaging: weighted average by number of samples."""
        total_samples = sum(u.num_samples for u in updates)
        if total_samples == 0:
            # Equal weight if no sample counts
            return np.mean([u.weights for u in updates], axis=0)
        result = np.zeros_like(updates[0].weights, dtype=np.float64)
        for u in updates:
            weight = u.num_samples / total_samples
            result += weight * u.weights
        return result.astype(np.float32)

    def _median(self, updates: List[PeerUpdate]) -> np.ndarray:
        """Coordinate-wise median: robust to Byzantine peers."""
        stacked = np.stack([u.weights for u in updates])
        return np.median(stacked, axis=0).astype(np.float32)

    def _trimmed_mean(self, updates: List[PeerUpdate]) -> np.ndarray:
        """Trimmed mean: remove top and bottom fraction, then average."""
        stacked = np.stack([u.weights for u in updates])
        n = len(updates)
        trim = max(1, int(n * self.trim_fraction))
        if n <= 2 * trim:
            # Not enough peers to trim, fall back to median
            return np.median(stacked, axis=0).astype(np.float32)
        sorted_stack = np.sort(stacked, axis=0)
        trimmed = sorted_stack[trim : n - trim]
        return np.mean(trimmed, axis=0).astype(np.float32)

    def _compute_divergence(self, updates: List[PeerUpdate], result: np.ndarray) -> float:
        """Compute how much peers disagreed (avg L2 distance from aggregate)."""
        if len(updates) <= 1:
            return 0.0
        distances = [np.linalg.norm(u.weights - result) for u in updates]
        return float(np.mean(distances))

    def aggregate(self) -> AggregationResult:
        """Aggregate all submitted updates and return the result."""
        if len(self._updates) < self.min_peers:
            raise ValueError(
                f"Need at least {self.min_peers} peers, got {len(self._updates)}"
            )

        updates = self._updates
        self._round += 1

        if self.strategy == AggregationStrategy.FEDAVG:
            agg = self._fedavg(updates)
        elif self.strategy == AggregationStrategy.MEDIAN:
            agg = self._median(updates)
        elif self.strategy == AggregationStrategy.TRIMMED_MEAN:
            agg = self._trimmed_mean(updates)
        else:
            agg = self._fedavg(updates)

        divergence = self._compute_divergence(updates, agg)
        total_samples = sum(u.num_samples for u in updates)

        self.clear()

        return AggregationResult(
            aggregated_weights=agg,
            num_peers=len(updates),
            strategy=self.strategy.value,
            total_samples=total_samples,
            weight_divergence=divergence,
        )

    def stats(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "pending_updates": self.num_pending,
            "rounds_completed": self._round,
            "min_peers": self.min_peers,
        }

"""
LeanAI Phase 5c — Federated Node
Represents a single LeanAI instance in the federated learning network.

Each node:
  1. Collects training data locally (from user interactions)
  2. Generates local weight updates (simulated until LoRA is added)
  3. Applies differential privacy before sharing
  4. Shares privatized updates with peers
  5. Receives and aggregates updates from peers
  6. Applies the aggregated update to improve its model
"""

import os
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from federated.privacy import DifferentialPrivacy, PrivacyConfig
from federated.aggregator import (
    WeightAggregator,
    AggregationStrategy,
    PeerUpdate,
    AggregationResult,
)


@dataclass
class NodeConfig:
    """Configuration for a federated learning node."""
    node_id: str = ""                    # unique ID (auto-generated if empty)
    node_name: str = "LeanAI Node"
    data_dir: str = ""                   # where to store federated data
    weight_dim: int = 1024               # dimension of weight vectors
    min_peers: int = 2                   # minimum peers before aggregation
    aggregation_strategy: str = "fedavg"
    privacy_epsilon: float = 1.0
    privacy_enabled: bool = True
    auto_participate: bool = True        # automatically join federated rounds
    rounds_between_sync: int = 5         # how many local training rounds between syncs


@dataclass
class FederatedRound:
    """Record of one federated learning round."""
    round_number: int
    timestamp: float
    num_peers: int
    strategy: str
    divergence: float
    local_samples: int
    privacy_epsilon: float


class FederatedNode:
    """
    A single node in the LeanAI federated learning network.
    
    Usage:
        node = FederatedNode()
        
        # After local training produces an update
        node.submit_local_update(weight_delta, num_samples=100)
        
        # Receive updates from peers (via HTTP or direct)
        node.receive_peer_update(peer_id="node2", weights=w2, num_samples=50)
        
        # When enough peers have contributed, aggregate
        if node.ready_to_aggregate():
            result = node.aggregate_round()
            # result.aggregated_weights can be applied to model
    """

    def __init__(self, config: Optional[NodeConfig] = None):
        self.config = config or NodeConfig()

        # Auto-generate node ID from machine info
        if not self.config.node_id:
            self.config.node_id = self._generate_node_id()

        # Setup data directory
        if not self.config.data_dir:
            self.config.data_dir = str(
                Path.home() / ".leanai" / "federated" / self.config.node_id[:8]
            )
        os.makedirs(self.config.data_dir, exist_ok=True)

        # Initialize components
        self.privacy = DifferentialPrivacy(PrivacyConfig(
            epsilon=self.config.privacy_epsilon,
            enabled=self.config.privacy_enabled,
        ))

        self.aggregator = WeightAggregator(
            strategy=AggregationStrategy(self.config.aggregation_strategy),
            min_peers=self.config.min_peers,
        )

        # State
        self._round = 0
        self._local_update: Optional[np.ndarray] = None
        self._local_samples: int = 0
        self._history: List[FederatedRound] = []
        self._peers: Dict[str, dict] = {}  # known peers
        self._aggregated_weights: Optional[np.ndarray] = None

        # Load history if exists
        self._load_state()

    def _generate_node_id(self) -> str:
        """Generate a unique node ID from machine characteristics."""
        import platform
        info = f"{platform.node()}-{platform.machine()}-{os.getpid()}-{time.time()}"
        return hashlib.sha256(info.encode()).hexdigest()[:16]

    def _state_path(self) -> str:
        return os.path.join(self.config.data_dir, "node_state.json")

    def _load_state(self):
        """Load node state from disk."""
        path = self._state_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    state = json.load(f)
                self._round = state.get("round", 0)
                self._history = [
                    FederatedRound(**r) for r in state.get("history", [])
                ]
                self._peers = state.get("peers", {})
            except (json.JSONDecodeError, Exception):
                pass

    def _save_state(self):
        """Persist node state to disk."""
        state = {
            "node_id": self.config.node_id,
            "node_name": self.config.node_name,
            "round": self._round,
            "history": [
                {
                    "round_number": r.round_number,
                    "timestamp": r.timestamp,
                    "num_peers": r.num_peers,
                    "strategy": r.strategy,
                    "divergence": r.divergence,
                    "local_samples": r.local_samples,
                    "privacy_epsilon": r.privacy_epsilon,
                }
                for r in self._history
            ],
            "peers": self._peers,
        }
        with open(self._state_path(), "w") as f:
            json.dump(state, f, indent=2)

    # ── Local training updates ────────────────────────────────────

    def submit_local_update(self, weights: np.ndarray, num_samples: int = 1):
        """Submit a local weight update from training."""
        self._local_update = weights.copy()
        self._local_samples = num_samples

    def generate_simulated_update(self, num_samples: int = 50) -> np.ndarray:
        """
        Generate a simulated weight update for testing.
        In production, this would come from actual LoRA fine-tuning.
        """
        # Small random update centered around zero (like a real gradient)
        update = np.random.randn(self.config.weight_dim).astype(np.float32) * 0.01
        self._local_update = update
        self._local_samples = num_samples
        return update

    def get_privatized_update(self) -> Optional[PeerUpdate]:
        """Get the local update with differential privacy applied, ready to share."""
        if self._local_update is None:
            return None

        # Apply differential privacy
        private_update = self.privacy.privatize(self._local_update)

        return PeerUpdate(
            peer_id=self.config.node_id,
            weights=private_update,
            num_samples=self._local_samples,
            round_number=self._round,
            timestamp=time.time(),
        )

    # ── Peer communication ────────────────────────────────────────

    def receive_peer_update(self, peer_id: str, weights: np.ndarray,
                            num_samples: int = 1, round_number: int = 0):
        """Receive a weight update from a peer."""
        update = PeerUpdate(
            peer_id=peer_id,
            weights=weights,
            num_samples=num_samples,
            round_number=round_number,
            timestamp=time.time(),
        )
        self.aggregator.submit(update)

        # Track this peer
        self._peers[peer_id] = {
            "last_seen": time.time(),
            "rounds_participated": self._peers.get(peer_id, {}).get("rounds_participated", 0) + 1,
        }

    def register_peer(self, peer_id: str, info: Optional[dict] = None):
        """Register a known peer."""
        self._peers[peer_id] = {
            "last_seen": time.time(),
            "rounds_participated": 0,
            **(info or {}),
        }

    @property
    def known_peers(self) -> Dict[str, dict]:
        return dict(self._peers)

    @property
    def num_peers(self) -> int:
        return len(self._peers)

    # ── Aggregation ───────────────────────────────────────────────

    def ready_to_aggregate(self) -> bool:
        """Check if we have enough peer updates to aggregate."""
        # Count: local update (if any) + received peer updates
        total = self.aggregator.num_pending
        if self._local_update is not None:
            total += 1
        return total >= self.config.min_peers

    def aggregate_round(self) -> Optional[AggregationResult]:
        """
        Run one federated aggregation round.
        Combines local privatized update with peer updates.
        Returns the aggregated weights.
        """
        # Submit our own privatized update
        local_update = self.get_privatized_update()
        if local_update is not None:
            self.aggregator.submit(local_update)

        if self.aggregator.num_pending < self.config.min_peers:
            return None

        # Aggregate
        result = self.aggregator.aggregate()
        self._aggregated_weights = result.aggregated_weights

        # Record this round
        self._round += 1
        record = FederatedRound(
            round_number=self._round,
            timestamp=time.time(),
            num_peers=result.num_peers,
            strategy=result.strategy,
            divergence=result.weight_divergence,
            local_samples=self._local_samples,
            privacy_epsilon=self.config.privacy_epsilon,
        )
        self._history.append(record)

        # Reset local update
        self._local_update = None
        self._local_samples = 0

        # Save state
        self._save_state()

        return result

    @property
    def aggregated_weights(self) -> Optional[np.ndarray]:
        """Get the latest aggregated weights (to apply to model)."""
        return self._aggregated_weights

    # ── Export/Import for HTTP transfer ────────────────────────────

    def export_update(self) -> Optional[dict]:
        """Export the local privatized update as a JSON-serializable dict."""
        update = self.get_privatized_update()
        if update is None:
            return None
        return update.to_dict()

    def import_update(self, data: dict):
        """Import a peer update from a JSON dict (received via HTTP)."""
        update = PeerUpdate.from_dict(data)
        self.aggregator.submit(update)
        self._peers[update.peer_id] = {
            "last_seen": time.time(),
            "rounds_participated": self._peers.get(update.peer_id, {}).get("rounds_participated", 0) + 1,
        }

    # ── Status and info ───────────────────────────────────────────

    @property
    def round_number(self) -> int:
        return self._round

    def stats(self) -> dict:
        return {
            "node_id": self.config.node_id[:8],
            "node_name": self.config.node_name,
            "round": self._round,
            "known_peers": len(self._peers),
            "pending_updates": self.aggregator.num_pending,
            "has_local_update": self._local_update is not None,
            "local_samples": self._local_samples,
            "ready_to_aggregate": self.ready_to_aggregate(),
            "privacy": self.privacy.stats(),
            "aggregator": self.aggregator.stats(),
            "total_rounds": len(self._history),
        }

    def history_summary(self) -> str:
        if not self._history:
            return "No federated rounds completed yet."
        lines = [f"Federated learning — {len(self._history)} rounds completed"]
        for r in self._history[-5:]:  # last 5 rounds
            lines.append(
                f"  Round {r.round_number}: {r.num_peers} peers, "
                f"divergence={r.divergence:.4f}, ε={r.privacy_epsilon}"
            )
        return "\n".join(lines)

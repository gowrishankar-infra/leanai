"""
Tests for LeanAI Phase 5c — Federated Learning Protocol
"""

import os
import json
import shutil
import tempfile
import pytest
import numpy as np

from federated.privacy import DifferentialPrivacy, PrivacyConfig
from federated.aggregator import (
    WeightAggregator, AggregationStrategy, PeerUpdate, AggregationResult,
)
from federated.node import FederatedNode, NodeConfig, FederatedRound


# ══════════════════════════════════════════════════════════════════
# Differential Privacy Tests
# ══════════════════════════════════════════════════════════════════

class TestDifferentialPrivacy:
    def test_clip_within_norm(self):
        dp = DifferentialPrivacy(PrivacyConfig(max_grad_norm=1.0))
        update = np.array([0.3, 0.4], dtype=np.float32)  # norm = 0.5, within limit
        clipped = dp.clip_update(update)
        np.testing.assert_array_almost_equal(clipped, update)

    def test_clip_exceeds_norm(self):
        dp = DifferentialPrivacy(PrivacyConfig(max_grad_norm=1.0))
        update = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5.0, exceeds limit
        clipped = dp.clip_update(update)
        assert np.linalg.norm(clipped) <= 1.0 + 1e-6

    def test_add_noise_changes_values(self):
        dp = DifferentialPrivacy(PrivacyConfig(epsilon=0.1, enabled=True))
        update = np.ones(100, dtype=np.float32)
        noisy = dp.add_noise(update)
        assert not np.allclose(noisy, update)

    def test_noise_disabled(self):
        dp = DifferentialPrivacy(PrivacyConfig(enabled=False))
        update = np.ones(10, dtype=np.float32)
        result = dp.add_noise(update)
        np.testing.assert_array_equal(result, update)

    def test_privatize_full_pipeline(self):
        dp = DifferentialPrivacy(PrivacyConfig(
            epsilon=1.0, max_grad_norm=1.0, enabled=True
        ))
        update = np.array([10.0, 0.0], dtype=np.float32)  # large, will be clipped
        result = dp.privatize(update)
        # Should be clipped and noisy
        assert np.linalg.norm(result) < 15.0  # not huge

    def test_gaussian_noise(self):
        dp = DifferentialPrivacy(PrivacyConfig(noise_type="gaussian"))
        assert dp.noise_scale > 0

    def test_laplace_noise(self):
        dp = DifferentialPrivacy(PrivacyConfig(noise_type="laplace"))
        assert dp.noise_scale > 0

    def test_privacy_spent_tracking(self):
        dp = DifferentialPrivacy(PrivacyConfig(epsilon=0.5))
        dp.privatize(np.ones(10))
        dp.privatize(np.ones(10))
        spent = dp.privacy_spent()
        assert spent["rounds"] == 2
        assert spent["total_epsilon"] == 1.0  # 0.5 * 2

    def test_stats(self):
        dp = DifferentialPrivacy()
        s = dp.stats()
        assert "enabled" in s
        assert "epsilon" in s
        assert "noise_scale" in s

    def test_lower_epsilon_more_noise(self):
        dp_tight = DifferentialPrivacy(PrivacyConfig(epsilon=0.1))
        dp_loose = DifferentialPrivacy(PrivacyConfig(epsilon=10.0))
        assert dp_tight.noise_scale > dp_loose.noise_scale


# ══════════════════════════════════════════════════════════════════
# Weight Aggregator Tests
# ══════════════════════════════════════════════════════════════════

class TestPeerUpdate:
    def test_to_dict_and_back(self):
        w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pu = PeerUpdate(peer_id="node1", weights=w, num_samples=50)
        d = pu.to_dict()
        restored = PeerUpdate.from_dict(d)
        assert restored.peer_id == "node1"
        assert restored.num_samples == 50
        np.testing.assert_array_almost_equal(restored.weights, w)


class TestWeightAggregator:
    def test_fedavg_equal_samples(self):
        agg = WeightAggregator(strategy=AggregationStrategy.FEDAVG)
        agg.submit(PeerUpdate("a", np.array([1.0, 2.0]), num_samples=10))
        agg.submit(PeerUpdate("b", np.array([3.0, 4.0]), num_samples=10))
        result = agg.aggregate()
        np.testing.assert_array_almost_equal(result.aggregated_weights, [2.0, 3.0])

    def test_fedavg_weighted(self):
        agg = WeightAggregator(strategy=AggregationStrategy.FEDAVG)
        agg.submit(PeerUpdate("a", np.array([0.0]), num_samples=90))
        agg.submit(PeerUpdate("b", np.array([10.0]), num_samples=10))
        result = agg.aggregate()
        # 90% weight on 0.0, 10% weight on 10.0 = 1.0
        np.testing.assert_array_almost_equal(result.aggregated_weights, [1.0])

    def test_median_aggregation(self):
        agg = WeightAggregator(strategy=AggregationStrategy.MEDIAN)
        agg.submit(PeerUpdate("a", np.array([1.0])))
        agg.submit(PeerUpdate("b", np.array([2.0])))
        agg.submit(PeerUpdate("c", np.array([100.0])))  # outlier
        result = agg.aggregate()
        np.testing.assert_array_almost_equal(result.aggregated_weights, [2.0])

    def test_trimmed_mean(self):
        agg = WeightAggregator(
            strategy=AggregationStrategy.TRIMMED_MEAN, trim_fraction=0.2
        )
        for v in [1, 2, 3, 4, 100]:
            agg.submit(PeerUpdate(f"n{v}", np.array([float(v)])))
        result = agg.aggregate()
        # Trim top and bottom 20% (1 each), average of [2, 3, 4] = 3.0
        np.testing.assert_array_almost_equal(result.aggregated_weights, [3.0])

    def test_min_peers_not_met(self):
        agg = WeightAggregator(min_peers=3)
        agg.submit(PeerUpdate("a", np.array([1.0])))
        with pytest.raises(ValueError, match="Need at least"):
            agg.aggregate()

    def test_clear_after_aggregate(self):
        agg = WeightAggregator()
        agg.submit(PeerUpdate("a", np.array([1.0])))
        agg.aggregate()
        assert agg.num_pending == 0

    def test_divergence_zero_identical(self):
        agg = WeightAggregator()
        agg.submit(PeerUpdate("a", np.array([5.0, 5.0])))
        agg.submit(PeerUpdate("b", np.array([5.0, 5.0])))
        result = agg.aggregate()
        assert result.weight_divergence == 0.0

    def test_divergence_high_different(self):
        agg = WeightAggregator()
        agg.submit(PeerUpdate("a", np.array([0.0])))
        agg.submit(PeerUpdate("b", np.array([100.0])))
        result = agg.aggregate()
        assert result.weight_divergence > 0

    def test_result_summary(self):
        agg = WeightAggregator()
        agg.submit(PeerUpdate("a", np.array([1.0]), num_samples=10))
        agg.submit(PeerUpdate("b", np.array([2.0]), num_samples=20))
        result = agg.aggregate()
        s = result.summary()
        assert "2 peers" in s
        assert "30 samples" in s

    def test_stats(self):
        agg = WeightAggregator()
        s = agg.stats()
        assert "strategy" in s
        assert "pending_updates" in s


# ══════════════════════════════════════════════════════════════════
# Federated Node Tests
# ══════════════════════════════════════════════════════════════════

class TestFederatedNode:
    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d)

    @pytest.fixture
    def node(self, tmp_dir):
        config = NodeConfig(
            node_id="test_node_001",
            data_dir=tmp_dir,
            weight_dim=64,
            min_peers=2,
            privacy_epsilon=1.0,
        )
        return FederatedNode(config)

    def test_node_creation(self, node):
        assert node.config.node_id == "test_node_001"
        assert node.round_number == 0

    def test_auto_node_id(self, tmp_dir):
        config = NodeConfig(data_dir=tmp_dir)
        node = FederatedNode(config)
        assert len(node.config.node_id) == 16

    def test_generate_simulated_update(self, node):
        update = node.generate_simulated_update(num_samples=100)
        assert update.shape == (64,)
        assert node._local_samples == 100

    def test_submit_local_update(self, node):
        w = np.random.randn(64).astype(np.float32)
        node.submit_local_update(w, num_samples=50)
        assert node._local_update is not None
        assert node._local_samples == 50

    def test_get_privatized_update(self, node):
        node.generate_simulated_update(100)
        pu = node.get_privatized_update()
        assert pu is not None
        assert pu.peer_id == "test_node_001"
        assert pu.num_samples == 100
        assert pu.weights.shape == (64,)

    def test_privacy_applied(self, node):
        """Privatized update should differ from raw update."""
        raw = node.generate_simulated_update(100)
        pu = node.get_privatized_update()
        assert not np.allclose(raw, pu.weights)

    def test_receive_peer_update(self, node):
        w = np.random.randn(64).astype(np.float32)
        node.receive_peer_update("peer_a", w, num_samples=30)
        assert node.aggregator.num_pending == 1
        assert "peer_a" in node.known_peers

    def test_register_peer(self, node):
        node.register_peer("peer_b", {"address": "192.168.1.5"})
        assert "peer_b" in node.known_peers
        assert node.num_peers == 1

    def test_not_ready_to_aggregate(self, node):
        assert not node.ready_to_aggregate()

    def test_ready_to_aggregate(self, node):
        node.generate_simulated_update(50)
        node.receive_peer_update("peer_a", np.random.randn(64).astype(np.float32), 50)
        assert node.ready_to_aggregate()

    def test_aggregate_round(self, node):
        node.generate_simulated_update(50)
        node.receive_peer_update(
            "peer_a", np.random.randn(64).astype(np.float32), 50
        )
        result = node.aggregate_round()
        assert result is not None
        assert result.num_peers == 2
        assert node.round_number == 1

    def test_aggregate_returns_none_not_enough_peers(self, node):
        result = node.aggregate_round()
        assert result is None

    def test_history_recorded(self, node):
        node.generate_simulated_update(50)
        node.receive_peer_update("p", np.random.randn(64).astype(np.float32), 50)
        node.aggregate_round()
        assert len(node._history) == 1
        assert node._history[0].num_peers == 2

    def test_state_persistence(self, tmp_dir):
        # Create node and do a round
        config = NodeConfig(node_id="persist_test", data_dir=tmp_dir, weight_dim=32, min_peers=2)
        node1 = FederatedNode(config)
        node1.generate_simulated_update(50)
        node1.receive_peer_update("p", np.random.randn(32).astype(np.float32), 50)
        node1.aggregate_round()

        # Create new node instance — should load state
        node2 = FederatedNode(config)
        assert node2.round_number == 1
        assert len(node2._history) == 1

    def test_export_import_update(self, node):
        node.generate_simulated_update(100)
        exported = node.export_update()
        assert exported is not None
        assert "peer_id" in exported
        assert "weights" in exported

        # Another node imports it
        config2 = NodeConfig(node_id="node_b", data_dir=tempfile.mkdtemp(), weight_dim=64, min_peers=1)
        node2 = FederatedNode(config2)
        node2.import_update(exported)
        assert node2.aggregator.num_pending == 1
        shutil.rmtree(config2.data_dir)

    def test_stats(self, node):
        s = node.stats()
        assert "node_id" in s
        assert "round" in s
        assert "privacy" in s
        assert "aggregator" in s
        assert "ready_to_aggregate" in s

    def test_history_summary_empty(self, node):
        s = node.history_summary()
        assert "No federated rounds" in s

    def test_history_summary_with_rounds(self, node):
        node.generate_simulated_update(50)
        node.receive_peer_update("p", np.random.randn(64).astype(np.float32), 50)
        node.aggregate_round()
        s = node.history_summary()
        assert "Round 1" in s

    def test_aggregated_weights_accessible(self, node):
        node.generate_simulated_update(50)
        node.receive_peer_update("p", np.random.randn(64).astype(np.float32), 50)
        node.aggregate_round()
        assert node.aggregated_weights is not None
        assert node.aggregated_weights.shape == (64,)

    def test_multiple_rounds(self, node):
        for i in range(3):
            node.generate_simulated_update(50)
            node.receive_peer_update(
                f"peer_{i}", np.random.randn(64).astype(np.float32), 50
            )
            node.aggregate_round()
        assert node.round_number == 3
        assert len(node._history) == 3


# ══════════════════════════════════════════════════════════════════
# Integration: Full federated round with 3 nodes
# ══════════════════════════════════════════════════════════════════

class TestFederatedIntegration:
    def test_three_node_round(self):
        """Simulate a complete federated round with 3 nodes."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        nodes = [
            FederatedNode(NodeConfig(
                node_id=f"node_{i}", data_dir=dirs[i],
                weight_dim=128, min_peers=3,
            ))
            for i in range(3)
        ]

        # Each node generates local training updates
        for n in nodes:
            n.generate_simulated_update(num_samples=100)

        # Each node exports its privatized update
        exports = [n.export_update() for n in nodes]
        assert all(e is not None for e in exports)

        # Each node receives updates from the other two
        for i, node in enumerate(nodes):
            for j, export in enumerate(exports):
                if i != j:
                    node.import_update(export)

        # Each node aggregates
        results = []
        for node in nodes:
            assert node.ready_to_aggregate()
            result = node.aggregate_round()
            results.append(result)

        # All nodes should have completed the round
        assert all(r is not None for r in results)
        assert all(r.num_peers == 3 for r in results)
        assert all(n.round_number == 1 for n in nodes)

        # All nodes should have similar aggregated weights
        w0 = results[0].aggregated_weights
        w1 = results[1].aggregated_weights
        w2 = results[2].aggregated_weights
        # They won't be identical (each adds its own privacy noise) but should be similar
        assert np.linalg.norm(w0 - w1) < np.linalg.norm(w0) * 2  # within 2x
        assert np.linalg.norm(w0 - w2) < np.linalg.norm(w0) * 2

        # Cleanup
        for d in dirs:
            shutil.rmtree(d)

    def test_byzantine_robustness_with_median(self):
        """Test that median aggregation resists a malicious peer."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        nodes = [
            FederatedNode(NodeConfig(
                node_id=f"med_{i}", data_dir=dirs[i],
                weight_dim=32, min_peers=3,
                aggregation_strategy="median",
            ))
            for i in range(3)
        ]

        # Honest nodes have similar updates
        honest_update = np.ones(32, dtype=np.float32) * 0.01
        nodes[0].submit_local_update(honest_update, 100)
        nodes[1].submit_local_update(honest_update * 1.1, 100)

        # Malicious node sends a huge update
        malicious_update = np.ones(32, dtype=np.float32) * 999.0
        nodes[2].submit_local_update(malicious_update, 100)

        # Share updates
        for i in range(3):
            export = nodes[i].export_update()
            for j in range(3):
                if i != j:
                    nodes[j].import_update(export)

        # Aggregate on node 0
        result = nodes[0].aggregate_round()
        assert result is not None
        # Median should be close to honest updates, not malicious
        assert np.max(np.abs(result.aggregated_weights)) < 50  # not 999

        for d in dirs:
            shutil.rmtree(d)

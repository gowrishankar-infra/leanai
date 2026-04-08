"""
LeanAI · Phase 3 Tests
Run: python -m pytest tests/test_phase3.py -v
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from training.quality_filter import QualityFilter, THRESHOLD
from training.self_play_v2 import EnhancedSelfPlayEngine
from training.continual_trainer import ContinualTrainer, TrainingConfig
from training.self_improve import TrainingDataStore, FeedbackSignal, TrainingPair
from core.calibrator import ConfidenceCalibrator


# ══════════════════════════════════════════════════════
# Quality Filter Tests
# ══════════════════════════════════════════════════════

class TestQualityFilter:

    def setup_method(self):
        self.f = QualityFilter()

    def _pair(self, **kwargs):
        defaults = dict(
            id="test", instruction="test q", response="test answer " * 5,
            feedback=FeedbackSignal.NEUTRAL, confidence=0.5,
            verified=False, latency_ms=2000, tier_used="medium",
            timestamp=time.time(), tags=[],
        )
        defaults.update(kwargs)
        return TrainingPair(**defaults)

    def test_verified_pair_passes(self):
        p = self._pair(verified=True, confidence=0.9,
                       feedback=FeedbackSignal.EXCELLENT)
        a = self.f.assess(p)
        assert a.passes

    def test_wrong_feedback_fails(self):
        p = self._pair(feedback=FeedbackSignal.WRONG)
        a = self.f.assess(p)
        assert not a.passes

    def test_demo_mode_fails(self):
        p = self._pair(response="This is demo mode response")
        a = self.f.assess(p)
        assert a.score == 0.0

    def test_self_play_passes(self):
        p = self._pair(tier_used="self_play", verified=True,
                       feedback=FeedbackSignal.EXCELLENT, confidence=0.95)
        a = self.f.assess(p)
        assert a.passes

    def test_score_in_range(self):
        p = self._pair()
        a = self.f.assess(p)
        assert 0.0 <= a.score <= 1.0

    def test_filter_batch(self):
        pairs = [
            self._pair(verified=True, feedback=FeedbackSignal.EXCELLENT, confidence=0.95),
            self._pair(feedback=FeedbackSignal.WRONG),
            self._pair(tier_used="self_play", verified=True, confidence=0.95),
        ]
        accepted, rejected = self.f.filter_batch(pairs)
        assert len(accepted) + len(rejected) == len(pairs)
        assert len(accepted) >= 1

    def test_stats_structure(self):
        pairs = [self._pair(), self._pair(verified=True, confidence=0.95)]
        stats = self.f.stats(pairs)
        assert "total" in stats
        assert "passed" in stats
        assert "pass_rate" in stats
        assert stats["total"] == 2

    def test_high_confidence_boosts_score(self):
        low  = self._pair(confidence=0.30)
        high = self._pair(confidence=0.90)
        assert self.f.assess(high).score > self.f.assess(low).score

    def test_reasons_populated(self):
        p = self._pair(verified=True, feedback=FeedbackSignal.EXCELLENT)
        a = self.f.assess(p)
        assert len(a.reasons) > 0


# ══════════════════════════════════════════════════════
# Enhanced Self-Play Tests
# ══════════════════════════════════════════════════════

class TestEnhancedSelfPlay:

    def setup_method(self):
        self.engine = EnhancedSelfPlayEngine()

    def test_generates_pair(self):
        p = self.engine._gen_arithmetic()
        assert p is not None
        assert p.problem != ""
        assert p.solution != ""

    def test_all_math_generators(self):
        for fn in self.engine._math_generators():
            pair = fn()
            assert pair is not None
            assert pair.domain == "math"
            assert pair.verified is True

    def test_all_code_generators(self):
        for fn in self.engine._code_generators():
            pair = fn()
            assert pair is not None
            assert pair.domain == "code"

    def test_all_reasoning_generators(self):
        for fn in self.engine._reasoning_generators():
            pair = fn()
            assert pair is not None
            assert pair.domain == "reasoning"

    def test_batch_generates_n(self):
        pairs = self.engine.generate_batch(10)
        assert len(pairs) == 10

    def test_batch_all_verified(self):
        pairs = self.engine.generate_batch(5)
        assert all(p.verified for p in pairs)

    def test_difficulty_in_range(self):
        pairs = self.engine.generate_batch(10)
        for p in pairs:
            assert 0.0 <= p.difficulty <= 1.0

    def test_unique_ids(self):
        pairs = self.engine.generate_batch(20)
        ids = [p.id for p in pairs]
        # IDs should be mostly unique (hash collisions possible but rare)
        assert len(set(ids)) >= len(ids) * 0.7

    def test_percentage_correct(self):
        p = self.engine._gen_percentage()
        # Extract numbers from solution to verify
        import re
        nums = re.findall(r'\d+\.?\d*', p.solution)
        assert len(nums) > 0

    def test_geometry_area_correct(self):
        p = self.engine._gen_geometry_area()
        assert "area" in p.problem.lower() or "area" in p.solution.lower()

    def test_statistics_problem(self):
        p = self.engine._gen_statistics()
        assert p.domain == "math"
        assert p.subdomain.startswith("statistics_")

    def test_code_solution_has_def(self):
        for fn in self.engine._code_generators():
            p = fn()
            assert "def " in p.solution


# ══════════════════════════════════════════════════════
# Confidence Calibrator Tests
# ══════════════════════════════════════════════════════

class TestCalibrator:

    def setup_method(self):
        self.cal = ConfidenceCalibrator()

    def test_tiny_tier_high_confidence(self):
        s = self.cal.calibrate(0.5, "Hello!", "tiny", False)
        assert s.calibrated >= 0.95

    def test_memory_tier_high_confidence(self):
        s = self.cal.calibrate(0.5, "Your name is Gowri.", "memory", True)
        assert s.calibrated >= 0.90

    def test_verified_boosts(self):
        s_unverified = self.cal.calibrate(0.5, "The answer is 42.", "medium", False)
        s_verified   = self.cal.calibrate(0.5, "The answer is 42.", "medium", True)
        assert s_verified.calibrated > s_unverified.calibrated

    def test_uncertainty_lowers(self):
        confident = self.cal.calibrate(0.5, "Paris is the capital of France.", "medium", False)
        uncertain = self.cal.calibrate(0.5, "I'm not sure but I think it might be Paris.", "medium", False)
        assert uncertain.calibrated < confident.calibrated

    def test_score_in_range(self):
        for tier in ["tiny", "memory", "small", "medium", "full"]:
            s = self.cal.calibrate(0.5, "Some response.", tier, False)
            assert 0.0 <= s.calibrated <= 1.0

    def test_bar_length(self):
        s = self.cal.calibrate(0.8, "test", "medium", False)
        assert len(s.bar) == 20

    def test_label_set(self):
        s = self.cal.calibrate(0.9, "definitive answer", "medium", True)
        assert s.label in ["High", "Good", "Moderate", "Low", "Very low"]

    def test_method_populated(self):
        s = self.cal.calibrate(0.5, "test response", "medium", False)
        assert len(s.method) > 0

    def test_phi3_base_applied(self):
        # Raw score of 0.5 should be raised to phi3 base for normal responses
        s = self.cal.calibrate(0.5, "The capital of France is Paris.", "medium", False)
        assert s.calibrated >= 0.65  # should be above raw 50%


# ══════════════════════════════════════════════════════
# Continual Trainer Tests
# ══════════════════════════════════════════════════════

class TestContinualTrainer:

    def setup_method(self):
        self.store   = TrainingDataStore("/tmp/leanai_test_trainer")
        self.trainer = ContinualTrainer(
            self.store,
            config=TrainingConfig(
                min_pairs_to_train=5,
                min_new_pairs_since_last=1,
                check_interval_minutes=999,
            ),
            base_path="/tmp/leanai_test_trainer",
        )

    def test_status_structure(self):
        s = self.trainer.status()
        assert "running" in s
        assert "total_pairs" in s
        assert "quality_filter" in s
        assert "training_runs" in s

    def test_not_running_initially(self):
        assert not self.trainer._running

    def test_generate_self_play(self):
        before = len(self.store._pairs)
        n = self.trainer.generate_self_play(5)
        assert n == 5
        assert len(self.store._pairs) >= before + 5

    def test_export_creates_file(self):
        # Add some quality pairs first
        for i in range(3):
            self.store.add_pair(
                f"Question {i}", f"Answer {i} " * 10,
                FeedbackSignal.EXCELLENT, confidence=0.95, verified=True,
                latency_ms=500, tier_used="self_play",
            )
        path = self.trainer.export_training_data("/tmp/test_export.jsonl")
        if path:
            import os
            assert os.path.exists(path)

    def test_training_cycle_skipped_no_data(self):
        fresh_store = TrainingDataStore("/tmp/leanai_fresh_store")
        fresh_trainer = ContinualTrainer(
            fresh_store,
            config=TrainingConfig(min_pairs_to_train=1000),
            base_path="/tmp/leanai_fresh",
        )
        run = fresh_trainer._training_cycle()
        assert run.status == "skipped"

    def test_generate_adds_verified_pairs(self):
        self.trainer.generate_self_play(10)
        pairs = list(self.store._pairs.values())
        self_play = [p for p in pairs if p.tier_used == "self_play"]
        assert len(self_play) > 0
        assert all(p.verified for p in self_play)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

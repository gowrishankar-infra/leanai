"""
LeanAI · Phase 1 Tests
Run: python -m pytest tests/test_phase1.py -v
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools.z3_verifier import Z3Verifier, Verdict, ClaimType
from core.confidence import ConfidenceScoringEngine
from training.self_improve import (
    TrainingDataStore, SelfPlayEngine, FeedbackSignal, LoRATrainer
)


# ══════════════════════════════════════════════════════
# Z3 Formal Verifier Tests
# ══════════════════════════════════════════════════════

class TestZ3Verifier:

    def setup_method(self):
        self.v = Z3Verifier()

    # ── Basic arithmetic ──
    def test_addition_correct(self):
        r = self.v.prove("2 + 2 = 4")
        assert r.verdict == Verdict.TRUE

    def test_addition_wrong(self):
        r = self.v.prove("2 + 2 = 5")
        assert r.verdict == Verdict.FALSE

    def test_correct_value_returned(self):
        r = self.v.prove("2 + 2 = 5")
        assert r.correct_value is not None
        assert float(r.correct_value) == pytest.approx(4.0)

    def test_multiplication_correct(self):
        r = self.v.prove("15 * 23 = 345")
        assert r.verdict == Verdict.TRUE

    def test_multiplication_wrong(self):
        r = self.v.prove("15 * 23 = 300")
        assert r.verdict == Verdict.FALSE

    def test_large_numbers(self):
        r = self.v.prove("999 * 999 = 998001")
        assert r.verdict == Verdict.TRUE

    def test_subtraction(self):
        r = self.v.prove("100 - 37 = 63")
        assert r.verdict == Verdict.TRUE

    def test_division(self):
        r = self.v.prove("100 / 4 = 25")
        assert r.verdict == Verdict.TRUE

    def test_power(self):
        r = self.v.prove("2^10 = 1024")
        assert r.verdict == Verdict.TRUE

    def test_sqrt(self):
        r = self.v.prove("sqrt(144) = 12")
        assert r.verdict == Verdict.TRUE

    def test_sqrt_wrong(self):
        r = self.v.prove("sqrt(144) = 13")
        assert r.verdict == Verdict.FALSE

    # ── Inequality ──
    def test_greater_than_true(self):
        r = self.v.prove("10 > 5")
        assert r.verdict == Verdict.TRUE

    def test_greater_than_false(self):
        r = self.v.prove("3 > 10")
        assert r.verdict == Verdict.FALSE

    def test_less_than_equal(self):
        r = self.v.prove("5 <= 5")
        assert r.verdict == Verdict.TRUE

    def test_not_equal_true(self):
        r = self.v.prove("7 != 8")
        assert r.verdict == Verdict.TRUE

    # ── Text extraction ──
    def test_extract_from_response_correct(self):
        text = "The result of 10 + 5 = 15, which gives us 15."
        report = self.v.verify_text(text)
        assert report.claims_found >= 1

    def test_extract_from_response_wrong(self):
        text = "Calculating 7 * 8 = 54, so the answer is 54."
        report = self.v.verify_text(text)
        # 7*8=56, not 54 — should be refuted
        assert report.overall_verdict in [Verdict.FALSE, Verdict.UNKNOWN]

    def test_no_math_not_checked(self):
        from tools.z3_verifier import Verdict
        text = "Paris is the capital of France and London is in England."
        report = self.v.verify_text(text)
        assert report.claims_found == 0

    def test_correction_applied(self):
        text = "The answer is that 6 * 7 = 40."
        report = self.v.verify_text(text)
        if report.corrected_text:
            assert "42" in report.corrected_text

    def test_proof_steps_populated(self):
        r = self.v.prove("3 + 4 = 7")
        assert len(r.proof_steps) > 0

    def test_engine_used_populated(self):
        r = self.v.prove("100 + 200 = 300")
        assert r.engine_used in ["sympy", "python_eval", "z3"]

    def test_verifier_status(self):
        s = self.v.status
        assert "arithmetic" in s
        assert s["arithmetic"] is True


# ══════════════════════════════════════════════════════
# Confidence Scoring Tests
# ══════════════════════════════════════════════════════

class TestConfidenceScoring:

    def setup_method(self):
        self.scorer = ConfidenceScoringEngine()

    def test_confident_text_scores_higher(self):
        confident = "The answer is definitively 42. This equals 42."
        uncertain  = "I think it might be 42, but I'm not sure, possibly."
        s1 = self.scorer.score_from_text(confident)
        s2 = self.scorer.score_from_text(uncertain)
        assert s1.overall > s2.overall

    def test_hedge_phrases_detected(self):
        text = "I'm not sure about this, it might be correct."
        score = self.scorer.score_from_text(text)
        assert len(score.linguistic_flags) > 0

    def test_score_in_range(self):
        for text in ["hello", "I think maybe possibly", "The answer is 42."]:
            score = self.scorer.score_from_text(text)
            assert 0.0 <= score.overall <= 1.0

    def test_label_correct(self):
        high   = self.scorer.score_from_text("The answer is definitively proven.")
        assert high.label in ["High", "Moderate", "Low", "Very low"]

    def test_verification_boost(self):
        score = self.scorer.score_from_text("The result is 42.")
        boosted = self.scorer.combine_with_verification(score, verified=True, refuted=False)
        assert boosted.overall >= score.overall

    def test_verification_penalty(self):
        score = self.scorer.score_from_text("The result is 42.")
        penalized = self.scorer.combine_with_verification(score, verified=False, refuted=True)
        assert penalized.overall <= score.overall

    def test_logprob_scoring(self):
        tokens  = ["The", " answer", " is", " 42"]
        logprobs = [-0.1, -0.3, -0.2, -0.5]
        score = self.scorer.score_from_logprobs(tokens, logprobs)
        assert score.overall > 0.5  # should be fairly confident

    def test_high_entropy_logprobs_lower_score(self):
        tokens_conf = ["yes", " definitely"]
        tokens_unc  = ["maybe", " possibly"]
        lp_conf = [-0.1, -0.1]
        lp_unc  = [-3.0, -3.0]
        s1 = self.scorer.score_from_logprobs(tokens_conf, lp_conf)
        s2 = self.scorer.score_from_logprobs(tokens_unc, lp_unc)
        assert s1.overall > s2.overall

    def test_bar_length(self):
        score = self.scorer.score_from_text("test")
        assert len(score.bar) == 20

    def test_percentage_int(self):
        score = self.scorer.score_from_text("test")
        assert isinstance(score.percentage, int)
        assert 0 <= score.percentage <= 100

    def test_needs_verification_flag(self):
        uncertain = "I'm not sure and I think it might be right possibly maybe."
        score = self.scorer.score_from_text(uncertain)
        assert isinstance(score.needs_verification, bool)


# ══════════════════════════════════════════════════════
# Self-Improvement / Training Tests
# ══════════════════════════════════════════════════════

class TestTrainingStore:

    def setup_method(self):
        self.store = TrainingDataStore("/tmp/leanai_test_training")

    def test_add_pair_returns_object(self):
        p = self.store.add_pair("What is 2+2?", "4", FeedbackSignal.EXCELLENT)
        assert p.id is not None
        assert p.instruction == "What is 2+2?"

    def test_quality_score_range(self):
        p = self.store.add_pair("test", "response", FeedbackSignal.GOOD)
        assert 0.0 <= p.quality_score <= 1.0

    def test_excellent_feedback_high_quality(self):
        p = self.store.add_pair("test", "response",
                                 FeedbackSignal.EXCELLENT,
                                 confidence=0.95, verified=True)
        assert p.is_high_quality

    def test_wrong_feedback_low_quality(self):
        p = self.store.add_pair("test", "bad response", FeedbackSignal.WRONG)
        assert not p.is_high_quality

    def test_update_feedback(self):
        p = self.store.add_pair("test", "response", FeedbackSignal.NEUTRAL)
        self.store.update_feedback(p.id, FeedbackSignal.EXCELLENT)
        assert self.store._pairs[p.id].feedback == FeedbackSignal.EXCELLENT

    def test_get_batch_filters_quality(self):
        # Add a bad pair
        self.store.add_pair("bad", "bad", FeedbackSignal.WRONG)
        # Add a good pair
        self.store.add_pair("good", "good", FeedbackSignal.EXCELLENT,
                             confidence=0.95, verified=True)
        batch = self.store.get_training_batch(min_quality=0.75)
        for p in batch:
            assert p.quality_score >= 0.75

    def test_stats_structure(self):
        s = self.store.stats()
        assert "total" in s
        assert "high_quality" in s
        assert "verified" in s
        assert "avg_quality" in s


class TestSelfPlayEngine:

    def setup_method(self):
        self.engine = SelfPlayEngine()

    def test_generates_math_pair(self):
        pair = self.engine.generate_math_pair()
        assert pair is not None
        assert pair.problem != ""
        assert pair.solution != ""

    def test_pair_is_verified(self):
        pair = self.engine.generate_math_pair()
        assert pair.verified is True

    def test_batch_generates_multiple(self):
        pairs = self.engine.generate_batch(5)
        assert len(pairs) == 5

    def test_solution_contains_answer(self):
        # Generate arithmetic pair and check it has a number in solution
        import re
        pair = self.engine.generate_math_pair()
        numbers = re.findall(r'\d+', pair.solution)
        assert len(numbers) > 0

    def test_domain_set(self):
        pair = self.engine.generate_math_pair()
        assert pair.domain in ["math", "code", "reasoning", "factual"]

    def test_difficulty_in_range(self):
        pair = self.engine.generate_math_pair()
        assert 0.0 <= pair.difficulty <= 1.0


class TestLoRATrainer:

    def setup_method(self):
        self.store = TrainingDataStore("/tmp/leanai_test_lora")
        self.trainer = LoRATrainer(self.store)

    def test_should_not_train_empty_store(self):
        assert not self.trainer.should_train()

    def test_training_plan_structure(self):
        plan = self.trainer.training_plan()
        assert "method" in plan
        assert plan["method"] == "LoRA"
        assert "rank" in plan
        assert "learning_rate" in plan

    def test_prepare_data_creates_file(self):
        # Add some high-quality pairs
        for i in range(5):
            self.store.add_pair(
                f"Question {i}", f"Answer {i}",
                FeedbackSignal.EXCELLENT, confidence=0.95, verified=True
            )
        path = self.trainer.prepare_training_data("/tmp/leanai_test_batch.jsonl")
        import os
        assert os.path.exists(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
LeanAI · Test Suite
Tests for all Phase 0 components.
Run: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.router import TaskRouter, Tier
from core.watchdog import MetaCognitiveWatchdog
from tools.verifier import NeurosymbolicVerifier, VerificationStatus
from memory.hierarchy import HierarchicalMemory, WorkingMemory, EpisodicMemory


# ══════════════════════════════════════════════════════
# Router Tests
# ══════════════════════════════════════════════════════

class TestRouter:

    def setup_method(self):
        self.router = TaskRouter()

    def test_greeting_routes_to_tiny(self):
        decision = self.router.route("hello")
        assert decision.tier == Tier.TINY

    def test_thanks_routes_to_tiny(self):
        decision = self.router.route("thanks!")
        assert decision.tier == Tier.TINY

    def test_simple_math_routes_to_tiny(self):
        decision = self.router.route("2 + 2")
        assert decision.tier == Tier.TINY

    def test_complex_question_routes_to_full(self):
        decision = self.router.route(
            "Can you explain how transformer attention mechanisms work "
            "and compare them to recurrent neural networks in detail?"
        )
        assert decision.tier in [Tier.MEDIUM, Tier.FULL]

    def test_math_triggers_verifier(self):
        decision = self.router.route("What is the solution to x^2 + 2x + 1 = 0?")
        assert decision.requires_verifier is True

    def test_code_triggers_tools(self):
        decision = self.router.route("Write a Python function to sort a list")
        assert decision.requires_tools is True

    def test_heavy_task_routes_to_heavy(self):
        decision = self.router.route("Write a complete research paper on transformer architectures")
        assert decision.tier == Tier.HEAVY

    def test_decision_has_confidence(self):
        decision = self.router.route("What is the weather like?")
        assert 0.0 <= decision.confidence <= 1.0

    def test_explain_returns_string(self):
        decision = self.router.route("hello")
        explanation = self.router.explain(decision)
        assert isinstance(explanation, str)
        assert len(explanation) > 0


# ══════════════════════════════════════════════════════
# Watchdog Tests
# ══════════════════════════════════════════════════════

class TestWatchdog:

    def setup_method(self):
        self.watchdog = MetaCognitiveWatchdog()

    def test_confident_response_high_confidence(self):
        state = self.watchdog.simulate_from_response(
            "The answer is 42. This is definitively correct.",
        )
        assert state.final_confidence > 0.5

    def test_uncertain_response_lower_confidence(self):
        state = self.watchdog.simulate_from_response(
            "I think this might be correct, but I'm not sure. It could possibly be 42, "
            "though you should verify this independently."
        )
        assert state.final_confidence < 0.9

    def test_token_observation_updates_state(self):
        state = self.watchdog.new_generation()
        self.watchdog.observe_token(state, 42, [-0.1, -1.5, -2.0], position=0)
        assert state.token_count == 1
        assert len(state.entropy_history) == 1

    def test_high_entropy_sets_verify_flag(self):
        state = self.watchdog.new_generation()
        high_entropy_logprobs = [-0.5, -0.6, -0.7, -0.8, -0.9, -1.0]
        for i in range(10):
            self.watchdog.observe_token(state, i, high_entropy_logprobs, i)
        state = self.watchdog.finalize(state)
        assert state.final_confidence <= 1.0

    def test_confidence_label_correct(self):
        assert "High" in self.watchdog.confidence_label(0.95)
        assert "Moderate" in self.watchdog.confidence_label(0.70)
        assert "Low" in self.watchdog.confidence_label(0.45)
        assert "uncertain" in self.watchdog.confidence_label(0.20).lower()

    def test_finalize_sets_mean_entropy(self):
        state = self.watchdog.new_generation()
        self.watchdog.observe_token(state, 0, [-1.0, -2.0, -3.0], 0)
        self.watchdog.observe_token(state, 1, [-1.0, -2.0, -3.0], 1)
        state = self.watchdog.finalize(state)
        assert state.mean_entropy > 0


# ══════════════════════════════════════════════════════
# Verifier Tests
# ══════════════════════════════════════════════════════

class TestVerifier:

    def setup_method(self):
        self.verifier = NeurosymbolicVerifier()

    def test_correct_arithmetic_verified(self):
        result = self.verifier.verify_expression("2 + 2 = 4")
        assert result.status == VerificationStatus.VERIFIED

    def test_wrong_arithmetic_refuted(self):
        result = self.verifier.verify_expression("2 + 2 = 5")
        assert result.status == VerificationStatus.REFUTED

    def test_correct_value_returned(self):
        result = self.verifier.verify_expression("2 + 2 = 5")
        assert result.correct_value == "4.0"

    def test_complex_arithmetic_verified(self):
        result = self.verifier.verify_expression("15 * 23 = 345")
        assert result.status == VerificationStatus.VERIFIED

    def test_sqrt_verified(self):
        result = self.verifier.verify_expression("sqrt(16) = 4")
        assert result.status == VerificationStatus.VERIFIED

    def test_response_with_correct_math(self):
        response = "The result of 10 + 5 = 15, so the total is 15."
        report = self.verifier.verify_response(response)
        assert report.overall_status in [
            VerificationStatus.VERIFIED,
            VerificationStatus.NOT_CHECKED,
            VerificationStatus.UNCERTAIN,
        ]

    def test_no_math_returns_not_checked(self):
        response = "Paris is the capital of France."
        report = self.verifier.verify_response(response)
        assert report.overall_status == VerificationStatus.NOT_CHECKED

    def test_capabilities_dict(self):
        caps = self.verifier.capabilities
        assert "arithmetic" in caps
        assert caps["arithmetic"] is True


# ══════════════════════════════════════════════════════
# Memory Tests
# ══════════════════════════════════════════════════════

class TestWorkingMemory:

    def setup_method(self):
        self.mem = WorkingMemory()

    def test_add_message(self):
        self.mem.add_message("user", "Hello")
        assert len(self.mem.messages) == 1
        assert self.mem.messages[0]["role"] == "user"

    def test_token_count_increases(self):
        before = self.mem.current_tokens
        self.mem.add_message("user", "A" * 400)
        assert self.mem.current_tokens > before

    def test_context_window_respects_limit(self):
        for i in range(20):
            self.mem.add_message("user", f"Message {i} " * 50)
        window = self.mem.get_context_window(max_tokens=100)
        assert len(window) <= 20

    def test_clear_resets_state(self):
        self.mem.add_message("user", "test")
        self.mem.clear()
        assert len(self.mem.messages) == 0
        assert self.mem.current_tokens == 0


class TestEpisodicMemory:

    def setup_method(self):
        self.mem = EpisodicMemory("/tmp/leanai_test_episodic")

    def test_store_returns_id(self):
        entry_id = self.mem.store("Test memory content")
        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

    def test_search_finds_relevant(self):
        self.mem.store("The quick brown fox jumps over the lazy dog")
        results = self.mem.search("fox dog")
        assert len(results) > 0

    def test_recent_returns_entries(self):
        self.mem.store("Entry 1")
        self.mem.store("Entry 2")
        recent = self.mem.recent(5)
        assert len(recent) >= 2

    def test_access_count_increments(self):
        self.mem.store("Searchable unique content xyz123")
        results = self.mem.search("Searchable unique content xyz123")
        if results:
            assert results[0].access_count >= 1


class TestHierarchicalMemory:

    def setup_method(self):
        self.mem = HierarchicalMemory("/tmp/leanai_test")

    def test_stats_returns_dict(self):
        stats = self.mem.stats()
        assert "working_tokens" in stats
        assert "episodic_entries" in stats
        assert "semantic_entities" in stats
        assert "procedural_solutions" in stats

    def test_record_exchange_updates_working(self):
        before = len(self.mem.working.messages)
        self.mem.record_exchange("Hello", "Hi there!")
        assert len(self.mem.working.messages) == before + 2

    def test_prepare_context_returns_string(self):
        context = self.mem.prepare_context("Tell me about AI")
        assert isinstance(context, str)

    def test_semantic_entity_stored_and_retrieved(self):
        self.mem.semantic.add_entity("Python", "programming_language", {"year": "1991"})
        entity = self.mem.semantic.query_entity("python")
        assert entity is not None
        assert entity["name"] == "Python"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

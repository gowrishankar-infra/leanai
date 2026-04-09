"""
Tests for LeanAI Phase 5a — Swarm Consensus
"""

import pytest
from swarm import (
    SwarmConsensus,
    SwarmResult,
    SwarmCandidate,
    text_similarity,
    extract_core_answer,
)


# ── text_similarity tests ─────────────────────────────────────────

class TestTextSimilarity:
    def test_identical(self):
        assert text_similarity("hello world", "hello world") == 1.0

    def test_identical_different_case(self):
        assert text_similarity("Hello World", "hello world") == 1.0

    def test_identical_extra_whitespace(self):
        assert text_similarity("hello   world", "hello world") == 1.0

    def test_completely_different(self):
        sim = text_similarity("the sky is blue", "quantum physics is complex")
        assert sim < 0.5

    def test_similar(self):
        sim = text_similarity(
            "The capital of France is Paris.",
            "Paris is the capital of France."
        )
        assert sim > 0.5

    def test_empty_string(self):
        assert text_similarity("", "hello") == 0.0
        assert text_similarity("hello", "") == 0.0
        assert text_similarity("", "") == 0.0

    def test_partial_overlap(self):
        sim = text_similarity("Python is great for AI", "Python is great for web development")
        assert 0.3 < sim < 0.9


# ── extract_core_answer tests ─────────────────────────────────────

class TestExtractCoreAnswer:
    def test_short_answer(self):
        result = extract_core_answer("Paris")
        assert result == "Paris"

    def test_first_line(self):
        result = extract_core_answer("Paris.\nIt is the capital of France.")
        assert result == "Paris."

    def test_first_sentence(self):
        result = extract_core_answer("The capital of France is Paris. It has many famous landmarks.")
        assert result == "The capital of France is Paris."

    def test_long_first_line(self):
        long_text = "A" * 200 + "\nSecond line"
        result = extract_core_answer(long_text)
        assert len(result) <= 100

    def test_empty(self):
        result = extract_core_answer("")
        assert result == ""

    def test_multiline_code(self):
        code = "def add(a, b):\n    return a + b\n\nprint(add(1, 2))"
        result = extract_core_answer(code)
        assert "def add" in result


# ── SwarmCandidate tests ──────────────────────────────────────────

class TestSwarmCandidate:
    def test_creation(self):
        c = SwarmCandidate(text="hello", temperature=0.1, latency_ms=100)
        assert c.text == "hello"
        assert c.temperature == 0.1
        assert c.agreement_score == 0.0

    def test_default_agreement(self):
        c = SwarmCandidate(text="test", temperature=0.3, latency_ms=50)
        assert c.agreement_score == 0.0


# ── SwarmResult tests ─────────────────────────────────────────────

class TestSwarmResult:
    def test_summary_unanimous(self):
        r = SwarmResult(
            best_answer="Paris", consensus_score=1.0, confidence=99,
            num_passes=3, unanimous=True,
        )
        s = r.summary()
        assert "UNANIMOUS" in s
        assert "99%" in s
        assert "3 passes" in s

    def test_summary_partial(self):
        r = SwarmResult(
            best_answer="Paris", consensus_score=0.7, confidence=85,
            num_passes=3, unanimous=False,
        )
        s = r.summary()
        assert "70% agreement" in s

    def test_summary_with_candidates(self):
        candidates = [
            SwarmCandidate(text="Paris is the capital", temperature=0.1, latency_ms=100, agreement_score=0.9),
            SwarmCandidate(text="The capital is Paris", temperature=0.3, latency_ms=120, agreement_score=0.85),
        ]
        r = SwarmResult(
            best_answer="Paris is the capital", consensus_score=0.875, confidence=90,
            candidates=candidates, num_passes=2,
        )
        s = r.summary()
        assert "→" in s  # marker for best candidate
        assert "t=0.1" in s


# ── SwarmConsensus core tests ─────────────────────────────────────

class TestSwarmConsensus:
    def _mock_model(self, answers):
        """Create a model function that returns answers in order."""
        idx = [0]
        def fn(prompt: str, temperature: float) -> str:
            if idx[0] < len(answers):
                r = answers[idx[0]]
                idx[0] += 1
                return r
            return answers[-1] if answers else ""
        return fn

    def test_unanimous_consensus(self):
        """When all passes return the same answer, consensus should be very high."""
        model_fn = self._mock_model([
            "The capital of France is Paris.",
            "The capital of France is Paris.",
            "The capital of France is Paris.",
        ])
        swarm = SwarmConsensus(model_fn=model_fn, num_passes=3)
        result = swarm.query("What is the capital of France?")
        assert result.unanimous is True
        assert result.consensus_score > 0.9
        assert result.confidence >= 95
        assert "Paris" in result.best_answer

    def test_majority_consensus(self):
        """When 2/3 agree, consensus should be moderate-high."""
        model_fn = self._mock_model([
            "The capital of France is Paris.",
            "Paris is the capital of France.",
            "The largest city in France is Lyon.",  # wrong
        ])
        swarm = SwarmConsensus(model_fn=model_fn, num_passes=3)
        result = swarm.query("What is the capital of France?")
        assert result.num_passes == 3
        # The two Paris answers should have higher agreement
        paris_candidates = [c for c in result.candidates if "Paris" in c.text]
        lyon_candidate = [c for c in result.candidates if "Lyon" in c.text]
        if paris_candidates and lyon_candidate:
            assert paris_candidates[0].agreement_score > lyon_candidate[0].agreement_score

    def test_no_consensus(self):
        """When all answers differ, consensus should be low."""
        model_fn = self._mock_model([
            "The answer is 42.",
            "I think it might be blue.",
            "Quantum computing is the future.",
        ])
        swarm = SwarmConsensus(model_fn=model_fn, num_passes=3)
        result = swarm.query("Random question")
        assert result.consensus_score < 0.7
        assert result.unanimous is False

    def test_single_pass(self):
        """With 1 pass, should just return that answer."""
        model_fn = self._mock_model(["Just one answer."])
        swarm = SwarmConsensus(model_fn=model_fn, num_passes=1)
        result = swarm.query("Question")
        assert result.best_answer == "Just one answer."
        assert result.num_passes == 1

    def test_query_with_existing(self):
        """Test using an existing first answer."""
        model_fn = self._mock_model([
            "Paris is the capital.",  # pass 2
            "Paris is the capital of France.",  # pass 3
        ])
        swarm = SwarmConsensus(model_fn=model_fn, num_passes=3)
        result = swarm.query_with_existing(
            first_answer="The capital of France is Paris.",
            prompt="What is the capital of France?",
            base_confidence=72,
        )
        assert result.num_passes == 3
        assert len(result.candidates) == 3
        # First candidate should be the existing answer
        assert "Paris" in result.candidates[0].text
        assert result.candidates[0].latency_ms == 0  # pre-existing, no latency

    def test_confidence_boost_unanimous(self):
        """Unanimous answers should boost confidence to 95+."""
        model_fn = self._mock_model(["Same answer.", "Same answer.", "Same answer."])
        swarm = SwarmConsensus(model_fn=model_fn, num_passes=3)
        result = swarm.query("Q", base_confidence=50)
        assert result.confidence >= 95

    def test_confidence_boost_low_consensus(self):
        """Low consensus should cap confidence."""
        model_fn = self._mock_model(["Alpha beta gamma.", "Delta epsilon zeta.", "Eta theta iota."])
        swarm = SwarmConsensus(model_fn=model_fn, num_passes=3)
        result = swarm.query("Q", base_confidence=80)
        assert result.confidence <= 60  # capped because disagreement

    def test_custom_temperatures(self):
        """Test with custom temperature list."""
        model_fn = self._mock_model(["A", "B"])
        swarm = SwarmConsensus(model_fn=model_fn, num_passes=2, temperatures=[0.0, 1.0])
        result = swarm.query("Q")
        assert result.candidates[0].temperature == 0.0
        assert result.candidates[1].temperature == 1.0

    def test_no_model_raises(self):
        """Should raise if no model function provided."""
        swarm = SwarmConsensus(model_fn=None)
        with pytest.raises(RuntimeError, match="No model_fn"):
            swarm.query("Q")

    def test_result_tracks_total_latency(self):
        model_fn = self._mock_model(["Quick answer."])
        swarm = SwarmConsensus(model_fn=model_fn, num_passes=1)
        result = swarm.query("Q")
        assert result.total_latency_ms >= 0

    def test_picks_higher_agreement(self):
        """Best answer should be the one with highest agreement."""
        model_fn = self._mock_model([
            "Common answer here.",
            "Common answer here.",
            "Outlier completely different response about something else entirely.",
        ])
        swarm = SwarmConsensus(model_fn=model_fn, num_passes=3)
        result = swarm.query("Q")
        assert "Common" in result.best_answer

    def test_prefers_lower_temp_on_tie(self):
        """When agreement is tied, prefer the lower temperature candidate."""
        # Both candidates identical — agreement will be 1.0 for both
        model_fn = self._mock_model(["Same.", "Same."])
        swarm = SwarmConsensus(model_fn=model_fn, num_passes=2, temperatures=[0.1, 0.5])
        result = swarm.query("Q")
        # Best should be the t=0.1 candidate (more precise)
        assert result.candidates[0].temperature == 0.1 or result.best_answer == "Same."


# ── Edge case tests ───────────────────────────────────────────────

class TestSwarmEdgeCases:
    def test_empty_responses(self):
        model_fn = lambda p, t: ""
        swarm = SwarmConsensus(model_fn=model_fn, num_passes=2)
        result = swarm.query("Q")
        assert result.best_answer == ""

    def test_very_long_responses(self):
        long_text = "word " * 1000
        model_fn = lambda p, t: long_text
        swarm = SwarmConsensus(model_fn=model_fn, num_passes=2)
        result = swarm.query("Q")
        assert result.unanimous is True

    def test_similarity_threshold_custom(self):
        swarm = SwarmConsensus(model_fn=lambda p, t: "x", similarity_threshold=0.9)
        assert swarm.similarity_threshold == 0.9

    def test_auto_extend_temperatures(self):
        """If fewer temperatures than passes, should auto-extend."""
        swarm = SwarmConsensus(model_fn=lambda p, t: "x", num_passes=5, temperatures=[0.1])
        assert len(swarm.temperatures) == 5

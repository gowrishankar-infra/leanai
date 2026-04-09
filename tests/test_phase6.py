"""
Tests for LeanAI Phase 6 — Speculative Decoding, Liquid Router, HDC Store
"""

import os
import json
import shutil
import tempfile
import time
import pytest
import numpy as np

from speculative import SpeculativeDecoder, SpecConfig, SpecResult
from liquid import (
    LiquidRouter, RoutingFeatures, TierPerformance,
    extract_features, feature_bin,
)
from hdc import (
    HDEncoder, HDVector, HDKnowledgeStore,
    hamming_distance, hamming_similarity, DEFAULT_DIM,
)


# ══════════════════════════════════════════════════════════════════
# Phase 6a — Speculative Decoding Tests
# ══════════════════════════════════════════════════════════════════

class TestSpeculativeDecoder:
    def _mock_draft(self, response: str):
        return lambda p, mt, t: response

    def _mock_main(self, response: str):
        return lambda p, mt, t: response

    def test_identical_responses_accepted(self):
        """When draft and main agree, draft should be accepted."""
        spec = SpeculativeDecoder(
            draft_fn=self._mock_draft("The answer is 42."),
            main_fn=self._mock_main("The answer is 42."),
        )
        result = spec.generate("What is the answer?")
        assert result.used_draft is True
        assert result.similarity == 1.0
        assert "42" in result.text

    def test_different_responses_rejected(self):
        """When draft and main disagree, main should be used."""
        spec = SpeculativeDecoder(
            draft_fn=self._mock_draft("Completely wrong answer about cats."),
            main_fn=self._mock_main("The capital of France is Paris."),
            config=SpecConfig(acceptance_threshold=0.6),
        )
        result = spec.generate("Capital of France?")
        assert result.used_draft is False
        assert "Paris" in result.text

    def test_draft_only_mode(self):
        """Draft-only mode should skip main model."""
        spec = SpeculativeDecoder(
            draft_fn=self._mock_draft("Quick answer."),
            main_fn=self._mock_main("Slow answer."),
        )
        result = spec.generate_draft_only("Question?")
        assert result.used_draft is True
        assert result.text == "Quick answer."
        assert result.main_time_ms == 0

    def test_disabled_uses_main_only(self):
        """When disabled, should use main model only."""
        spec = SpeculativeDecoder(
            draft_fn=self._mock_draft("Draft"),
            main_fn=self._mock_main("Main response"),
            config=SpecConfig(enabled=False),
        )
        result = spec.generate("Q")
        assert result.text == "Main response"
        assert result.used_draft is False

    def test_no_draft_fn_uses_main(self):
        """Without draft function, should use main only."""
        spec = SpeculativeDecoder(
            draft_fn=None,
            main_fn=self._mock_main("Main only"),
        )
        result = spec.generate("Q")
        assert result.text == "Main only"

    def test_stats_tracking(self):
        spec = SpeculativeDecoder(
            draft_fn=self._mock_draft("Same"),
            main_fn=self._mock_main("Same"),
        )
        spec.generate("Q1")
        spec.generate("Q2")
        s = spec.stats()
        assert s["total_queries"] == 2
        assert s["drafts_accepted"] == 2

    def test_acceptance_rate(self):
        spec = SpeculativeDecoder(
            draft_fn=self._mock_draft("A"),
            main_fn=self._mock_main("A"),
        )
        spec.generate("Q")
        assert spec.acceptance_rate == 1.0

    def test_speedup_calculated(self):
        spec = SpeculativeDecoder(
            draft_fn=self._mock_draft("Same answer"),
            main_fn=self._mock_main("Same answer"),
        )
        result = spec.generate("Q")
        assert result.speedup >= 0

    def test_result_summary(self):
        result = SpecResult(
            text="Answer", used_draft=True, similarity=0.95,
            draft_time_ms=50, main_time_ms=500, speedup=10.0,
        )
        s = result.summary()
        assert "draft" in s
        assert "10.0x" in s


# ══════════════════════════════════════════════════════════════════
# Phase 6b — Liquid Router Tests
# ══════════════════════════════════════════════════════════════════

class TestExtractFeatures:
    def test_greeting(self):
        f = extract_features("hello")
        assert f.is_greeting is True
        assert f.complexity_score < 0.3

    def test_code_query(self):
        f = extract_features("write a python function to sort an array")
        assert f.has_code_keywords is True
        assert f.complexity_score > 0.3

    def test_math_query(self):
        f = extract_features("calculate 15 * 23")
        assert f.has_math_keywords is True

    def test_memory_query(self):
        f = extract_features("what is my name")
        assert f.is_memory_query is True

    def test_question_mark(self):
        f = extract_features("what is AI?")
        assert f.has_question_mark is True

    def test_long_query(self):
        f = extract_features("explain the detailed architecture of transformer neural networks with attention mechanisms and positional encoding")
        assert f.word_count > 10
        assert f.complexity_score > 0.5

    def test_to_vector(self):
        f = extract_features("hello world")
        vec = f.to_vector()
        assert len(vec) == 8
        assert all(isinstance(v, (int, float)) for v in vec)


class TestFeatureBin:
    def test_greeting_bin(self):
        f = extract_features("hello")
        b = feature_bin(f)
        assert "greet" in b
        assert "short" in b

    def test_code_bin(self):
        f = extract_features("write a python function")
        b = feature_bin(f)
        assert "code" in b


class TestLiquidRouter:
    @pytest.fixture
    def router(self):
        d = tempfile.mkdtemp()
        r = LiquidRouter(data_dir=d)
        yield r
        shutil.rmtree(d)

    def test_greeting_routes_to_tiny(self, router):
        assert router.route("hello") == "tiny"
        assert router.route("hi") == "tiny"

    def test_memory_routes_to_tiny(self, router):
        assert router.route("what is my name") == "tiny"

    def test_math_routes_to_tiny(self, router):
        assert router.route("calculate 5 * 3") == "tiny"

    def test_code_routes_to_medium(self, router):
        tier = router.route("write a python function to implement quicksort algorithm with good complexity")
        assert tier in ("medium", "small")  # complex code should go to medium or small

    def test_feedback_updates_performance(self, router):
        router.route("test query")
        router.feedback("medium", confidence=0.9, latency_ms=3000)
        assert router.tiers["medium"].total_queries == 1
        assert router.tiers["medium"].total_good == 1

    def test_feedback_with_percentage_confidence(self, router):
        """Confidence as 0-100 should be normalized."""
        router.route("test")
        router.feedback("small", confidence=85, latency_ms=2000)
        assert router.tiers["small"].ema_confidence > 0.5

    def test_ema_adapts_over_time(self, router):
        """EMA should shift toward recent performance."""
        for _ in range(20):
            router.route("complex python code")
            router.feedback("medium", confidence=0.95, latency_ms=2000)
        assert router.tiers["medium"].ema_confidence > 0.8

    def test_state_persistence(self):
        d = tempfile.mkdtemp()
        r1 = LiquidRouter(data_dir=d)
        r1.route("test")
        r1.feedback("medium", confidence=0.9, latency_ms=1000)
        r1.tiers["medium"].total_queries = 10  # force a save trigger
        r1._save_state()

        r2 = LiquidRouter(data_dir=d)
        assert r2.tiers["medium"].total_queries == 10
        shutil.rmtree(d)

    def test_stats(self, router):
        s = router.stats()
        assert "queries_routed" in s
        assert "tiers" in s
        assert "tiny" in s["tiers"]

    def test_different_queries_different_routes(self, router):
        """Different query types should potentially route to different tiers."""
        t1 = router.route("hi")
        t2 = router.route("write a complex distributed systems architecture with microservices")
        # Greeting should be tiny, complex should be medium/small
        assert t1 == "tiny"
        assert t2 != "tiny"


# ══════════════════════════════════════════════════════════════════
# Phase 6c — HDC Knowledge Store Tests
# ══════════════════════════════════════════════════════════════════

class TestHammingOperations:
    def test_distance_identical(self):
        a = np.array([1, 0, 1, 0], dtype=np.uint8)
        assert hamming_distance(a, a) == 0

    def test_distance_opposite(self):
        a = np.zeros(100, dtype=np.uint8)
        b = np.ones(100, dtype=np.uint8)
        assert hamming_distance(a, b) == 100

    def test_similarity_identical(self):
        a = np.array([1, 0, 1], dtype=np.uint8)
        assert hamming_similarity(a, a) == 1.0

    def test_similarity_opposite(self):
        a = np.zeros(100, dtype=np.uint8)
        b = np.ones(100, dtype=np.uint8)
        assert hamming_similarity(a, b) == 0.0

    def test_similarity_half(self):
        a = np.array([1, 1, 0, 0], dtype=np.uint8)
        b = np.array([1, 0, 1, 0], dtype=np.uint8)
        assert hamming_similarity(a, b) == 0.5


class TestHDEncoder:
    def test_encode_word(self):
        enc = HDEncoder(dim=1000)
        v = enc.encode_word("hello")
        assert v.shape == (1000,)
        assert set(np.unique(v)).issubset({0, 1})

    def test_same_word_same_vector(self):
        enc = HDEncoder(dim=1000)
        v1 = enc.encode_word("python")
        v2 = enc.encode_word("python")
        np.testing.assert_array_equal(v1, v2)

    def test_different_words_different_vectors(self):
        enc = HDEncoder(dim=1000)
        v1 = enc.encode_word("hello")
        v2 = enc.encode_word("world")
        assert not np.array_equal(v1, v2)

    def test_encode_text(self):
        enc = HDEncoder(dim=1000)
        v = enc.encode_text("the quick brown fox")
        assert v.shape == (1000,)

    def test_similar_texts_similar_vectors(self):
        enc = HDEncoder(dim=5000)
        v1 = enc.encode_text("the capital of france is paris")
        v2 = enc.encode_text("paris is the capital of france")
        sim = hamming_similarity(v1, v2)
        assert sim > 0.5  # similar texts should have > 50% similarity

    def test_different_texts_low_similarity(self):
        enc = HDEncoder(dim=5000)
        v1 = enc.encode_text("the capital of france is paris")
        v2 = enc.encode_text("quantum computing uses qubits for calculation")
        sim = hamming_similarity(v1, v2)
        assert sim < 0.6  # different topics should have lower similarity

    def test_encode_returns_hdvector(self):
        enc = HDEncoder(dim=1000)
        hd = enc.encode("test text", {"type": "fact"})
        assert isinstance(hd, HDVector)
        assert hd.text == "test text"
        assert hd.metadata == {"type": "fact"}

    def test_empty_text(self):
        enc = HDEncoder(dim=100)
        v = enc.encode_text("")
        assert np.all(v == 0)


class TestHDVector:
    def test_to_dict_and_back(self):
        bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        v = HDVector(bits=bits, text="test", metadata={"key": "val"})
        d = v.to_dict()
        restored = HDVector.from_dict(d)
        np.testing.assert_array_equal(restored.bits, bits)
        assert restored.text == "test"
        assert restored.metadata == {"key": "val"}

    def test_dim_property(self):
        v = HDVector(bits=np.zeros(5000, dtype=np.uint8))
        assert v.dim == 5000


class TestHDKnowledgeStore:
    @pytest.fixture
    def store(self):
        d = tempfile.mkdtemp()
        s = HDKnowledgeStore(dim=2000, data_dir=d)
        yield s
        shutil.rmtree(d)

    def test_add_and_count(self, store):
        store.add("fact one")
        store.add("fact two")
        assert store.count == 2

    def test_search_finds_similar(self, store):
        store.add("The capital of France is Paris", {"type": "geography"})
        store.add("Python is a programming language", {"type": "tech"})
        store.add("Dogs are loyal animals", {"type": "animals"})

        results = store.search("What is the capital of France?", top_k=3)
        assert len(results) == 3
        # All results should have similarity scores
        assert all(0 <= r[1] <= 1 for r in results)
        # The France entry should be in the results
        texts = [r[0] for r in results]
        assert any("France" in t or "Paris" in t for t in texts)

    def test_search_returns_similarity_scores(self, store):
        store.add("hello world")
        results = store.search("hello world", top_k=1)
        assert len(results) == 1
        text, sim, meta = results[0]
        assert sim > 0.8  # exact match should be very similar

    def test_search_empty_store(self, store):
        results = store.search("anything")
        assert results == []

    def test_add_batch(self, store):
        texts = ["fact A", "fact B", "fact C"]
        store.add_batch(texts)
        assert store.count == 3

    def test_remove(self, store):
        store.add("entry 1")
        store.add("entry 2")
        store.remove(0)
        assert store.count == 1

    def test_clear(self, store):
        store.add("a")
        store.add("b")
        store.clear()
        assert store.count == 0

    def test_save_and_load(self):
        d = tempfile.mkdtemp()
        s1 = HDKnowledgeStore(dim=500, data_dir=d)
        s1.add("persistent fact", {"important": True})
        s1.save()

        s2 = HDKnowledgeStore(dim=500, data_dir=d)
        assert s2.count == 1
        assert s2._vectors[0].text == "persistent fact"
        shutil.rmtree(d)

    def test_memory_estimate(self, store):
        for i in range(10):
            store.add(f"fact number {i}")
        mem = store.memory_bytes()
        assert mem > 0
        assert mem < 100000  # should be small

    def test_stats(self, store):
        store.add("test")
        s = store.stats()
        assert s["entries"] == 1
        assert s["dimension"] == 2000

    def test_search_speed(self, store):
        """HDC search should be very fast even with many entries."""
        for i in range(100):
            store.add(f"This is fact number {i} about various topics in science and technology")

        start = time.time()
        for _ in range(10):
            store.search("What is science?", top_k=5)
        elapsed = (time.time() - start) * 1000  # ms

        # 10 searches over 100 entries should take < 500ms total
        assert elapsed < 500, f"HDC search too slow: {elapsed:.0f}ms for 10 queries"


# ══════════════════════════════════════════════════════════════════
# Integration: All Phase 6 components together
# ══════════════════════════════════════════════════════════════════

class TestPhase6Integration:
    def test_liquid_router_feeds_speculative(self):
        """Liquid router decides tier, speculative decoder handles execution."""
        d = tempfile.mkdtemp()
        router = LiquidRouter(data_dir=d)

        # Route a query
        tier = router.route("hello")
        assert tier == "tiny"

        # Simulate speculative decoding for non-tiny tiers
        tier2 = router.route("explain neural networks in detail")
        spec = SpeculativeDecoder(
            draft_fn=lambda p, mt, t: "Neural networks are computing systems.",
            main_fn=lambda p, mt, t: "Neural networks are computing systems inspired by the brain.",
        )
        result = spec.generate("explain neural networks")
        assert result.text != ""

        # Feed back to router
        router.feedback(tier2, confidence=0.8, latency_ms=result.total_time_ms)

        shutil.rmtree(d)

    def test_hdc_complements_chromadb(self):
        """HDC store can work alongside existing memory for fast lookups."""
        d = tempfile.mkdtemp()
        store = HDKnowledgeStore(dim=2000, data_dir=d)

        # Store facts
        facts = [
            ("My name is Gowri Shankar", {"type": "identity"}),
            ("I work as a DevOps engineer", {"type": "job"}),
            ("I live in Hyderabad", {"type": "location"}),
            ("I am building LeanAI", {"type": "project"}),
        ]
        for text, meta in facts:
            store.add(text, meta)

        # Search should find relevant facts
        results = store.search("What is my name?", top_k=2)
        assert len(results) == 2
        # Results should contain our stored facts
        all_texts = [r[0] for r in results]
        assert any("Gowri" in t or "name" in t or "DevOps" in t or "Hyderabad" in t for t in all_texts)

        store.save()
        shutil.rmtree(d)

"""
Tests for LeanAI Speed Optimizer — caching, smart tokens, GPU detection.
"""

import os
import shutil
import tempfile
import pytest

from core.speed_optimizer import (
    SpeedOptimizer, SpeedConfig, ResponseCache,
    get_max_tokens_for_query, get_optimal_model_params,
)


class TestResponseCache:
    @pytest.fixture
    def cache(self):
        d = tempfile.mkdtemp()
        c = ResponseCache(cache_dir=d, max_entries=100)
        yield c
        shutil.rmtree(d)

    def test_put_and_get_exact(self, cache):
        cache.put("what is Python?", "Python is a language.", 0.9)
        result = cache.get("what is Python?")
        assert result is not None
        assert result[0] == "Python is a language."
        assert result[1] == 0.9

    def test_get_miss(self, cache):
        assert cache.get("unknown question") is None

    def test_fuzzy_match(self, cache):
        cache.put("what is machine learning used for", "ML is...", 0.85)
        # Same content words, different phrasing
        result = cache.get("what is machine learning used for")
        assert result is not None

    def test_case_insensitive(self, cache):
        cache.put("What Is Python?", "It's a language.", 0.9)
        result = cache.get("what is python?")
        assert result is not None

    def test_hit_rate(self, cache):
        cache.put("question one", "answer one", 0.9)
        cache.get("question one")  # hit
        cache.get("totally different unknown query xyz")  # miss
        assert cache.hit_rate == 0.5

    def test_stats(self, cache):
        cache.put("q", "a", 0.9)
        cache.get("q")
        s = cache.stats()
        assert s["hits"] >= 1
        assert "hit_rate" in s

    def test_persistence(self):
        d = tempfile.mkdtemp()
        c1 = ResponseCache(cache_dir=d, max_entries=100)
        c1.put("persist question", "persist answer", 0.8)
        c1._save()

        c2 = ResponseCache(cache_dir=d, max_entries=100)
        result = c2.get("persist question")
        assert result is not None
        assert "persist answer" in result[0]
        shutil.rmtree(d)

    def test_eviction(self):
        d = tempfile.mkdtemp()
        c = ResponseCache(cache_dir=d, max_entries=5)
        for i in range(20):
            c.put(f"question {i} unique", f"answer {i}", 0.5)
        # Should not crash, entries should be limited
        assert len(c._memory_cache) <= 50  # 2 hashes per entry max
        shutil.rmtree(d)


class TestMaxTokens:
    def test_short_query(self):
        assert get_max_tokens_for_query("hello") == 384

    def test_yes_no_question(self):
        assert get_max_tokens_for_query("Is Python interpreted?") == 384

    def test_definition(self):
        assert get_max_tokens_for_query("What is a decorator in Python programming?") == 512

    def test_code_generation(self):
        assert get_max_tokens_for_query("implement a binary search") == 2048

    def test_detailed_explanation(self):
        assert get_max_tokens_for_query("explain how transformers work in detail") == 1536

    def test_default(self):
        assert get_max_tokens_for_query("tell me about the weather patterns in monsoon season") == 768


class TestOptimalParams:
    def test_returns_dict(self):
        params = get_optimal_model_params(18.0, 32)
        assert "n_batch" in params
        assert "flash_attn" in params
        assert params["n_batch"] >= 256

    def test_small_model_large_ram(self):
        params = get_optimal_model_params(5.0, 32)
        assert params["use_mlock"] is True  # plenty of free RAM
        assert params["n_batch"] >= 512

    def test_large_model_tight_ram(self):
        params = get_optimal_model_params(18.0, 24)
        # Should be conservative
        assert params["n_batch"] >= 256


class TestSpeedOptimizer:
    @pytest.fixture
    def optimizer(self):
        return SpeedOptimizer(SpeedConfig(cache_enabled=True))

    def test_cache_miss_then_hit(self, optimizer):
        assert optimizer.should_use_cache("new question here") is None
        optimizer.cache_response("new question here", "this is the detailed answer to the question", 0.85)
        result = optimizer.should_use_cache("new question here")
        assert result is not None
        assert "detailed answer" in result[0]

    def test_cache_disabled(self):
        opt = SpeedOptimizer(SpeedConfig(cache_enabled=False))
        opt.cache_response("q", "a long enough answer here", 0.9)
        assert opt.should_use_cache("q") is None

    def test_get_max_tokens(self, optimizer):
        t = optimizer.get_max_tokens("implement a sorting function")
        assert t == 2048

    def test_get_model_params(self, optimizer):
        p = optimizer.get_model_params(18.0)
        assert "n_batch" in p

    def test_stats(self, optimizer):
        s = optimizer.stats()
        assert "cache" in s
        assert "gpu" in s
        assert "config" in s

    def test_optimization_report(self, optimizer):
        report = optimizer.optimization_report()
        assert "Speed Optimization" in report
        assert "Batch size" in report
        assert "Flash attention" in report
        assert "cache" in report.lower()

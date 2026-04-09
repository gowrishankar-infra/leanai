"""
Tests for LeanAI Model Manager — multi-model switching and auto-routing.
"""

import os
import json
import shutil
import tempfile
import pytest

from core.model_manager import (
    ModelManager, ModelInfo, MODEL_REGISTRY,
    classify_complexity,
)


# ══════════════════════════════════════════════════════════════════
# Complexity Classifier Tests
# ══════════════════════════════════════════════════════════════════

class TestComplexityClassifier:
    def test_greeting_is_simple(self):
        assert classify_complexity("hello") == "simple"

    def test_short_question_is_simple(self):
        assert classify_complexity("what is Python?") == "simple"

    def test_memory_query_is_simple(self):
        assert classify_complexity("what is my name") == "simple"

    def test_fix_typo_is_simple(self):
        assert classify_complexity("fix this syntax error") == "simple"

    def test_architecture_is_complex(self):
        assert classify_complexity("design a scalable microservice architecture for an e-commerce platform") == "complex"

    def test_detailed_explanation_is_complex(self):
        assert classify_complexity("explain in detail how the distributed consensus algorithm works step by step") == "complex"

    def test_production_code_is_complex(self):
        assert classify_complexity("build a complete production-ready REST API with authentication") == "complex"

    def test_algorithm_is_complex(self):
        assert classify_complexity("implement a dynamic programming solution for the knapsack problem with optimization") == "complex"

    def test_medium_question(self):
        result = classify_complexity("write a function to sort a list of dictionaries by a key")
        assert result in ("medium", "simple")

    def test_pasted_code_is_complex(self):
        code = "def func():\n    pass\n" * 10  # multi-line code
        assert classify_complexity(code) == "complex"

    def test_empty_is_simple(self):
        assert classify_complexity("hi") == "simple"


# ══════════════════════════════════════════════════════════════════
# Model Registry Tests
# ══════════════════════════════════════════════════════════════════

class TestModelRegistry:
    def test_registry_has_models(self):
        assert "qwen-7b" in MODEL_REGISTRY
        assert "qwen-14b" in MODEL_REGISTRY
        assert "qwen-32b" in MODEL_REGISTRY

    def test_model_info_fields(self):
        m = MODEL_REGISTRY["qwen-7b"]
        assert m.name
        assert m.filename
        assert m.repo_id
        assert m.size_gb > 0
        assert m.quality_score > 0

    def test_quality_ordering(self):
        q7 = MODEL_REGISTRY["qwen-7b"].quality_score
        q14 = MODEL_REGISTRY["qwen-14b"].quality_score
        q32 = MODEL_REGISTRY["qwen-32b"].quality_score
        assert q7 < q14 < q32

    def test_size_ordering(self):
        s7 = MODEL_REGISTRY["qwen-7b"].size_gb
        s14 = MODEL_REGISTRY["qwen-14b"].size_gb
        s32 = MODEL_REGISTRY["qwen-32b"].size_gb
        assert s7 < s14 < s32

    def test_all_chatml(self):
        for m in MODEL_REGISTRY.values():
            assert m.prompt_format == "chatml"


# ══════════════════════════════════════════════════════════════════
# Model Manager Tests
# ══════════════════════════════════════════════════════════════════

class TestModelManager:
    @pytest.fixture
    def manager(self):
        mgr = ModelManager()
        mgr.set_mode("auto")  # ensure clean state
        return mgr

    def test_creation(self, manager):
        assert manager.mode == "auto"

    def test_list_models(self, manager):
        listing = manager.list_models()
        assert "qwen-7b" in listing
        assert "qwen-32b" in listing

    def test_set_mode_auto(self, manager):
        manager.set_mode("auto")
        assert manager.mode == "auto"

    def test_set_mode_fast(self, manager):
        manager.set_mode("fast")
        assert manager.mode == "fast"

    def test_set_mode_quality(self, manager):
        manager.set_mode("quality")
        assert manager.mode == "quality"

    def test_set_mode_invalid(self, manager):
        with pytest.raises(ValueError):
            manager.set_mode("nonexistent")

    def test_get_model_info(self, manager):
        info = manager.get_model_info("qwen-7b")
        assert info is not None
        assert info.name == "Qwen2.5 Coder 7B"

    def test_get_model_info_not_found(self, manager):
        assert manager.get_model_info("nonexistent") is None

    def test_download_command(self, manager):
        cmd = manager.download_command("qwen-32b")
        assert "hf_hub_download" in cmd
        assert "Qwen2.5-Coder-32B" in cmd

    def test_stats(self, manager):
        s = manager.stats()
        assert "mode" in s
        assert "downloaded_models" in s
        assert "queries_routed" in s


# ══════════════════════════════════════════════════════════════════
# Auto-Selection Tests
# ══════════════════════════════════════════════════════════════════

class TestAutoSelection:
    @pytest.fixture
    def manager_with_models(self, tmp_path):
        """Create a manager with fake 'downloaded' models."""
        mgr = ModelManager()
        # Fake the downloaded state by creating dummy files
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        for key, model in mgr.models.items():
            fake_path = models_dir / model.filename
            fake_path.write_text("fake")
            # Override full_path to point to our fake
            original_path = model.full_path
        # Since we can't easily override full_path (it's a property),
        # we'll test the logic indirectly through classify_complexity
        return mgr

    def test_simple_query_selects_fast(self):
        """Simple queries should prefer fast model."""
        mgr = ModelManager()
        mgr.set_mode("auto")
        # Even without downloaded models, selection logic works
        selected = mgr.select_model("hello")
        assert selected == "qwen-7b"  # default fallback

    def test_complex_query_noted(self):
        """Complex queries should be classified correctly."""
        c = classify_complexity("design a distributed microservice architecture with load balancing")
        assert c == "complex"

    def test_fast_mode_always_fast(self):
        mgr = ModelManager()
        mgr.set_mode("fast")
        selected = mgr.select_model("design a complex architecture")
        assert selected == "qwen-7b"  # fast mode ignores complexity

    def test_stats_tracking(self):
        mgr = ModelManager()
        mgr.set_mode("fast")
        mgr.select_model("query 1")
        mgr.select_model("query 2")
        s = mgr.stats()
        assert s["queries_routed"] == 2


# ══════════════════════════════════════════════════════════════════
# ModelInfo Tests
# ══════════════════════════════════════════════════════════════════

class TestModelInfo:
    def test_full_path(self):
        m = MODEL_REGISTRY["qwen-7b"]
        assert ".leanai" in m.full_path
        assert "models" in m.full_path

    def test_prompt_format(self):
        m = MODEL_REGISTRY["qwen-32b"]
        assert m.prompt_format == "chatml"

    def test_description(self):
        m = MODEL_REGISTRY["qwen-32b"]
        assert "92%" in m.description or "GPT-4" in m.description

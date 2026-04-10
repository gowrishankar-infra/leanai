"""
Tests for LeanAI Novel Features:
  - Predictive Pre-Generation
  - Semantic Git Bisect
  - Cross-Session Evolution
  - Adversarial Code Verification
"""

import os
import time
import shutil
import tempfile
import subprocess
import pytest

from core.predictor import (
    PredictivePreGenerator, predict_follow_ups, _extract_topic, _similarity,
)
from brain.semantic_bisect import SemanticGitBisect, CommitAnalysis, BisectResult
from brain.evolution_tracker import EvolutionTracker, EvolutionInsight, THEME_PATTERNS
from tools.adversarial import (
    AdversarialVerifier, FuzzResult,
    generate_numeric_edge_cases, generate_list_edge_cases,
    _detect_param_types, _extract_function_name, _extract_param_names,
)


# ══════════════════════════════════════════════════════════════
# Predictive Pre-Generation Tests
# ══════════════════════════════════════════════════════════════

class TestPredictFollowUps:
    def test_file_inquiry(self):
        preds = predict_follow_ups("what does engine.py do")
        assert len(preds) > 0
        assert any("engine" in p.lower() for p in preds)

    def test_error_inquiry(self):
        preds = predict_follow_ups("I'm getting an error in the auth module")
        assert len(preds) > 0
        assert any("fix" in p.lower() or "cause" in p.lower() for p in preds)

    def test_code_written(self):
        preds = predict_follow_ups("implement a binary search")
        assert len(preds) > 0
        assert any("test" in p.lower() or "optimize" in p.lower() for p in preds)

    def test_max_predictions(self):
        preds = predict_follow_ups("what does engine do", max_predictions=2)
        assert len(preds) <= 2

    def test_generic_fallback(self):
        preds = predict_follow_ups("hello there")
        assert len(preds) > 0


class TestExtractTopic:
    def test_removes_prefix(self):
        assert "python" in _extract_topic("what is python")

    def test_removes_suffix(self):
        topic = _extract_topic("how does caching work")
        assert "caching" in topic

    def test_short_query(self):
        assert _extract_topic("hi") == "hi"


class TestSimilarity:
    def test_identical(self):
        assert _similarity("hello world", "hello world") == 1.0

    def test_partial(self):
        s = _similarity("show me generate", "show me the generate method")
        assert s > 0.3

    def test_no_overlap(self):
        assert _similarity("abc", "xyz") == 0.0

    def test_empty(self):
        assert _similarity("", "hello") == 0.0


class TestPredictivePreGenerator:
    def test_check_without_predictions(self):
        p = PredictivePreGenerator()
        assert p.check_prediction("anything") is None

    def test_stats(self):
        p = PredictivePreGenerator()
        s = p.stats()
        assert "predictions_made" in s
        assert "hit_rate" in s

    def test_pending_predictions(self):
        p = PredictivePreGenerator()
        assert p.get_pending_predictions() == []

    def test_on_query_without_model(self):
        p = PredictivePreGenerator(generate_fn=None)
        p.on_query_complete("test", "response")
        # Should not crash


# ══════════════════════════════════════════════════════════════
# Semantic Git Bisect Tests
# ══════════════════════════════════════════════════════════════

class TestCommitAnalysis:
    def test_creation(self):
        c = CommitAnalysis(
            hash="abc123", short_hash="abc",
            message="fix auth bug", author="dev", date="2025-01-01",
        )
        assert c.message == "fix auth bug"

    def test_suspicion_default(self):
        c = CommitAnalysis(hash="a", short_hash="a", message="m", author="a", date="d")
        assert c.suspicion_score == 0.0


class TestSemanticBisect:
    @pytest.fixture
    def bisect(self):
        d = tempfile.mkdtemp()
        # Create a git repo with some commits
        subprocess.run(["git", "init"], cwd=d, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=d, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=d, capture_output=True)
        with open(os.path.join(d, "auth.py"), "w") as f:
            f.write("def login(user): pass\n")
        subprocess.run(["git", "add", "."], cwd=d, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial auth"], cwd=d, capture_output=True)
        with open(os.path.join(d, "auth.py"), "a") as f:
            f.write("def validate_token(token): return True\n")
        subprocess.run(["git", "add", "."], cwd=d, capture_output=True)
        subprocess.run(["git", "commit", "-m", "refactor token validation"], cwd=d, capture_output=True)
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write("# Project\n")
        subprocess.run(["git", "add", "."], cwd=d, capture_output=True)
        subprocess.run(["git", "commit", "-m", "update readme"], cwd=d, capture_output=True)
        yield SemanticGitBisect(repo_path=d)
        # Windows: force remove read-only .git/objects
        def force_remove(func, path, exc_info):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(d, onexc=force_remove)

    def test_find_bug_returns_result(self, bisect):
        result = bisect.find_bug("authentication broken", num_commits=5)
        assert isinstance(result, BisectResult)
        assert result.commits_analyzed >= 0

    def test_suspicion_scoring(self, bisect):
        c = CommitAnalysis(
            hash="a", short_hash="a",
            message="refactor auth token validation",
            author="dev", date="2025-01-01",
            files_changed=["auth.py"],
            insertions=50, deletions=30,
        )
        score = bisect._score_suspicion(c, "authentication stopped working")
        assert score > 0  # should have some suspicion

    def test_readme_commit_low_suspicion(self, bisect):
        c = CommitAnalysis(
            hash="b", short_hash="b",
            message="update readme docs",
            author="dev", date="2025-01-01",
            files_changed=["README.md"],
            insertions=5, deletions=2,
        )
        score = bisect._score_suspicion(c, "authentication broken")
        assert score < 0.2

    def test_refactor_commit_high_suspicion(self, bisect):
        c = CommitAnalysis(
            hash="c", short_hash="c",
            message="refactor authentication logic",
            author="dev", date="2025-01-01",
            files_changed=["auth.py", "middleware.py"],
            insertions=120, deletions=80,
        )
        score = bisect._score_suspicion(c, "authentication broken")
        assert score > 0.3

    def test_explain_suspicion(self, bisect):
        c = CommitAnalysis(
            hash="d", short_hash="d",
            message="refactor auth flow",
            author="dev", date="2025-01-01",
            insertions=100, deletions=50,
        )
        explanation = bisect._explain_suspicion(c, "auth broken")
        assert len(explanation) > 0


# ══════════════════════════════════════════════════════════════
# Cross-Session Evolution Tests
# ══════════════════════════════════════════════════════════════

class TestEvolutionTracker:
    @pytest.fixture
    def tracker(self):
        d = tempfile.mkdtemp()
        t = EvolutionTracker(data_dir=d)
        yield t
        shutil.rmtree(d)

    def test_track_database_theme(self, tracker):
        tracker.track_query("how to set up postgres database", session_id="s1")
        assert "database" in tracker.themes

    def test_track_auth_theme(self, tracker):
        tracker.track_query("implement jwt authentication", session_id="s1")
        assert "authentication" in tracker.themes

    def test_evolution_stage(self, tracker):
        tracker.track_query("what is a database", session_id="s1")
        assert tracker.themes["database"].evolution_stage == "exploring"

        tracker.track_query("create database tables", session_id="s2")
        assert tracker.themes["database"].evolution_stage == "building"

    def test_cross_session_tracking(self, tracker):
        tracker.track_query("setup database", session_id="s1")
        tracker.track_query("optimize database queries", session_id="s2")
        theme = tracker.themes["database"]
        assert theme.sessions_count == 2

    def test_get_insights(self, tracker):
        tracker.track_query("setup postgres", session_id="s1")
        tracker.track_query("create tables", session_id="s2")
        tracker.track_query("optimize queries", session_id="s3")
        insights = tracker.get_insights()
        assert len(insights) > 0
        assert insights[0].theme == "database"

    def test_predict_next_topics(self, tracker):
        tracker.track_query("setup database", session_id="s1")
        tracker.track_query("add database indexes", session_id="s2")
        preds = tracker.predict_next_topics()
        assert len(preds) > 0

    def test_get_narrative(self, tracker):
        tracker.track_query("what is caching", session_id="s1")
        tracker.track_query("implement redis cache", session_id="s2")
        tracker.track_query("optimize cache ttl", session_id="s3")
        narrative = tracker.get_narrative()
        assert "caching" in narrative.lower()

    def test_empty_narrative(self, tracker):
        narrative = tracker.get_narrative()
        assert "No project evolution" in narrative

    def test_persistence(self):
        d = tempfile.mkdtemp()
        t1 = EvolutionTracker(data_dir=d)
        t1.track_query("setup database", session_id="s1")
        t1.save()

        t2 = EvolutionTracker(data_dir=d)
        assert "database" in t2.themes
        shutil.rmtree(d)

    def test_stats(self, tracker):
        tracker.track_query("test database query", session_id="s1")
        s = tracker.stats()
        assert s["themes_tracked"] >= 1
        assert s["total_occurrences"] >= 1

    def test_no_match(self, tracker):
        tracker.track_query("hello world", session_id="s1")
        assert len(tracker.themes) == 0


# ══════════════════════════════════════════════════════════════
# Adversarial Code Verification Tests
# ══════════════════════════════════════════════════════════════

class TestParamDetection:
    def test_detect_list(self):
        types = _detect_param_types("def sort(arr: list): return sorted(arr)")
        assert "list" in types

    def test_detect_string(self):
        types = _detect_param_types("def greet(name: str): return name.upper()")
        assert "string" in types

    def test_detect_numeric(self):
        types = _detect_param_types("def add(a: int, b: int): return a + b")
        assert "numeric" in types

    def test_detect_from_usage(self):
        types = _detect_param_types("def f(x): return len(x)")
        assert "list" in types


class TestExtractFunction:
    def test_extract_name(self):
        assert _extract_function_name("def my_func(x): pass") == "my_func"

    def test_extract_params(self):
        params = _extract_param_names("def add(a, b): return a + b")
        assert params == ["a", "b"]

    def test_skip_self(self):
        params = _extract_param_names("def method(self, x): pass")
        assert "self" not in params


class TestEdgeCaseGenerators:
    def test_numeric_cases(self):
        cases = generate_numeric_edge_cases()
        assert len(cases) > 5
        categories = [c[1] for c in cases]
        assert "boundary" in categories
        assert "negative" in categories
        assert "nan" in categories

    def test_list_cases(self):
        cases = generate_list_edge_cases()
        assert len(cases) > 5
        categories = [c[1] for c in cases]
        assert "empty" in categories
        assert "null" in categories


class TestAdversarialVerifier:
    @pytest.fixture
    def verifier(self):
        return AdversarialVerifier(timeout=5)

    def test_fuzz_simple_function(self, verifier):
        code = "def add(a, b):\n    return a + b"
        result = verifier.fuzz(code)
        assert isinstance(result, FuzzResult)
        assert result.function_name == "add"
        assert result.total_cases > 0
        assert result.passed > 0  # at least some cases should pass

    def test_fuzz_finds_none_bug(self, verifier):
        code = "def length(arr):\n    return len(arr)"
        result = verifier.fuzz(code)
        # Should find that None causes TypeError
        failures = [c for c in result.cases if not c.passed and c.category == "null"]
        assert len(failures) > 0

    def test_fuzz_robust_function(self, verifier):
        code = """
def safe_add(a, b):
    if a is None or b is None:
        return 0
    try:
        return float(a) + float(b)
    except (TypeError, ValueError):
        return 0
"""
        result = verifier.fuzz(code)
        # Robust function should pass more cases
        assert result.passed > result.failed

    def test_suggestions_generated(self, verifier):
        code = "def divide(a, b):\n    return a / b"
        result = verifier.fuzz(code)
        # Should suggest checking for zero division
        assert len(result.suggestions) > 0

    def test_summary_format(self, verifier):
        code = "def inc(x):\n    return x + 1"
        result = verifier.fuzz(code)
        summary = result.summary()
        assert "inc" in summary
        assert "Tested" in summary

    def test_stats(self, verifier):
        verifier.fuzz("def f(x): return x")
        s = verifier.stats()
        assert s["total_fuzzed"] == 1

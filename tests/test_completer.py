"""
Tests for LeanAI Autocomplete Engine.
"""

import pytest
from core.completer import AutoCompleter, Completion, PYTHON_BUILTINS, PYTHON_SNIPPETS


# ── Mock Brain ────────────────────────────────────────────────

class MockAnalysis:
    def __init__(self, classes=None):
        self.classes = classes or []


class MockGraph:
    def __init__(self):
        self._function_lookup = {
            "generate": "core/engine_v3.py:generate",
            "generate_response": "core/engine_v3.py:generate_response",
            "get_model_path": "core/model_manager.py:get_model_path",
            "scan": "brain/project_brain.py:scan",
            "search": "brain/session_store.py:search",
            "add_example": "training/finetune_pipeline.py:add_example",
            "build_prompt": "core/engine_v3.py:build_prompt",
            "classify_complexity": "core/model_manager.py:classify_complexity",
            "execute_and_verify": "tools/executor.py:execute_and_verify",
        }

    def stats(self):
        return {"files": 10, "functions": 9, "classes": 5}


class MockBrain:
    def __init__(self):
        self.graph = MockGraph()
        self._file_analyses = {
            "core/engine_v3.py": MockAnalysis(classes=["LeanAIEngine", "GenerationConfig", "LeanAIResponse"]),
            "core/model_manager.py": MockAnalysis(classes=["ModelManager", "ModelInfo"]),
            "brain/project_brain.py": MockAnalysis(classes=["ProjectBrain"]),
        }


# ══════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════

class TestCompletion:
    def test_to_dict(self):
        c = Completion(text="generate", label="generate()", kind="function", detail="engine.py")
        d = c.to_dict()
        assert d["text"] == "generate"
        assert d["label"] == "generate()"
        assert d["kind"] == "function"

    def test_insert_text_default(self):
        c = Completion(text="hello", label="hello")
        assert c.to_dict()["insertText"] == "hello"

    def test_insert_text_custom(self):
        c = Completion(text="func", label="func", insert_text="func()")
        assert c.to_dict()["insertText"] == "func()"


class TestAutoCompleterWithBrain:
    @pytest.fixture
    def completer(self):
        return AutoCompleter(brain=MockBrain())

    def test_complete_function_prefix(self, completer):
        results = completer.complete("gen")
        names = [r.text for r in results]
        assert "generate" in names
        assert "generate_response" in names

    def test_complete_function_exact_start(self, completer):
        results = completer.complete("scan")
        names = [r.text for r in results]
        assert "scan" in names

    def test_complete_partial_match(self, completer):
        results = completer.complete("compl")
        names = [r.text for r in results]
        assert "classify_complexity" in names

    def test_complete_class(self, completer):
        results = completer.complete("Lean")
        names = [r.text for r in results]
        assert "LeanAIEngine" in names

    def test_complete_model(self, completer):
        results = completer.complete("Model")
        names = [r.text for r in results]
        assert "ModelManager" in names

    def test_brain_results_have_file_detail(self, completer):
        results = completer.complete("gen")
        func_results = [r for r in results if r.kind == "function"]
        assert any("engine" in r.detail for r in func_results)

    def test_brain_results_highest_priority(self, completer):
        results = completer.complete("gen")
        if results:
            brain_results = [r for r in results if r.sort_order == 0]
            assert len(brain_results) > 0

    def test_no_results_for_unknown(self, completer):
        results = completer.complete("xyzabc123")
        # Should only get language keywords if any match
        brain_results = [r for r in results if r.sort_order == 0]
        assert len(brain_results) == 0

    def test_short_prefix_returns_empty(self, completer):
        results = completer.complete("g")
        # Single character — too short for brain matching
        # May return keyword matches
        assert isinstance(results, list)

    def test_empty_prefix(self, completer):
        results = completer.complete("")
        assert results == []

    def test_dot_notation(self, completer):
        results = completer.complete("engine.gen")
        names = [r.text for r in results]
        assert "generate" in names


class TestAutoCompleterWithoutBrain:
    @pytest.fixture
    def completer(self):
        return AutoCompleter(brain=None)

    def test_python_keywords(self, completer):
        results = completer.complete("fo")
        labels = [r.label for r in results]
        assert any("for" in l for l in labels)

    def test_python_builtins(self, completer):
        results = completer.complete("pri")
        texts = [r.text for r in results]
        assert any("print" in t for t in texts)

    def test_snippets(self, completer):
        results = completer.complete("cla")
        has_snippet = any(r.kind == "snippet" for r in results)
        has_keyword = any(r.kind == "keyword" for r in results)
        assert has_snippet or has_keyword

    def test_javascript_keywords(self, completer):
        results = completer.complete("con", language="javascript")
        texts = [r.text for r in results]
        assert any("const" in t for t in texts)

    def test_go_keywords(self, completer):
        results = completer.complete("fun", language="go")
        texts = [r.text for r in results]
        assert any("func" in t for t in texts)


class TestAutoCompleterStats:
    def test_stats_tracking(self):
        completer = AutoCompleter(brain=MockBrain())
        completer.complete("gen")
        completer.complete("xyz_nothing")
        s = completer.stats()
        assert s["total_completions"] == 2
        assert s["functions_indexed"] > 0

    def test_update_brain(self):
        completer = AutoCompleter(brain=None)
        assert completer.stats()["functions_indexed"] == 0
        completer.update_brain(MockBrain())
        assert completer.stats()["functions_indexed"] > 0


class TestMaxResults:
    def test_respects_max_results(self):
        completer = AutoCompleter(brain=MockBrain())
        results = completer.complete("gen", max_results=2)
        assert len(results) <= 2


class TestDeduplication:
    def test_no_duplicate_completions(self):
        completer = AutoCompleter(brain=MockBrain())
        results = completer.complete("gen")
        texts = [r.text for r in results]
        assert len(texts) == len(set(texts))

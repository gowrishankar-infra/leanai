"""
Tests for LeanAI Phase 7a — Persistent Project Brain
"""

import os
import time
import shutil
import tempfile
import pytest

from brain.analyzer import (
    analyze_python_file, analyze_file, FileAnalysis,
    FunctionInfo, ClassInfo, ImportInfo,
)
from brain.dependency_graph import DependencyGraph, GraphNode, GraphEdge
from brain.project_brain import ProjectBrain, BrainConfig


# ══════════════════════════════════════════════════════════════════
# Code Analyzer Tests
# ══════════════════════════════════════════════════════════════════

SAMPLE_PYTHON = '''
"""Module docstring for testing."""

import os
from pathlib import Path

MAX_SIZE = 1024

class Animal:
    """An animal class."""
    def __init__(self, name: str):
        self.name = name
    
    def speak(self) -> str:
        """Make a sound."""
        return "..."

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

def greet(name: str) -> str:
    """Greet someone by name."""
    message = f"Hello, {name}!"
    print(message)
    return message

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def complex_func(data):
    result = []
    for item in data:
        if item > 0:
            if item % 2 == 0:
                result.append(item)
            else:
                result.append(item * 2)
        else:
            result.append(0)
    return result
'''


class TestAnalyzePython:
    @pytest.fixture
    def analysis(self):
        return analyze_python_file("test.py", source=SAMPLE_PYTHON)

    def test_module_docstring(self, analysis):
        assert analysis.docstring == "Module docstring for testing."

    def test_total_lines(self, analysis):
        assert analysis.total_lines > 30

    def test_finds_functions(self, analysis):
        names = [f.name for f in analysis.functions]
        assert "greet" in names
        assert "add" in names
        assert "complex_func" in names

    def test_finds_classes(self, analysis):
        names = [c.name for c in analysis.classes]
        assert "Animal" in names
        assert "Dog" in names

    def test_class_inheritance(self, analysis):
        dog = next(c for c in analysis.classes if c.name == "Dog")
        assert "Animal" in dog.bases

    def test_class_methods(self, analysis):
        animal = next(c for c in analysis.classes if c.name == "Animal")
        assert "speak" in animal.methods
        assert "__init__" in animal.methods

    def test_function_args(self, analysis):
        greet = next(f for f in analysis.functions if f.name == "greet")
        assert any("name" in a for a in greet.args)

    def test_function_return_type(self, analysis):
        add = next(f for f in analysis.functions if f.name == "add")
        assert add.return_type == "int"

    def test_function_docstring(self, analysis):
        greet = next(f for f in analysis.functions if f.name == "greet")
        assert greet.docstring == "Greet someone by name."

    def test_function_calls_detected(self, analysis):
        greet = next(f for f in analysis.functions if f.name == "greet")
        assert "print" in greet.calls

    def test_complexity_simple(self, analysis):
        add = next(f for f in analysis.functions if f.name == "add")
        assert add.complexity == 1  # no branches

    def test_complexity_complex(self, analysis):
        cf = next(f for f in analysis.functions if f.name == "complex_func")
        assert cf.complexity > 3  # for loop + if/elif/else

    def test_imports(self, analysis):
        modules = [i.module for i in analysis.imports]
        assert "os" in modules
        assert "pathlib" in modules

    def test_from_import(self, analysis):
        path_imp = next(i for i in analysis.imports if i.module == "pathlib")
        assert path_imp.is_from is True
        assert "Path" in path_imp.names

    def test_global_vars(self, analysis):
        assert "MAX_SIZE" in analysis.global_vars

    def test_line_numbers(self, analysis):
        greet = next(f for f in analysis.functions if f.name == "greet")
        assert greet.line_start > 0
        assert greet.line_end >= greet.line_start

    def test_method_detection(self, analysis):
        speak_funcs = [f for f in analysis.functions if f.name == "speak"]
        assert len(speak_funcs) >= 1
        # At least one should be detected as a method
        assert any(f.is_method for f in speak_funcs)

    def test_function_summary(self, analysis):
        s = analysis.summary()
        assert "test.py" in s
        assert "functions" in s

    def test_to_dict(self, analysis):
        d = analysis.to_dict()
        assert "functions" in d
        assert "classes" in d
        assert "imports" in d

    def test_syntax_error_handled(self):
        bad_code = "def broken(\n  return 42"
        analysis = analyze_python_file("bad.py", source=bad_code)
        assert analysis.error is not None
        assert "SyntaxError" in analysis.error

    def test_empty_file(self):
        analysis = analyze_python_file("empty.py", source="")
        assert analysis.total_lines == 1
        assert len(analysis.functions) == 0

    def test_analyze_file_nonpython(self):
        analysis = analyze_file("script.js", source="function hello() { return 1; }")
        assert analysis.language == "js"
        assert analysis.total_lines > 0


# ══════════════════════════════════════════════════════════════════
# Dependency Graph Tests
# ══════════════════════════════════════════════════════════════════

class TestDependencyGraph:
    @pytest.fixture
    def graph(self):
        g = DependencyGraph()
        # Analyze two related files
        code_a = '''
from b import helper
def main():
    result = helper()
    print(result)
'''
        code_b = '''
def helper():
    return 42
'''
        analysis_a = analyze_python_file("a.py", source=code_a)
        analysis_b = analyze_python_file("b.py", source=code_b)
        g.add_file_analysis(analysis_a)
        g.add_file_analysis(analysis_b)
        g.resolve_references()
        return g

    def test_files_added(self, graph):
        assert graph.num_files == 2

    def test_functions_added(self, graph):
        assert graph.num_functions >= 2  # main + helper

    def test_import_edge(self, graph):
        import_edges = [e for e in graph.edges if e.edge_type == "imports"]
        assert len(import_edges) >= 1

    def test_defines_edge(self, graph):
        defines = [e for e in graph.edges if e.edge_type == "defines"]
        assert len(defines) >= 2  # a.py defines main, b.py defines helper

    def test_calls_edge_resolved(self, graph):
        calls = [e for e in graph.edges if e.edge_type == "calls"]
        # main() calls helper() and print()
        assert len(calls) >= 1

    def test_find_function(self, graph):
        node = graph.find_function("helper")
        assert node is not None
        assert node.name == "helper"

    def test_find_function_not_found(self, graph):
        assert graph.find_function("nonexistent") is None

    def test_get_file_functions(self, graph):
        funcs = graph.get_file_functions("a.py")
        assert len(funcs) >= 1

    def test_stats(self, graph):
        s = graph.stats()
        assert s["files"] == 2
        assert s["functions"] >= 2
        assert s["edges"] > 0

    def test_summary(self, graph):
        s = graph.summary()
        assert "files" in s
        assert "functions" in s


# ══════════════════════════════════════════════════════════════════
# Project Brain Tests
# ══════════════════════════════════════════════════════════════════

class TestProjectBrain:
    @pytest.fixture
    def project_dir(self):
        d = tempfile.mkdtemp()
        # Create a mini project
        with open(os.path.join(d, "main.py"), "w") as f:
            f.write('''
"""Main application module."""
from utils import format_name

def run():
    """Run the application."""
    name = format_name("Gowri", "Shankar")
    print(f"Hello, {name}")

if __name__ == "__main__":
    run()
''')
        with open(os.path.join(d, "utils.py"), "w") as f:
            f.write('''
"""Utility functions."""

def format_name(first: str, last: str) -> str:
    """Format a full name."""
    return f"{first} {last}"

def validate_email(email: str) -> bool:
    """Check if email is valid."""
    return "@" in email and "." in email
''')
        os.makedirs(os.path.join(d, "tests"), exist_ok=True)
        with open(os.path.join(d, "tests", "test_utils.py"), "w") as f:
            f.write('''
from utils import format_name, validate_email

def test_format_name():
    assert format_name("A", "B") == "A B"

def test_validate_email():
    assert validate_email("a@b.com") is True
    assert validate_email("invalid") is False
''')
        yield d
        shutil.rmtree(d)

    @pytest.fixture
    def brain(self, project_dir):
        data_dir = tempfile.mkdtemp()
        brain = ProjectBrain(project_dir, BrainConfig(
            project_path=project_dir, data_dir=data_dir,
        ))
        brain.scan()
        yield brain
        shutil.rmtree(data_dir)

    def test_scan_finds_files(self, brain):
        stats = brain.stats()
        assert stats["files_indexed"] == 3  # main.py, utils.py, test_utils.py

    def test_scan_returns_stats(self, project_dir):
        data_dir = tempfile.mkdtemp()
        brain = ProjectBrain(project_dir, BrainConfig(
            project_path=project_dir, data_dir=data_dir,
        ))
        result = brain.scan()
        assert result["files_found"] == 3
        assert result["files_analyzed"] == 3
        assert result["scan_time_ms"] >= 0
        shutil.rmtree(data_dir)

    def test_incremental_scan(self, brain, project_dir):
        """Second scan should detect no changes."""
        result = brain.scan()
        assert result["files_unchanged"] == 3
        assert result["files_analyzed"] == 0

    def test_incremental_scan_detects_change(self, brain, project_dir):
        """Modified file should be re-analyzed."""
        time.sleep(0.1)
        with open(os.path.join(project_dir, "utils.py"), "a") as f:
            f.write("\n\ndef new_func():\n    return True\n")
        # Touch to update mtime
        os.utime(os.path.join(project_dir, "utils.py"))
        result = brain.scan()
        assert result["files_analyzed"] >= 1

    def test_describe_file(self, brain):
        desc = brain.describe_file("utils.py")
        assert "utils.py" in desc
        assert "format_name" in desc
        assert "validate_email" in desc

    def test_describe_file_not_found(self, brain):
        desc = brain.describe_file("nonexistent.py")
        assert "not indexed" in desc.lower()

    def test_find_function(self, brain):
        result = brain.find_function("format_name")
        assert "format_name" in result
        assert "first" in result or "Args" in result

    def test_find_function_not_found(self, brain):
        result = brain.find_function("nonexistent")
        assert "not found" in result.lower()

    def test_project_summary(self, brain):
        summary = brain.project_summary()
        assert "Files:" in summary
        assert "Functions:" in summary

    def test_graph_has_data(self, brain):
        g = brain.graph.stats()
        assert g["files"] >= 3
        assert g["functions"] >= 4  # run, format_name, validate_email, test funcs

    def test_get_context_for_query(self, brain):
        ctx = brain.get_context_for_query("what does utils.py do")
        assert "utils.py" in ctx
        assert "format_name" in ctx

    def test_stats(self, brain):
        s = brain.stats()
        assert s["files_indexed"] == 3
        assert "graph" in s

    def test_cache_persistence(self, project_dir):
        """Brain should cache state and load it on next instantiation."""
        data_dir = tempfile.mkdtemp()
        b1 = ProjectBrain(project_dir, BrainConfig(
            project_path=project_dir, data_dir=data_dir,
        ))
        b1.scan()

        # Create new brain instance with same data_dir
        b2 = ProjectBrain(project_dir, BrainConfig(
            project_path=project_dir, data_dir=data_dir,
        ))
        # Should have cached file states
        assert len(b2._file_states) == 3
        shutil.rmtree(data_dir)

    def test_rescan_single_file(self, brain, project_dir):
        filepath = os.path.join(project_dir, "utils.py")
        result = brain.rescan_file(filepath)
        assert result is not None
        assert len(result.functions) >= 2

    def test_watcher_starts_and_stops(self, brain):
        brain.start_watching()
        assert brain._watching is True
        brain.stop_watching()
        assert brain._watching is False

    def test_scan_skips_pycache(self, brain, project_dir):
        """__pycache__ should be skipped."""
        cache_dir = os.path.join(project_dir, "__pycache__")
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "cached.py"), "w") as f:
            f.write("x = 1")
        result = brain.scan(force=True)
        # cached.py should NOT be in the results
        assert "__pycache__" not in str(brain._file_analyses.keys())

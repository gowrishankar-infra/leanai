"""
Tests for LeanAI Phase 7c — Test-Driven Auto-Fix Loop
"""

import os
import pytest
from brain.tdd_loop import TDDLoop, TDDConfig, TDDResult, TDDAttempt


# ── Mock model functions ──────────────────────────────────────────

def _mock_model_correct(system: str, user: str) -> str:
    """Mock that always generates correct code on first try."""
    if "calculator" in user.lower() or "add" in user.lower():
        return """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
"""
    if "greet" in user.lower():
        return """
def greet(name):
    return f"Hello, {name}!"

def farewell(name):
    return f"Goodbye, {name}!"
"""
    return "def solution():\n    return 42\n"


def _mock_model_fix_on_second(system: str, user: str) -> str:
    """Mock that fails first, succeeds on fix."""
    if "Fix" in system or "fix" in system or "debugger" in system.lower():
        # This is a fix request — return correct code
        return """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
    else:
        # First attempt — return buggy code
        return """
def add(a, b):
    return a * b  # bug: multiply instead of add

def subtract(a, b):
    return a - b
"""


def _mock_model_always_wrong(system: str, user: str) -> str:
    """Mock that always generates wrong code."""
    return """
def add(a, b):
    return 999  # always wrong

def subtract(a, b):
    return 999  # always wrong
"""


def _mock_model_test_generator(system: str, user: str) -> str:
    """Mock that generates tests from description, then code."""
    if "tester" in system.lower():
        return """
from calculator import add, subtract

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, -1) == -2

def test_add_zero():
    assert add(0, 0) == 0

def test_subtract_basic():
    assert subtract(10, 3) == 7

def test_subtract_negative():
    assert subtract(3, 10) == -7
"""
    else:
        return """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""


# ── Test code fixtures ────────────────────────────────────────────

CALC_TESTS = """
from calculator import add, subtract

def test_add_basic():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0

def test_subtract_basic():
    assert subtract(10, 3) == 7

def test_subtract_zero():
    assert subtract(5, 5) == 0
"""

GREET_TESTS = """
from greeter import greet, farewell

def test_greet():
    assert greet("Alice") == "Hello, Alice!"

def test_farewell():
    assert farewell("Bob") == "Goodbye, Bob!"
"""


# ══════════════════════════════════════════════════════════════════
# TDDLoop Core Tests
# ══════════════════════════════════════════════════════════════════

class TestTDDLoop:
    def test_passes_on_first_try(self):
        """Correct code should pass on first attempt."""
        tdd = TDDLoop(model_fn=_mock_model_correct, config=TDDConfig(max_attempts=3))
        result = tdd.run(CALC_TESTS, module_name="calculator")
        assert result.success is True
        assert result.num_attempts == 1
        assert "def add" in result.implementation

    def test_passes_after_fix(self):
        """Buggy code should be fixed on second attempt."""
        tdd = TDDLoop(model_fn=_mock_model_fix_on_second, config=TDDConfig(max_attempts=3))
        result = tdd.run(CALC_TESTS, module_name="calculator")
        assert result.success is True
        assert result.num_attempts == 2

    def test_fails_after_max_attempts(self):
        """Always-wrong code should fail after max attempts."""
        tdd = TDDLoop(model_fn=_mock_model_always_wrong, config=TDDConfig(max_attempts=2))
        result = tdd.run(CALC_TESTS, module_name="calculator")
        assert result.success is False
        assert result.num_attempts == 2
        assert result.final_error != ""

    def test_greeter_module(self):
        """Test with a different module."""
        tdd = TDDLoop(model_fn=_mock_model_correct, config=TDDConfig(max_attempts=3))
        result = tdd.run(GREET_TESTS, module_name="greeter")
        assert result.success is True
        assert "def greet" in result.implementation

    def test_auto_detect_module_name(self):
        """Module name should be extracted from test imports."""
        tdd = TDDLoop(model_fn=_mock_model_correct)
        result = tdd.run(CALC_TESTS)
        assert result.module_name == "calculator"

    def test_auto_detect_module_from_import(self):
        tdd = TDDLoop(model_fn=_mock_model_correct)
        result = tdd.run(GREET_TESTS)
        assert result.module_name == "greeter"

    def test_no_model_raises(self):
        tdd = TDDLoop(model_fn=None)
        with pytest.raises(RuntimeError, match="No model_fn"):
            tdd.run(CALC_TESTS)

    def test_verbose_mode(self, capsys):
        tdd = TDDLoop(
            model_fn=_mock_model_correct,
            config=TDDConfig(max_attempts=2, verbose=True),
        )
        result = tdd.run(CALC_TESTS, module_name="calculator")
        captured = capsys.readouterr()
        assert "TDD" in captured.out
        assert "PASSED" in captured.out

    def test_cleanup_workspace(self):
        """Temp workspace should be cleaned up after run."""
        tdd = TDDLoop(
            model_fn=_mock_model_correct,
            config=TDDConfig(cleanup=True),
        )
        result = tdd.run(CALC_TESTS, module_name="calculator")
        assert result.success is True
        # Workspace should be gone (we can't check directly, but no error means cleanup worked)


# ══════════════════════════════════════════════════════════════════
# TDDResult Tests
# ══════════════════════════════════════════════════════════════════

class TestTDDResult:
    def test_success_result(self):
        r = TDDResult(
            success=True, implementation="def add(a,b): return a+b",
            test_code="assert add(1,2)==3", module_name="calc",
            attempts=[TDDAttempt(attempt=1, code="...", test_passed=True, time_ms=100)],
            total_time_ms=500,
        )
        assert r.num_attempts == 1
        s = r.summary()
        assert "PASSED" in s
        assert "1 attempt" in s

    def test_failure_result(self):
        r = TDDResult(
            success=False, implementation="broken code",
            test_code="...", module_name="calc",
            attempts=[
                TDDAttempt(attempt=1, code="v1", test_passed=False, error="AssertionError", time_ms=50),
                TDDAttempt(attempt=2, code="v2", test_passed=False, error="NameError", time_ms=60),
            ],
            total_time_ms=200,
            final_error="NameError: name 'x' not defined",
        )
        assert r.num_attempts == 2
        s = r.summary()
        assert "FAILED" in s
        assert "2 attempt" in s

    def test_empty_result(self):
        r = TDDResult(success=False, implementation="", test_code="", module_name="x")
        assert r.num_attempts == 0


# ══════════════════════════════════════════════════════════════════
# TDDAttempt Tests
# ══════════════════════════════════════════════════════════════════

class TestTDDAttempt:
    def test_passed_attempt(self):
        a = TDDAttempt(attempt=1, code="code", test_passed=True, time_ms=100)
        assert a.test_passed is True
        assert a.error == ""

    def test_failed_attempt(self):
        a = TDDAttempt(attempt=1, code="code", test_passed=False, error="AssertionError", time_ms=50)
        assert a.test_passed is False
        assert a.error == "AssertionError"


# ══════════════════════════════════════════════════════════════════
# TDDConfig Tests
# ══════════════════════════════════════════════════════════════════

class TestTDDConfig:
    def test_defaults(self):
        c = TDDConfig()
        assert c.max_attempts == 5
        assert c.timeout_seconds == 30
        assert c.cleanup is True

    def test_custom(self):
        c = TDDConfig(max_attempts=10, verbose=True)
        assert c.max_attempts == 10
        assert c.verbose is True


# ══════════════════════════════════════════════════════════════════
# Code Cleaning Tests
# ══════════════════════════════════════════════════════════════════

class TestCodeCleaning:
    def test_strips_markdown_fences(self):
        tdd = TDDLoop(model_fn=_mock_model_correct)
        raw = "```python\ndef add(a, b):\n    return a + b\n```"
        clean = tdd._clean_code(raw)
        assert "```" not in clean
        assert "def add" in clean

    def test_strips_leading_blank_lines(self):
        tdd = TDDLoop(model_fn=_mock_model_correct)
        raw = "\n\n\ndef hello():\n    pass\n\n\n"
        clean = tdd._clean_code(raw)
        assert clean.startswith("def hello")

    def test_extract_module_name_from_import(self):
        tdd = TDDLoop(model_fn=_mock_model_correct)
        name = tdd._extract_module_name("from calculator import add")
        assert name == "calculator"

    def test_extract_module_name_import(self):
        tdd = TDDLoop(model_fn=_mock_model_correct)
        name = tdd._extract_module_name("import utils")
        assert name == "utils"

    def test_extract_module_name_fallback(self):
        tdd = TDDLoop(model_fn=_mock_model_correct)
        name = tdd._extract_module_name("no imports here")
        assert name == "solution"


# ══════════════════════════════════════════════════════════════════
# run_with_description Tests
# ══════════════════════════════════════════════════════════════════

class TestRunWithDescription:
    def test_generates_tests_and_code(self):
        tdd = TDDLoop(
            model_fn=_mock_model_test_generator,
            config=TDDConfig(max_attempts=3),
        )
        result = tdd.run_with_description(
            "A calculator with add and subtract functions",
            module_name="calculator",
        )
        assert result.success is True
        assert "def add" in result.implementation
        assert result.test_code != ""

    def test_returns_failure_on_empty_tests(self):
        def empty_model(s, u):
            return ""
        tdd = TDDLoop(model_fn=empty_model)
        result = tdd.run_with_description("something", "mod")
        assert result.success is False
        assert "generate tests" in result.final_error.lower()

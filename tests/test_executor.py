"""
LeanAI · Phase 4b Tests — Code Executor
Run: python -m pytest tests/test_executor.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools.executor import CodeExecutor, ExecutionResult, VerifiedCode


class TestCodeExecutor:

    def setup_method(self):
        self.ex = CodeExecutor()

    # ── Basic execution ────────────────────────────────────────────────

    def test_simple_python_runs(self):
        result = self.ex.execute('print("hello")', "python")
        assert result.success
        assert "hello" in result.stdout

    def test_math_output_correct(self):
        result = self.ex.execute("print(2 + 2)", "python")
        assert result.success
        assert "4" in result.stdout

    def test_syntax_error_caught(self):
        result = self.ex.execute("def broken(\n    pass", "python")
        assert not result.success
        assert result.return_code != 0

    def test_runtime_error_caught(self):
        result = self.ex.execute("x = 1/0", "python")
        assert not result.success
        assert "ZeroDivisionError" in result.stderr or result.return_code != 0

    def test_name_error_caught(self):
        result = self.ex.execute("print(undefined_variable)", "python")
        assert not result.success
        assert result.error_type == "NameError"

    def test_timeout_enforced(self):
        result = self.ex.execute("while True: pass", "python")
        assert not result.success
        assert "Timeout" in result.stderr or result.return_code != 0

    def test_execution_time_recorded(self):
        result = self.ex.execute("print('fast')", "python")
        assert result.execution_time_ms >= 0

    def test_language_detected(self):
        result = self.ex.execute("print('hello')", "python")
        assert result.language == "python"

    def test_multiline_code(self):
        code = """
def add(a, b):
    return a + b

print(add(3, 4))
"""
        result = self.ex.execute(code, "python")
        assert result.success
        assert "7" in result.stdout

    def test_markdown_fences_stripped(self):
        code = """```python
print("hello from fenced code")
```"""
        result = self.ex.execute(code, "python")
        assert result.success
        assert "hello from fenced code" in result.stdout

    # ── Verify and auto-fix ───────────────────────────────────────────

    def test_verified_code_passes(self):
        code = "print('verified')"
        verified = self.ex.execute_and_verify(code)
        assert verified.passed
        assert verified.attempts == 1

    def test_verified_code_failing(self):
        code = "this is not valid python code!!!"
        verified = self.ex.execute_and_verify(code, auto_fix=False)
        assert not verified.passed

    def test_attempts_tracked(self):
        code = "print('works')"
        verified = self.ex.execute_and_verify(code)
        assert verified.attempts >= 1

    def test_history_populated(self):
        code = "print('test')"
        verified = self.ex.execute_and_verify(code)
        assert len(verified.history) >= 1

    def test_indentation_fix(self):
        code = "def f():\n\tprint('tab indented')\nf()"
        verified = self.ex.execute_and_verify(code)
        # Should either pass directly or after fix
        assert isinstance(verified, VerifiedCode)

    # ── Format result ─────────────────────────────────────────────────

    def test_format_passing(self):
        code = "print('hello')"
        verified = self.ex.execute_and_verify(code)
        formatted = self.ex.format_result(verified)
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_format_failing(self):
        code = "raise ValueError('intentional error')"
        verified = self.ex.execute_and_verify(code, auto_fix=False)
        formatted = self.ex.format_result(verified)
        assert isinstance(formatted, str)

    # ── Language detection ────────────────────────────────────────────

    def test_detect_python(self):
        assert self.ex._detect_language("def foo(): pass", "python") == "python"

    def test_detect_from_content(self):
        assert self.ex._detect_language("import os\nprint(os.getcwd())") == "python"

    # ── Test generation ───────────────────────────────────────────────

    def test_generate_tests_for_function(self):
        code = "def add(a, b):\n    return a + b"
        tests = self.ex.generate_tests(code)
        assert "add" in tests
        assert "def test_add" in tests

    def test_generate_tests_no_function(self):
        code = "x = 42\nprint(x)"
        tests = self.ex.generate_tests(code)
        assert tests == ""

    # ── Availability ──────────────────────────────────────────────────

    def test_python_available(self):
        assert self.ex.is_available("python")

    def test_available_languages_list(self):
        langs = self.ex.available_languages
        assert "python" in langs
        assert isinstance(langs, list)

    # ── Real code samples ─────────────────────────────────────────────

    def test_fibonacci(self):
        code = """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

for i in range(8):
    print(fib(i), end=' ')
"""
        result = self.ex.execute(code)
        assert result.success
        assert "0" in result.stdout

    def test_list_comprehension(self):
        code = "squares = [x**2 for x in range(5)]\nprint(squares)"
        result = self.ex.execute(code)
        assert result.success
        assert "0" in result.stdout and "16" in result.stdout

    def test_class_definition(self):
        code = """
class Dog:
    def __init__(self, name):
        self.name = name
    def bark(self):
        return f"{self.name} says woof!"

d = Dog("Rex")
print(d.bark())
"""
        result = self.ex.execute(code)
        assert result.success
        assert "woof" in result.stdout

    def test_bubble_sort_verified(self):
        code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

arr = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(arr))
"""
        verified = self.ex.execute_and_verify(code)
        assert verified.passed
        assert "11" in verified.final_result.stdout

    def test_sieve_of_eratosthenes(self):
        code = """
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

print(sieve(30))
"""
        verified = self.ex.execute_and_verify(code)
        assert verified.passed
        assert "29" in verified.final_result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
LeanAI — Adversarial Code Verification
Generates edge-case inputs designed to BREAK your code.

Instead of just running tests, this system:
  1. Analyzes your function's signature and logic
  2. Generates adversarial inputs: empty lists, None, negative numbers,
     huge inputs, special characters, boundary values
  3. Runs each adversarial input and reports failures
  4. Suggests fixes for each failure found

Example:
  /fuzz def sort(arr): return sorted(arr)
  
  LeanAI generates:
    ✓ sort([3,1,2]) → [1,2,3]
    ✓ sort([]) → []
    ✗ sort(None) → TypeError: 'NoneType' is not iterable
    ✗ sort([1, "2", 3]) → TypeError: '<' not supported
    ✗ sort([float('nan'), 1, 2]) → unstable sort with NaN
  
  Suggestion: Add input validation — check for None and non-numeric types
"""

import ast
import sys
import time
import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FuzzCase:
    """A single adversarial test case."""
    input_repr: str       # string representation of input
    category: str         # "boundary", "type_error", "empty", "overflow", etc.
    passed: bool = False
    output: str = ""
    error: str = ""
    error_type: str = ""

    def summary(self) -> str:
        icon = "✓" if self.passed else "✗"
        if self.passed:
            return f"  {icon} {self.input_repr} → {self.output[:80]}"
        else:
            return f"  {icon} {self.input_repr} → {self.error_type}: {self.error[:80]}"


@dataclass
class FuzzResult:
    """Result of adversarial verification."""
    function_name: str
    total_cases: int
    passed: int
    failed: int
    cases: List[FuzzCase] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    time_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Adversarial Verification: {self.function_name}",
            f"Tested: {self.total_cases} | Passed: {self.passed} | Failed: {self.failed}",
            "",
        ]

        # Show failures first
        failures = [c for c in self.cases if not c.passed]
        if failures:
            lines.append("Failures:")
            for c in failures[:10]:
                lines.append(c.summary())

        # Show passes
        passes = [c for c in self.cases if c.passed]
        if passes:
            lines.append(f"\nPassed ({len(passes)}):")
            for c in passes[:5]:
                lines.append(c.summary())
            if len(passes) > 5:
                lines.append(f"  ... and {len(passes)-5} more")

        # Suggestions
        if self.suggestions:
            lines.append(f"\nSuggested fixes:")
            for s in self.suggestions:
                lines.append(f"  → {s}")

        lines.append(f"\nTime: {self.time_ms:.0f}ms")
        return "\n".join(lines)


# ── Adversarial Input Generators ──────────────────────────

def generate_numeric_edge_cases() -> List[Tuple[str, str]]:
    """Edge cases for numeric parameters."""
    return [
        ("0", "boundary"),
        ("-1", "negative"),
        ("1", "boundary"),
        ("-999999", "overflow"),
        ("999999", "overflow"),
        ("0.0", "float"),
        ("-0.0", "negative_zero"),
        ("float('inf')", "infinity"),
        ("float('-inf')", "negative_infinity"),
        ("float('nan')", "nan"),
        ("0.1 + 0.2", "float_precision"),
    ]


def generate_string_edge_cases() -> List[Tuple[str, str]]:
    """Edge cases for string parameters."""
    return [
        ('""', "empty"),
        ('" "', "whitespace"),
        ('"a"', "single_char"),
        ('"a" * 10000', "very_long"),
        ('None', "null"),
        ('"\\n\\t\\r"', "special_chars"),
        ('"hello\\x00world"', "null_byte"),
        ('"<script>alert(1)</script>"', "xss"),
        ('"DROP TABLE users;--"', "sql_injection"),
        ('"' + "é" + '"', "unicode"),
    ]


def generate_list_edge_cases() -> List[Tuple[str, str]]:
    """Edge cases for list/array parameters."""
    return [
        ("[]", "empty"),
        ("[1]", "single"),
        ("[1, 2, 3]", "normal"),
        ("list(range(10000))", "very_large"),
        ("None", "null"),
        ("[None]", "contains_null"),
        ("[1, None, 3]", "mixed_null"),
        ("[1, 'a', 2.0]", "mixed_types"),
        ("[[1, 2], [3, 4]]", "nested"),
        ("[1, 1, 1, 1]", "all_same"),
        ("[5, 4, 3, 2, 1]", "reverse_sorted"),
        ("[float('nan'), 1, 2]", "contains_nan"),
    ]


def generate_dict_edge_cases() -> List[Tuple[str, str]]:
    """Edge cases for dictionary parameters."""
    return [
        ("{}", "empty"),
        ("None", "null"),
        ('{"a": 1}', "single"),
        ('{"a": None}', "null_value"),
        ('{1: "a", 2: "b"}', "int_keys"),
    ]


def _detect_param_types(code: str) -> List[str]:
    """Detect likely parameter types from function code."""
    types = []
    lower = code.lower()

    # Check type hints
    if ": int" in code or ": float" in code or "number" in lower:
        types.append("numeric")
    if ": str" in code or "string" in lower:
        types.append("string")
    if ": list" in code or ": List" in code or "array" in lower or "arr" in lower:
        types.append("list")
    if ": dict" in code or ": Dict" in code:
        types.append("dict")

    # Check usage patterns
    if "len(" in code or "append(" in code or "for " in code:
        if "list" not in types:
            types.append("list")
    if ".split(" in code or ".strip(" in code or ".lower(" in code:
        if "string" not in types:
            types.append("string")
    if "+" in code or "-" in code or "*" in code or "/" in code:
        if "numeric" not in types:
            types.append("numeric")

    if not types:
        types = ["numeric", "string", "list"]  # test all

    return types


def _extract_function_name(code: str) -> str:
    """Extract function name from code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except:
        pass
    # Fallback: regex
    import re
    m = re.search(r"def\s+(\w+)", code)
    return m.group(1) if m else "unknown"


def _extract_param_names(code: str) -> List[str]:
    """Extract parameter names from function signature."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                params = []
                for arg in node.args.args:
                    if arg.arg != "self":
                        params.append(arg.arg)
                return params
    except:
        pass
    return ["x"]


class AdversarialVerifier:
    """
    Generates and runs adversarial test cases against code.
    
    Usage:
        verifier = AdversarialVerifier()
        
        code = '''
        def sort_list(arr):
            return sorted(arr)
        '''
        
        result = verifier.fuzz(code)
        print(result.summary())
    """

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self._stats = {"total_fuzzed": 0, "total_bugs_found": 0}

    def fuzz(self, code: str, verbose: bool = False) -> FuzzResult:
        """
        Run adversarial verification on a piece of code.
        
        Args:
            code: Python code containing a function to test
            verbose: print progress
        
        Returns:
            FuzzResult with all test cases and suggestions
        """
        start = time.time()
        self._stats["total_fuzzed"] += 1

        func_name = _extract_function_name(code)
        param_names = _extract_param_names(code)
        param_types = _detect_param_types(code)

        if verbose:
            print(f"[Fuzz] Testing {func_name}({', '.join(param_names)})", flush=True)
            print(f"[Fuzz] Detected types: {', '.join(param_types)}", flush=True)

        # Generate adversarial inputs
        test_cases = []

        for ptype in param_types:
            if ptype == "numeric":
                cases = generate_numeric_edge_cases()
            elif ptype == "string":
                cases = generate_string_edge_cases()
            elif ptype == "list":
                cases = generate_list_edge_cases()
            elif ptype == "dict":
                cases = generate_dict_edge_cases()
            else:
                cases = generate_numeric_edge_cases()

            for input_repr, category in cases:
                # Build test script
                if len(param_names) == 1:
                    call = f"{func_name}({input_repr})"
                else:
                    # For multi-param functions, use the same input for all
                    args = ", ".join([input_repr] * len(param_names))
                    call = f"{func_name}({args})"

                fuzz_case = self._run_case(code, call, input_repr, category)
                test_cases.append(fuzz_case)

        # Count results
        passed = sum(1 for c in test_cases if c.passed)
        failed = sum(1 for c in test_cases if not c.passed)
        self._stats["total_bugs_found"] += failed

        # Generate suggestions based on failures
        suggestions = self._generate_suggestions(test_cases, func_name)

        elapsed = (time.time() - start) * 1000

        return FuzzResult(
            function_name=func_name,
            total_cases=len(test_cases),
            passed=passed,
            failed=failed,
            cases=test_cases,
            suggestions=suggestions,
            time_ms=elapsed,
        )

    def _run_case(self, code: str, call: str, input_repr: str,
                  category: str) -> FuzzCase:
        """Run a single adversarial test case in a sandbox."""
        script = f"""
{code}

try:
    result = {call}
    print(f"OK: {{repr(result)}}")
except Exception as e:
    print(f"ERROR:{{type(e).__name__}}:{{e}}")
"""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(script)
                f.flush()
                script_path = f.name

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=self.timeout,
            )
            os.unlink(script_path)

            output = result.stdout.strip()
            stderr = result.stderr.strip()

            if output.startswith("OK:"):
                return FuzzCase(
                    input_repr=input_repr,
                    category=category,
                    passed=True,
                    output=output[3:].strip()[:200],
                )
            elif output.startswith("ERROR:"):
                parts = output.split(":", 2)
                return FuzzCase(
                    input_repr=input_repr,
                    category=category,
                    passed=False,
                    error_type=parts[1] if len(parts) > 1 else "Error",
                    error=parts[2] if len(parts) > 2 else output,
                )
            else:
                return FuzzCase(
                    input_repr=input_repr,
                    category=category,
                    passed=False,
                    error_type="UnexpectedOutput",
                    error=stderr[:200] if stderr else output[:200],
                )

        except subprocess.TimeoutExpired:
            try:
                os.unlink(script_path)
            except:
                pass
            return FuzzCase(
                input_repr=input_repr,
                category=category,
                passed=False,
                error_type="Timeout",
                error=f"Execution exceeded {self.timeout}s",
            )
        except Exception as e:
            return FuzzCase(
                input_repr=input_repr,
                category=category,
                passed=False,
                error_type=type(e).__name__,
                error=str(e)[:200],
            )

    def _generate_suggestions(self, cases: List[FuzzCase],
                              func_name: str) -> List[str]:
        """Generate fix suggestions based on failure patterns."""
        suggestions = []
        failure_types = set()

        for c in cases:
            if not c.passed:
                failure_types.add((c.error_type, c.category))

        # Pattern-based suggestions
        error_types = {et for et, _ in failure_types}
        categories = {cat for _, cat in failure_types}

        if "TypeError" in error_types:
            if "null" in categories or "null_value" in categories:
                suggestions.append(f"Add None check: if argument is None, raise ValueError or return default")
            if "mixed_types" in categories:
                suggestions.append(f"Add type validation: ensure all elements are the same type")

        if "ValueError" in error_types:
            suggestions.append(f"Validate input values before processing")

        if "IndexError" in error_types:
            if "empty" in categories:
                suggestions.append(f"Handle empty input: check len() > 0 before accessing elements")

        if "ZeroDivisionError" in error_types:
            suggestions.append(f"Check for zero before division")

        if "Timeout" in error_types:
            if "very_large" in categories or "overflow" in categories:
                suggestions.append(f"Add input size limit or optimize for large inputs")

        if "nan" in categories:
            suggestions.append(f"Handle NaN values: use math.isnan() to filter or reject")

        if "infinity" in categories:
            suggestions.append(f"Handle infinity: use math.isinf() to validate numeric inputs")

        if not suggestions and failure_types:
            suggestions.append(f"Add comprehensive input validation at the start of {func_name}()")

        return suggestions

    def stats(self) -> dict:
        return dict(self._stats)

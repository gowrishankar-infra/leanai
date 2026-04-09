"""
LeanAI Phase 7c — Test-Driven Auto-Fix Loop
Given a failing test, generates code until the test passes.

Flow:
  1. User provides test code (or test description)
  2. LeanAI generates implementation code
  3. Runs the test
  4. If FAIL → reads error, generates fix, reruns
  5. Loops until GREEN or max attempts
  6. Returns verified-working code

This is how LeanAI matches cloud AI quality with a 7B model:
the output is proven correct regardless of model capability.
"""

import os
import sys
import time
import re
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple


@dataclass
class TDDConfig:
    """Configuration for the TDD loop."""
    max_attempts: int = 5
    timeout_seconds: int = 30
    verbose: bool = False
    cleanup: bool = True  # clean up temp files after


@dataclass
class TDDAttempt:
    """Record of one attempt in the TDD loop."""
    attempt: int
    code: str
    test_passed: bool
    error: str = ""
    output: str = ""
    time_ms: float = 0


@dataclass
class TDDResult:
    """Result of a complete TDD loop."""
    success: bool
    implementation: str          # the final working code (or best attempt)
    test_code: str               # the test that was provided
    module_name: str             # the module name generated
    attempts: List[TDDAttempt] = field(default_factory=list)
    total_time_ms: float = 0
    final_error: str = ""

    @property
    def num_attempts(self) -> int:
        return len(self.attempts)

    def summary(self) -> str:
        status = "PASSED" if self.success else "FAILED"
        lines = [
            f"TDD Loop: {status} after {self.num_attempts} attempt(s)",
            f"Module: {self.module_name}",
            f"Time: {self.total_time_ms:.0f}ms",
        ]
        for a in self.attempts:
            icon = "●" if a.test_passed else "✗"
            err_preview = f" — {a.error[:80]}" if a.error else ""
            lines.append(f"  {icon} Attempt {a.attempt}: {a.time_ms:.0f}ms{err_preview}")
        if self.final_error:
            lines.append(f"Final error: {self.final_error[:200]}")
        return "\n".join(lines)


# ── Prompt templates ──────────────────────────────────────────────

IMPL_SYSTEM = """You are an expert Python programmer. Write implementation code that passes the given tests.
Rules:
1. Output ONLY the Python code. No markdown fences. No explanation.
2. Include all necessary imports.
3. Match function signatures exactly as the tests expect.
4. Handle edge cases the tests check for.
5. Make the code complete and correct."""

IMPL_USER = """Write Python code for module '{module_name}' that passes these tests:

{test_code}

Output ONLY the implementation code for {module_name}.py:"""

FIX_SYSTEM = """You are an expert Python debugger. Fix the code so the tests pass.
Rules:
1. Output ONLY the fixed code. No markdown. No explanation.
2. Keep the same function signatures.
3. Fix the specific error shown."""

FIX_USER = """The tests failed. Fix the implementation.

Module: {module_name}.py
Current code:
{current_code}

Test error:
{error}

Output ONLY the fixed complete code for {module_name}.py:"""


class TDDLoop:
    """
    Test-Driven Development auto-fix loop.
    
    Usage:
        tdd = TDDLoop(model_fn=my_model)
        
        test_code = '''
        from calculator import add, subtract
        
        def test_add():
            assert add(2, 3) == 5
            assert add(-1, 1) == 0
        
        def test_subtract():
            assert subtract(10, 3) == 7
        '''
        
        result = tdd.run(test_code, module_name="calculator")
        if result.success:
            print(result.implementation)  # working code!
    """

    def __init__(
        self,
        model_fn: Optional[Callable] = None,
        config: Optional[TDDConfig] = None,
    ):
        """
        Args:
            model_fn: function(system_prompt, user_prompt) -> str
        """
        self.model_fn = model_fn
        self.config = config or TDDConfig()

    def _call_model(self, system: str, user: str) -> str:
        if self.model_fn is None:
            raise RuntimeError("No model_fn provided to TDDLoop")
        return self.model_fn(system, user)

    def _clean_code(self, raw: str) -> str:
        """Strip markdown fences and clean model output."""
        text = raw.strip()
        text = text.replace("```python", "").replace("```py", "").replace("```", "")
        lines = text.split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    def _extract_module_name(self, test_code: str) -> str:
        """Try to extract the module name from test imports."""
        # Look for "from X import" or "import X"
        match = re.search(r"from\s+(\w+)\s+import", test_code)
        if match:
            return match.group(1)
        match = re.search(r"import\s+(\w+)", test_code)
        if match:
            return match.group(1)
        return "solution"

    def _run_tests(self, workspace: str, test_file: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """Run pytest on the test file. Returns (passed, stdout, stderr)."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "--no-header"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            passed = result.returncode == 0
            return passed, stdout, stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Tests timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)

    def _extract_error(self, stdout: str, stderr: str) -> str:
        """Extract the meaningful error from test output."""
        # Combine outputs
        output = stdout + "\n" + stderr

        # Look for FAILED lines
        failed_lines = [l for l in output.split("\n") if "FAILED" in l or "Error" in l or "assert" in l.lower()]
        if failed_lines:
            return "\n".join(failed_lines[:10])

        # Look for traceback
        if "Traceback" in output:
            lines = output.split("\n")
            tb_start = None
            for i, line in enumerate(lines):
                if "Traceback" in line:
                    tb_start = i
            if tb_start is not None:
                return "\n".join(lines[tb_start:tb_start + 15])

        # Fallback: last 10 lines
        lines = output.strip().split("\n")
        return "\n".join(lines[-10:])

    def _generate_implementation(self, test_code: str, module_name: str) -> str:
        """Generate initial implementation from tests."""
        raw = self._call_model(
            IMPL_SYSTEM,
            IMPL_USER.format(module_name=module_name, test_code=test_code),
        )
        return self._clean_code(raw)

    def _generate_fix(self, module_name: str, current_code: str, error: str) -> str:
        """Generate a fix for failing code."""
        raw = self._call_model(
            FIX_SYSTEM,
            FIX_USER.format(
                module_name=module_name,
                current_code=current_code,
                error=error[:2000],
            ),
        )
        return self._clean_code(raw)

    def run(self, test_code: str, module_name: Optional[str] = None) -> TDDResult:
        """
        Run the TDD loop: generate code until tests pass.
        
        Args:
            test_code: the pytest test code
            module_name: name of the module to generate (auto-detected if None)
        
        Returns:
            TDDResult with the implementation and attempt history
        """
        if module_name is None:
            module_name = self._extract_module_name(test_code)

        total_start = time.time()
        workspace = tempfile.mkdtemp(prefix="leanai_tdd_")
        test_file = os.path.join(workspace, f"test_{module_name}.py")
        impl_file = os.path.join(workspace, f"{module_name}.py")
        attempts = []
        current_code = ""
        success = False
        final_error = ""

        try:
            # Write the test file
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(test_code)

            # Attempt 1: Generate initial implementation
            if self.config.verbose:
                print(f"[TDD] Generating {module_name}.py from tests...", flush=True)

            current_code = self._generate_implementation(test_code, module_name)

            for attempt_num in range(1, self.config.max_attempts + 1):
                attempt_start = time.time()

                # Write the implementation
                with open(impl_file, "w", encoding="utf-8") as f:
                    f.write(current_code)

                # Run tests
                passed, stdout, stderr = self._run_tests(
                    workspace, test_file, self.config.timeout_seconds
                )
                elapsed = (time.time() - attempt_start) * 1000

                error = "" if passed else self._extract_error(stdout, stderr)
                output = stdout if passed else ""

                attempt = TDDAttempt(
                    attempt=attempt_num,
                    code=current_code,
                    test_passed=passed,
                    error=error,
                    output=output,
                    time_ms=elapsed,
                )
                attempts.append(attempt)

                if self.config.verbose:
                    icon = "●" if passed else "✗"
                    print(f"  {icon} Attempt {attempt_num}/{self.config.max_attempts}: "
                          f"{'PASSED' if passed else 'FAILED'} ({elapsed:.0f}ms)", flush=True)

                if passed:
                    success = True
                    break

                # Generate fix for next attempt
                if attempt_num < self.config.max_attempts:
                    if self.config.verbose:
                        print(f"  [TDD] Generating fix...", flush=True)
                    fixed = self._generate_fix(module_name, current_code, error)
                    if fixed and fixed != current_code:
                        current_code = fixed
                    else:
                        # Model couldn't generate a different fix — try regenerating from scratch
                        current_code = self._generate_implementation(test_code, module_name)

                final_error = error

        finally:
            if self.config.cleanup:
                try:
                    shutil.rmtree(workspace, ignore_errors=True)
                except Exception:
                    pass

        total_ms = (time.time() - total_start) * 1000

        return TDDResult(
            success=success,
            implementation=current_code,
            test_code=test_code,
            module_name=module_name,
            attempts=attempts,
            total_time_ms=total_ms,
            final_error="" if success else final_error,
        )

    def run_with_description(self, description: str, module_name: str = "solution") -> TDDResult:
        """
        Generate both tests and implementation from a natural language description.
        First generates tests, then runs the TDD loop.
        
        Args:
            description: what the code should do
            module_name: name for the module
        """
        # Step 1: Generate tests from description
        test_prompt = f"""Write pytest tests for this module:

Module name: {module_name}
Description: {description}

Write at least 5 test cases covering normal cases, edge cases, and error cases.
Use 'from {module_name} import ...' for imports.
Output ONLY the test code:"""

        raw_tests = self._call_model(
            "You are an expert tester. Write comprehensive pytest tests. Output ONLY code, no markdown.",
            test_prompt,
        )
        test_code = self._clean_code(raw_tests)

        if not test_code.strip():
            return TDDResult(
                success=False,
                implementation="",
                test_code="",
                module_name=module_name,
                final_error="Failed to generate tests from description",
            )

        # Step 2: Run TDD loop with generated tests
        return self.run(test_code, module_name)

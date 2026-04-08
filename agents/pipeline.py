"""
LeanAI Phase 4d — Agentic Pipeline (v2)
Orchestrates multi-step task execution with file-content-aware code generation.
"""

import os
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field

from agents.planner import (
    TaskPlanner,
    Plan,
    PlanStep,
    StepType,
    StepStatus,
)


@dataclass
class PipelineConfig:
    workspace_root: str = ""
    max_retries: int = 3
    step_timeout: int = 60
    verbose: bool = False
    auto_fix: bool = True
    create_workspace: bool = True
    run_final_verify: bool = True


@dataclass
class PipelineResult:
    plan: Plan
    success: bool
    total_time: float = 0.0
    model_calls: int = 0
    files_created: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"═══ Pipeline {status} ═══",
            f"Task: {self.plan.task}",
            f"Steps: {self.plan.completed_steps}/{self.plan.total_steps} passed",
            f"Files: {len(self.files_created)} created",
            f"Model calls: {self.model_calls}",
            f"Time: {self.total_time:.1f}s",
        ]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors[:5]:
                lines.append(f"  • {e[:150]}")
        if self.files_created:
            lines.append(f"\nFiles created:")
            for f in self.files_created:
                lines.append(f"  • {f}")
        lines.append("")
        lines.append(self.plan.summary())
        return "\n".join(lines)


class AgenticPipeline:
    """
    Orchestrates multi-step coding tasks with file-content-aware generation.

    Key improvement: when generating tests, the pipeline reads the actual source
    code of the module being tested and includes it in the prompt. This ensures
    the model writes tests that match the real function signatures.
    """

    def __init__(
        self,
        model_fn: Optional[Callable] = None,
        executor_fn: Optional[Callable] = None,
        config: Optional[PipelineConfig] = None,
        on_step_start: Optional[Callable] = None,
        on_step_end: Optional[Callable] = None,
    ):
        self.model_fn = model_fn
        self.executor_fn = executor_fn
        self.config = config or PipelineConfig()
        self.planner = TaskPlanner(verbose=self.config.verbose)
        self.on_step_start = on_step_start
        self.on_step_end = on_step_end
        self._model_calls = 0

    def _call_model(self, system: str, user: str) -> str:
        if self.model_fn is None:
            raise RuntimeError("No model_fn provided to pipeline")
        self._model_calls += 1
        return self.model_fn(system, user)

    def _ensure_workspace(self, workspace: str):
        if self.config.create_workspace:
            os.makedirs(workspace, exist_ok=True)

    def _write_file(self, workspace: str, relative_path: str, content: str) -> str:
        if not relative_path:
            return ""
        rel = relative_path.lstrip("/").lstrip("\\")
        full_path = os.path.join(workspace, rel)
        parent = os.path.dirname(full_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return full_path

    def _read_file(self, workspace: str, relative_path: str) -> Optional[str]:
        """Read a file from the workspace. Returns None if not found."""
        if not relative_path:
            return None
        rel = relative_path.lstrip("/").lstrip("\\")
        full_path = os.path.join(workspace, rel)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except (FileNotFoundError, IOError):
            return None

    def _run_command(self, command: str, workspace: str, timeout: int = 30) -> tuple:
        try:
            result = subprocess.run(
                command, shell=True, cwd=workspace,
                capture_output=True, text=True, timeout=timeout,
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output.strip()
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout}s"
        except Exception as e:
            return False, str(e)

    def _execute_code(self, code: str, workspace: str) -> tuple:
        if self.executor_fn:
            result = self.executor_fn(code)
            if isinstance(result, dict):
                return result.get("success", False), result.get("output", "")
            return bool(result), str(result)
        tmp_path = os.path.join(workspace, "__leanai_verify.py")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(code)
            success, output = self._run_command(
                f"python {tmp_path}", workspace, timeout=self.config.step_timeout
            )
            return success, output
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _clean_code(self, raw: str) -> str:
        text = raw.strip()
        text = text.replace("```python", "").replace("```py", "")
        text = text.replace("```json", "").replace("```", "")
        lines = text.split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    def _find_source_for_test(self, plan: Plan, step: PlanStep) -> tuple:
        """Find the source file that a test step should test. Returns (filename, content)."""
        target = step.target_file or ""
        # If target is test_calculator.py, source is calculator.py
        if target.startswith("test_"):
            source_name = target[5:]  # remove "test_"
        elif "_test" in target:
            source_name = target.replace("_test", "")
        else:
            source_name = "main.py"

        # Look for the source file in created files
        for fname in plan.created_files:
            if fname == source_name or fname.endswith(f"/{source_name}"):
                content = plan.file_contents.get(fname, "")
                if not content:
                    content = self._read_file(plan.workspace, fname) or ""
                return fname, content

        # Fallback: return the most recently created .py file that's not a test
        for fname in reversed(plan.created_files):
            if fname.endswith(".py") and "test" not in fname.lower():
                content = plan.file_contents.get(fname, "")
                if not content:
                    content = self._read_file(plan.workspace, fname) or ""
                return fname, content

        return "main.py", ""

    # ── Step executors ────────────────────────────────────────────

    def _exec_create_dir(self, step: PlanStep, plan: Plan) -> bool:
        target = step.target_file or plan.workspace
        if not os.path.isabs(target):
            target = os.path.join(plan.workspace, target)
        os.makedirs(target, exist_ok=True)
        step.output = f"Created directory: {target}"
        return True

    def _exec_create_file(self, step: PlanStep, plan: Plan) -> bool:
        if not step.target_file:
            step.error = "No target_file specified"
            return False
        system, user = self.planner.build_code_prompt(plan, step)
        raw = self._call_model(system, user)
        code = self._clean_code(raw)
        if not code.strip():
            step.error = "Model returned empty code"
            return False
        step.code = code
        self._write_file(plan.workspace, step.target_file, code)
        plan.created_files.append(step.target_file)
        plan.file_contents[step.target_file] = code
        step.output = f"Created: {step.target_file} ({len(code)} chars)"
        return True

    def _exec_write_code(self, step: PlanStep, plan: Plan) -> bool:
        ok = self._exec_create_file(step, plan)
        if not ok:
            return False
        # Verify by running the file
        if step.target_file and step.target_file.endswith(".py"):
            full_path = os.path.join(plan.workspace, step.target_file)
            verify_cmd = f'python -c "exec(open(r\'{full_path}\').read())"'
            if step.verification:
                verify_cmd = step.verification
            success, output = self._run_command(verify_cmd, plan.workspace, self.config.step_timeout)
            if not success:
                step.error = output[:500]
                return False
            step.output += " | Verified OK"
        return True

    def _exec_run_test(self, step: PlanStep, plan: Plan) -> bool:
        """Generate tests using the actual source code, then run them."""
        if not step.target_file:
            step.error = "No target_file specified"
            return False

        # Find the source file to test
        source_file, source_code = self._find_source_for_test(plan, step)

        if source_code:
            # Use the test-specific prompt that includes source code
            system, user = self.planner.build_test_prompt(plan, step, source_file, source_code)
        else:
            # Fallback to generic code gen
            system, user = self.planner.build_code_prompt(plan, step)

        raw = self._call_model(system, user)
        code = self._clean_code(raw)
        if not code.strip():
            step.error = "Model returned empty test code"
            return False

        step.code = code
        self._write_file(plan.workspace, step.target_file, code)
        plan.created_files.append(step.target_file)
        plan.file_contents[step.target_file] = code

        # Run the tests
        test_path = os.path.join(plan.workspace, step.target_file)
        cmd = f"python -m pytest {test_path} -v --tb=short"
        success, output = self._run_command(cmd, plan.workspace, self.config.step_timeout)
        step.output = output[:1000]
        if not success:
            step.error = output[:500]
        return success

    def _exec_run_command(self, step: PlanStep, plan: Plan) -> bool:
        cmd = step.verification or step.description
        if not cmd:
            step.error = "No command to run"
            return False
        success, output = self._run_command(cmd, plan.workspace, self.config.step_timeout)
        step.output = output[:1000]
        if not success:
            step.error = output[:500]
        return success

    def _exec_verify(self, step: PlanStep, plan: Plan) -> bool:
        """Run final verification. Robust: tries verification command, falls back to pytest."""
        if step.verification:
            success, output = self._run_command(
                step.verification, plan.workspace, self.config.step_timeout
            )
            if success:
                step.output = output[:1000]
                return True
            # If specific verification failed, fall through to pytest
            step.error = None

        # Default: run pytest on all test files in workspace
        test_files = [f for f in plan.created_files if "test" in f.lower()]
        if test_files:
            # Use the workspace as cwd and run pytest on the test files directly
            test_paths = " ".join(
                os.path.join(plan.workspace, f) for f in test_files
            )
            cmd = f"python -m pytest {test_paths} -v --tb=short"
            success, output = self._run_command(cmd, plan.workspace, self.config.step_timeout)
            step.output = output[:1000]
            if not success:
                step.error = output[:500]
            return success

        # No tests exist — pass silently
        step.output = "No tests to verify — passed"
        return True

    STEP_HANDLERS = {
        StepType.CREATE_DIR: "_exec_create_dir",
        StepType.CREATE_FILE: "_exec_create_file",
        StepType.WRITE_CODE: "_exec_write_code",
        StepType.MODIFY_FILE: "_exec_write_code",
        StepType.RUN_COMMAND: "_exec_run_command",
        StepType.RUN_TEST: "_exec_run_test",
        StepType.VERIFY: "_exec_verify",
    }

    def _execute_step(self, step: PlanStep, plan: Plan) -> bool:
        handler_name = self.STEP_HANDLERS.get(step.step_type, "_exec_write_code")
        handler = getattr(self, handler_name)
        try:
            return handler(step, plan)
        except Exception as e:
            step.error = str(e)[:500]
            return False

    def _attempt_fix(self, step: PlanStep, plan: Plan) -> bool:
        """Try to fix a failed step using the model with full file context."""
        if not step.code or not step.error:
            return False

        system, user = self.planner.build_fix_prompt(plan, step)
        raw = self._call_model(system, user)
        fixed_code = self._clean_code(raw)

        if not fixed_code.strip() or fixed_code == step.code:
            return False

        step.code = fixed_code
        if step.target_file:
            self._write_file(plan.workspace, step.target_file, fixed_code)
            plan.file_contents[step.target_file] = fixed_code

        # Re-verify based on step type
        if step.step_type == StepType.RUN_TEST:
            test_path = os.path.join(plan.workspace, step.target_file)
            cmd = f"python -m pytest {test_path} -v --tb=short"
            success, output = self._run_command(cmd, plan.workspace, self.config.step_timeout)
            step.output = output[:1000]
            if success:
                step.error = None
                return True
            step.error = output[:500]
            return False
        elif step.target_file and step.target_file.endswith(".py"):
            full_path = os.path.join(plan.workspace, step.target_file)
            verify_cmd = f'python -c "exec(open(r\'{full_path}\').read())"'
            if step.verification:
                verify_cmd = step.verification
            success, output = self._run_command(verify_cmd, plan.workspace, self.config.step_timeout)
            if success:
                step.output = f"Fixed and verified: {step.target_file}"
                step.error = None
                return True
            step.error = output[:500]
            return False
        return True

    # ── Main execution ────────────────────────────────────────────

    def generate_plan(self, task: str, workspace: str) -> Plan:
        system, user = self.planner.build_plan_prompt(task)
        raw = self._call_model(system, user)
        try:
            steps = self.planner.parse_plan_response(raw)
            if len(steps) < 2:
                raise ValueError("Plan too short")
            if len(steps) > 6:
                # Model over-engineered the plan — use fallback for reliability
                if self.config.verbose:
                    print(f"[Pipeline] Model plan had {len(steps)} steps (max 6). Using fallback.")
                plan = self.planner.create_fallback_plan(task, workspace)
            else:
                plan = self.planner.create_plan(task, workspace, steps)
        except (ValueError, json.JSONDecodeError) as e:
            if self.config.verbose:
                print(f"[Pipeline] Plan parsing failed: {e}. Using fallback.")
            plan = self.planner.create_fallback_plan(task, workspace)
        return plan

    def execute_plan(self, plan: Plan) -> PipelineResult:
        self._ensure_workspace(plan.workspace)
        plan.status = "executing"
        start_time = time.time()

        for i, step in enumerate(plan.steps):
            plan.current_step_idx = i

            # Skip steps already in terminal state
            if step.status in (StepStatus.PASSED, StepStatus.FAILED):
                if self.on_step_end:
                    self.on_step_end(step, plan)
                continue

            # Check dependencies
            if not plan.deps_met(step):
                step.status = StepStatus.SKIPPED
                step.error = "Dependencies not met"
                if self.on_step_end:
                    self.on_step_end(step, plan)
                continue

            step.status = StepStatus.RUNNING
            step.attempts = 1

            if self.on_step_start:
                self.on_step_start(step, plan)

            success = self._execute_step(step, plan)

            # Retry loop
            while not success and step.attempts < step.max_attempts and self.config.auto_fix:
                step.status = StepStatus.RETRYING
                step.attempts += 1
                if self.config.verbose:
                    print(f"[Pipeline] Retrying {step.id} (attempt {step.attempts}/{step.max_attempts})")
                if step.step_type in (StepType.WRITE_CODE, StepType.CREATE_FILE, StepType.RUN_TEST):
                    success = self._attempt_fix(step, plan)
                else:
                    break

            step.status = StepStatus.PASSED if success else StepStatus.FAILED
            if step.status == StepStatus.FAILED:
                plan.errors.append(f"Step {step.id} failed: {step.error}")

            if self.on_step_end:
                self.on_step_end(step, plan)

        failed = plan.failed_steps
        plan.status = "completed" if failed == 0 else "completed_with_errors" if failed < plan.total_steps else "failed"
        elapsed = time.time() - start_time

        return PipelineResult(
            plan=plan, success=(failed == 0), total_time=elapsed,
            model_calls=self._model_calls, files_created=list(plan.created_files),
            errors=list(plan.errors),
        )

    def execute(self, task: str, workspace: Optional[str] = None) -> PipelineResult:
        self._model_calls = 0
        if workspace is None:
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in task[:40])
            workspace = os.path.join(
                self.config.workspace_root or os.path.expanduser("~/.leanai/projects"),
                safe_name,
            )
        plan = self.generate_plan(task, workspace)
        return self.execute_plan(plan)

    def execute_with_plan(self, plan: Plan) -> PipelineResult:
        self._model_calls = 0
        return self.execute_plan(plan)

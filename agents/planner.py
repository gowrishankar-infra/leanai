"""
LeanAI Phase 4d — Task Planner (v2)
Decomposes complex coding tasks into ordered, executable steps.
Optimized prompts for 7B local models.
"""

import json
import re
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum


class StepType(str, Enum):
    CREATE_DIR = "create_dir"
    CREATE_FILE = "create_file"
    WRITE_CODE = "write_code"
    MODIFY_FILE = "modify_file"
    RUN_COMMAND = "run_command"
    RUN_TEST = "run_test"
    VERIFY = "verify"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class PlanStep:
    id: str
    title: str
    description: str
    step_type: StepType
    target_file: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    code: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    verification: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["step_type"] = self.step_type.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlanStep":
        d["step_type"] = StepType(d.get("step_type", "write_code"))
        d["status"] = StepStatus(d.get("status", "pending"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Plan:
    task: str
    steps: List[PlanStep] = field(default_factory=list)
    workspace: str = ""
    status: str = "planning"
    current_step_idx: int = 0
    created_files: List[str] = field(default_factory=list)
    file_contents: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "workspace": self.workspace,
            "status": self.status,
            "current_step_idx": self.current_step_idx,
            "created_files": self.created_files,
            "errors": self.errors,
        }

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.PASSED)

    @property
    def failed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.FAILED)

    @property
    def progress_pct(self) -> float:
        if not self.steps:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def next_step(self) -> Optional[PlanStep]:
        for s in self.steps:
            if s.status == StepStatus.PENDING:
                return s
        return None

    def deps_met(self, step: PlanStep) -> bool:
        for dep_id in step.dependencies:
            dep = self.get_step(dep_id)
            if dep is None or dep.status != StepStatus.PASSED:
                return False
        return True

    def summary(self) -> str:
        lines = [f"Task: {self.task}", f"Status: {self.status}"]
        lines.append(f"Progress: {self.completed_steps}/{self.total_steps} ({self.progress_pct:.0f}%)")
        lines.append("")
        for s in self.steps:
            icon = {
                "pending": "○", "running": "◐", "passed": "●",
                "failed": "✗", "skipped": "◌", "retrying": "↻",
            }.get(s.status.value, "?")
            lines.append(f"  {icon} [{s.id}] {s.title} — {s.status.value}")
            if s.error:
                lines.append(f"       Error: {s.error[:120]}")
        if self.created_files:
            lines.append(f"\nFiles created: {len(self.created_files)}")
            for f in self.created_files:
                lines.append(f"  • {f}")
        return "\n".join(lines)


# ── Prompts optimized for 7B models ──────────────────────────────

PLAN_SYSTEM_PROMPT = """You break coding tasks into steps. Output ONLY a JSON array. No other text.

Example for "Build a todo app":
[{"id":"s0","title":"Create directory","step_type":"create_dir","target_file":"src","dependencies":[]},{"id":"s1","title":"Write todo module","step_type":"write_code","target_file":"todo.py","dependencies":["s0"],"description":"Functions for add, remove, list todos"},{"id":"s2","title":"Write tests","step_type":"run_test","target_file":"test_todo.py","dependencies":["s1"],"description":"Test all todo functions with pytest"},{"id":"s3","title":"Verify","step_type":"verify","dependencies":["s2"],"description":"Run all tests"}]

Rules:
- 4-6 steps MAXIMUM. Keep plans short.
- Start with create_dir. End with ONE verify step.
- NEVER create separate verify steps per function. ONE verify at the end.
- Step types: create_dir, write_code, run_test, verify.
- Put ALL tests in ONE test file. Do NOT split tests across files."""

PLAN_USER_TEMPLATE = """Task: {task}

Output ONLY the JSON array:"""


CODE_GEN_SYSTEM_PROMPT = """You are an expert programmer. Write complete, runnable Python code.
Rules:
1. Output ONLY code. No markdown fences, no explanation, no backticks.
2. Include all imports.
3. Add docstrings and type hints.
4. Make it complete and runnable."""

CODE_GEN_USER_TEMPLATE = """Project: {task}

{file_contents}

Now write: {target_file}
Description: {step_description}

Output ONLY the Python code:"""

TEST_GEN_SYSTEM_PROMPT = """You are an expert tester. Write pytest tests for the given code.
Rules:
1. Output ONLY code. No markdown fences, no explanation, no backticks.
2. Import directly from the module file (e.g. from calculator import add, subtract).
3. Use simple test functions, not classes.
4. Include at least 5 test cases.
5. Match imports EXACTLY to the function names defined in the source code."""

TEST_GEN_USER_TEMPLATE = """Write pytest tests for this code.

Source file: {source_file}
Source code:
{source_code}

Test file to create: {target_file}

Output ONLY the pytest code:"""

FIX_SYSTEM_PROMPT = """Fix the broken code. Output ONLY the fixed code. No markdown, no explanation."""

FIX_USER_TEMPLATE = """This code failed.

File: {target_file}
Error:
{error}

Code:
{code}

{related_files}

Output ONLY the fixed complete code:"""


class TaskPlanner:
    """Decomposes complex tasks into executable step plans."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _parse_steps_array(self, steps_data: list) -> List[PlanStep]:
        """Parse a raw JSON array into PlanStep objects."""
        if not isinstance(steps_data, list):
            raise ValueError("Expected a JSON array of steps")

        steps = []
        for i, sd in enumerate(steps_data):
            if not isinstance(sd, dict):
                continue
            step = PlanStep(
                id=sd.get("id", f"step_{i:02d}"),
                title=sd.get("title", f"Step {i + 1}"),
                description=sd.get("description", ""),
                step_type=StepType(sd.get("step_type", "write_code")),
                target_file=sd.get("target_file"),
                dependencies=sd.get("dependencies", []),
                verification=sd.get("verification"),
            )
            if not step.dependencies and i > 0:
                step.dependencies = [steps[i - 1].id]
            steps.append(step)

        return steps

    def parse_plan_response(self, response: str) -> List[PlanStep]:
        """Parse model response into PlanStep objects."""
        text = response.strip()

        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Find JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in response")

        json_str = text[start : end + 1]

        # Clean trailing commas
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*\]", "]", json_str)

        # Try parsing as-is first
        try:
            return self._parse_steps_array(json.loads(json_str))
        except json.JSONDecodeError:
            pass

        # Fallback: single quote replacement
        json_str_fixed = json_str.replace("'", '"')
        try:
            return self._parse_steps_array(json.loads(json_str_fixed))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    def build_plan_prompt(self, task: str) -> tuple:
        """Build the prompt pair for plan generation."""
        return (PLAN_SYSTEM_PROMPT, PLAN_USER_TEMPLATE.format(task=task))

    def build_code_prompt(self, plan: Plan, step: PlanStep) -> tuple:
        """Build prompt for code generation — includes actual file contents."""
        contents_parts = []
        for fname, content in plan.file_contents.items():
            truncated = content[:2000] if len(content) > 2000 else content
            contents_parts.append(f"=== {fname} ===\n{truncated}")
        file_contents = "\n\n".join(contents_parts) if contents_parts else "No files created yet."

        return (
            CODE_GEN_SYSTEM_PROMPT,
            CODE_GEN_USER_TEMPLATE.format(
                task=plan.task,
                file_contents=file_contents,
                step_description=step.description,
                target_file=step.target_file or "(no file)",
            ),
        )

    def build_test_prompt(self, plan: Plan, step: PlanStep, source_file: str, source_code: str) -> tuple:
        """Build prompt specifically for test generation — includes full source code."""
        return (
            TEST_GEN_SYSTEM_PROMPT,
            TEST_GEN_USER_TEMPLATE.format(
                source_file=source_file,
                source_code=source_code,
                target_file=step.target_file or "test_main.py",
            ),
        )

    def build_fix_prompt(self, plan: Plan, step: PlanStep) -> tuple:
        """Build prompt for fixing failed code — includes related file contents."""
        related = []
        for fname, content in plan.file_contents.items():
            if fname != step.target_file:
                truncated = content[:1500] if len(content) > 1500 else content
                related.append(f"=== {fname} ===\n{truncated}")

        related_str = "\n\n".join(related) if related else ""
        if related_str:
            related_str = f"Related files in project:\n{related_str}"

        return (
            FIX_SYSTEM_PROMPT,
            FIX_USER_TEMPLATE.format(
                target_file=step.target_file or "(no file)",
                error=step.error or "Unknown error",
                code=step.code or "",
                related_files=related_str,
            ),
        )

    def create_plan(self, task: str, workspace: str, steps: List[PlanStep]) -> Plan:
        return Plan(task=task, workspace=workspace, steps=steps, status="ready")

    def create_fallback_plan(self, task: str, workspace: str) -> Plan:
        """Create a smart fallback plan when model parsing fails."""
        words = task.lower().split()
        module = "main"
        for w in ["calculator", "api", "app", "server", "tool", "game", "bot", "cli"]:
            if w in words:
                module = w
                break

        steps = [
            PlanStep(
                id="step_00", title="Create project structure",
                description=f"Create directory structure for: {task}",
                step_type=StepType.CREATE_DIR, target_file=workspace,
            ),
            PlanStep(
                id="step_01", title="Write main code",
                description=f"Write the complete code for: {task}. Put all functions in one file.",
                step_type=StepType.WRITE_CODE, target_file=f"{module}.py",
                dependencies=["step_00"],
            ),
            PlanStep(
                id="step_02", title="Write tests",
                description=f"Write pytest tests for {module}.py. Import functions from {module}.",
                step_type=StepType.RUN_TEST, target_file=f"test_{module}.py",
                dependencies=["step_01"],
            ),
            PlanStep(
                id="step_03", title="Verify everything works",
                description="Run all tests and verify the project",
                step_type=StepType.VERIFY, dependencies=["step_02"],
            ),
        ]
        return Plan(task=task, workspace=workspace, steps=steps, status="ready")

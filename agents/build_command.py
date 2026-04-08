"""
LeanAI Phase 4d — Build Command Handler
Manages the /build command interaction: shows plan, progress, results.
"""

import os
import time
from typing import Optional, Callable

from agents.planner import Plan, PlanStep, StepStatus
from agents.pipeline import AgenticPipeline, PipelineConfig, PipelineResult


def step_icon(status: str) -> str:
    return {
        "pending": "○",
        "running": "◐",
        "passed": "●",
        "failed": "✗",
        "skipped": "◌",
        "retrying": "↻",
    }.get(status, "?")


class BuildHandler:
    """Handles /build command interactions."""

    def __init__(self, model_fn: Optional[Callable] = None, verbose: bool = False):
        self.model_fn = model_fn
        self.verbose = verbose
        self.last_result: Optional[PipelineResult] = None

    def _print_step_start(self, step: PlanStep, plan: Plan):
        pct = plan.progress_pct
        print(f"\n  ◐ [{step.id}] {step.title}... ", end="", flush=True)

    def _print_step_end(self, step: PlanStep, plan: Plan):
        icon = step_icon(step.status.value)
        extra = ""
        if step.attempts > 1:
            extra = f" ({step.attempts} attempts)"
        if step.status == StepStatus.FAILED:
            print(f"{icon} FAILED{extra}")
            if step.error:
                print(f"       Error: {step.error[:200]}")
        elif step.status == StepStatus.SKIPPED:
            print(f"{icon} skipped")
        else:
            print(f"{icon} done{extra}")

    def execute_build(self, task: str, workspace: Optional[str] = None) -> PipelineResult:
        """Run the full build pipeline with interactive output."""
        if not self.model_fn:
            print("[Build] Error: No model loaded. Run a query first to load the model.")
            return None

        # Workspace
        if workspace is None:
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in task[:40])
            workspace = os.path.join(
                os.path.expanduser("~/.leanai/projects"), safe
            )

        print(f"\n{'═' * 60}")
        print(f"  BUILD: {task}")
        print(f"  Workspace: {workspace}")
        print(f"{'═' * 60}")

        config = PipelineConfig(
            workspace_root=os.path.expanduser("~/.leanai/projects"),
            auto_fix=True,
            verbose=self.verbose,
        )

        pipeline = AgenticPipeline(
            model_fn=self.model_fn,
            config=config,
            on_step_start=self._print_step_start,
            on_step_end=self._print_step_end,
        )

        # Phase 1: Generate plan
        print("\n[Build] Generating plan...", flush=True)
        start = time.time()
        plan = pipeline.generate_plan(task, workspace)
        plan_time = time.time() - start
        print(f"[Build] Plan ready ({plan.total_steps} steps, {plan_time:.1f}s)")
        print()

        # Show plan
        for s in plan.steps:
            icon = step_icon(s.status.value)
            dep_str = f" (after {', '.join(s.dependencies)})" if s.dependencies else ""
            file_str = f" → {s.target_file}" if s.target_file else ""
            print(f"  {icon} [{s.id}] {s.title}{file_str}{dep_str}")
        print()

        # Phase 2: Execute
        print("[Build] Executing...", flush=True)
        result = pipeline.execute_plan(plan)
        self.last_result = result

        # Phase 3: Report
        print(f"\n{'═' * 60}")
        if result.success:
            print(f"  ● BUILD COMPLETE")
        else:
            print(f"  ✗ BUILD COMPLETED WITH ERRORS")
        print(f"  Steps: {plan.completed_steps}/{plan.total_steps} passed")
        print(f"  Files: {len(result.files_created)} created")
        print(f"  Model calls: {result.model_calls}")
        print(f"  Time: {result.total_time:.1f}s")

        if result.files_created:
            print(f"\n  Files:")
            for f in result.files_created:
                full = os.path.join(workspace, f)
                exists = os.path.exists(full)
                icon = "●" if exists else "✗"
                print(f"    {icon} {f}")

        if result.errors:
            print(f"\n  Errors:")
            for e in result.errors[:5]:
                print(f"    • {e[:200]}")

        print(f"  Workspace: {workspace}")
        print(f"{'═' * 60}")

        return result

    def show_last_result(self):
        """Show the result of the last build."""
        if not self.last_result:
            print("No build has been run yet. Use: /build <task>")
            return
        print(self.last_result.summary())

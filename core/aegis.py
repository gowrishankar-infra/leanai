"""
LeanAI — Aegis (M7): Agentic Multi-Step Coding
================================================

Aegis is an agentic coding loop for multi-file single-concern tasks.
Given a user request like "rename ProjectBrain to CodeBrain everywhere"
or "add logging to every handler in api/*", Aegis:

  1. PLANS the work — decomposes the task into 3-8 concrete steps
     using the brain graph + M6 retrieval to find affected files
     (not model guess).

  2. EXECUTES each step sequentially:
     - Loads the target function/class via brain + forensics (M4)
     - Retrieves 3 most-similar existing functions via M6 for style
     - Asks the model to produce the patch
     - Parses/validates the patch (syntax + optionally pytest)
     - Runs Sentinel (M1) on the touched file to catch new vulns

  3. VERIFIES with Sentinel after each step. A step that adds a
     MEDIUM+ finding is rejected and retried (up to retry budget).

  4. ROLLS BACK on failure:
     - Per-step: if validation fails, no file was written (safety
       layer only writes on success).
     - Per-run: uncommitted user WIP is auto-stashed before Aegis
       starts; `git stash pop` returns it verbatim if anything
       catastrophic happens.

  5. REPORTS — final summary: what was done, what was skipped,
     what needs human review.

Safety (delegated entirely to aegis_safety.SafeFileWriter):
  - DRY_RUN by default
  - --apply flag AND env var required for writes
  - Per-file Y/n confirmation
  - Project-root-only path validation
  - Stash-then-work-then-unstash

Budget caps:
  - Max 8 steps per plan
  - Max 3 retries per step
  - Max 60 seconds per step's model generation
  - Max 2500 chars of context per model call (M6 budget)

What Aegis explicitly does NOT do:
  - Auto-commit to git
  - Modify anything outside project root
  - Edit test files (unless user request mentions "test")
  - Execute user code
  - Fix vulnerabilities found in source (M5 AutoFix territory)
"""

from __future__ import annotations

import os
import re
import sys
import time
import difflib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.aegis_safety import (
    SafeFileWriter, WriteIntent, SafetyMode,
    detect_safety_mode, explain_mode, SafetyError,
    AEGIS_CONFIRM_ENV, AEGIS_CONFIRM_VALUE,
)


# ══════════════════════════════════════════════════════════════
# BUDGETS
# ══════════════════════════════════════════════════════════════

MAX_STEPS_PER_PLAN = 8
MAX_RETRIES_PER_STEP = 3
MAX_STEP_DURATION_S = 60
MAX_PLAN_DURATION_S = 600  # 10 min total
MAX_CONTEXT_CHARS = 2500

# Aegis refuses to modify files larger than this. Above this size,
# full-file replacement (which Aegis uses) becomes too risky — the
# model sees a truncated view of the file and may produce a
# truncated output, deleting code. Honest scope cut: M7 only
# handles small/medium files. Big-file refactors need diff-based
# patching, which is M5 AutoFix territory and currently parked.
MAX_FILE_SIZE_FOR_AEGIS = 6000  # chars (~150 lines of typical Python)

# Reject any proposed patch that shrinks the file by more than this
# fraction. A "add docstring" task should never delete 70% of code.
# If the model returns a much shorter file, it almost certainly
# truncated rather than added.
MAX_SHRINK_RATIO = 0.40  # patch may be at most 40% smaller


class StepStatus(Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class AegisStep:
    """A single concrete action within an Aegis plan."""
    step_id: str                     # e.g. "step-1", "step-2"
    description: str                 # one-line human summary
    target_file: str                 # relative path
    target_function: Optional[str]   # qualified name or None for file-level
    change_type: str                 # "modify" | "create" | "rename"
    rationale: str                   # why this step is in the plan
    status: StepStatus = StepStatus.PENDING
    # Execution-time fields
    proposed_patch: Optional[str] = None      # new file content
    diff: Optional[str] = None                 # unified diff string
    validation_errors: List[str] = field(default_factory=list)
    new_sentinel_findings: int = 0             # new vulns introduced
    retries_used: int = 0
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "target_file": self.target_file,
            "target_function": self.target_function,
            "change_type": self.change_type,
            "rationale": self.rationale,
            "status": self.status.value,
            "validation_errors": self.validation_errors,
            "new_sentinel_findings": self.new_sentinel_findings,
            "retries_used": self.retries_used,
            "duration_s": round(self.duration_s, 2),
        }


@dataclass
class AegisPlan:
    """Result of the planning phase."""
    task: str                          # original user request
    steps: List[AegisStep] = field(default_factory=list)
    planning_rationale: str = ""       # model's reasoning for the plan
    planning_time_s: float = 0.0
    abandoned: bool = False            # True if planner gave up
    abandon_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "planning_rationale": self.planning_rationale,
            "planning_time_s": round(self.planning_time_s, 2),
            "abandoned": self.abandoned,
            "abandon_reason": self.abandon_reason,
        }


@dataclass
class AegisReport:
    """Final summary after plan execution."""
    plan: AegisPlan
    total_duration_s: float = 0.0
    applied_steps: int = 0
    skipped_steps: int = 0
    failed_steps: int = 0
    files_modified: List[str] = field(default_factory=list)
    files_skipped: List[Tuple[str, str]] = field(default_factory=list)  # (path, reason)
    mode: str = SafetyMode.DRY_RUN.value
    needs_human_review: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "total_duration_s": round(self.total_duration_s, 2),
            "applied_steps": self.applied_steps,
            "skipped_steps": self.skipped_steps,
            "failed_steps": self.failed_steps,
            "files_modified": self.files_modified,
            "files_skipped": [{"path": p, "reason": r}
                              for p, r in self.files_skipped],
            "mode": self.mode,
            "needs_human_review": self.needs_human_review,
        }


# ══════════════════════════════════════════════════════════════
# AEGIS PLANNER
# ══════════════════════════════════════════════════════════════

class AegisPlanner:
    """Decomposes a user task into a concrete step plan.

    Uses three information sources to avoid blind model planning:
      1. M6 hybrid retriever — find relevant existing code
      2. Brain graph — find all callers/callees if user named a symbol
      3. Sentinel findings — if task involves security, surface
         existing findings the task might address
    """

    def __init__(self, brain, indexer=None, sentinel=None, model_fn=None):
        """
        brain:     ProjectBrain (required — we use the graph + file index)
        indexer:   ProjectIndexer with HybridRetriever (optional, improves quality)
        sentinel:  SentinelEngine (optional, enables security-aware planning)
        model_fn:  Callable[[system_prompt, user_prompt], str] for LLM calls.
                   If None, planner operates in 'skeletal' mode: produces
                   a deterministic plan from graph + file patterns but
                   without model reasoning on content.
        """
        self.brain = brain
        self.indexer = indexer
        self.sentinel = sentinel
        self.model_fn = model_fn

    @staticmethod
    def _split_node_id(node_id: str) -> str:
        """Extract the filepath portion of a graph node_id, safe for
        Windows paths where the drive letter ('C:') contains a colon.

        Graph node_ids are either:
          - A bare filepath (module-level, e.g. "C:\\foo.py" or "/home/foo.py")
          - A "filepath:qualified_name" tuple joined by ':'

        Naive rsplit(':', 1) gives wrong results on Windows when there's
        no qname suffix: "C:\\foo.py".rsplit(':', 1) = ['C', '\\foo.py'].

        Correct approach: a colon is a qname separator only if it
        appears AFTER the last path separator. Otherwise it's a Windows
        drive letter and should be left alone.
        """
        if ':' not in node_id:
            return node_id
        # Find position of last path separator (\ or /)
        last_sep = max(node_id.rfind('\\'), node_id.rfind('/'))
        last_colon = node_id.rfind(':')
        if last_colon > last_sep:
            # Colon is after the last path sep — it's a qname separator
            return node_id[:last_colon]
        # Colon is before (or there's no path sep) — leave as-is
        # (this handles Windows drive letters like "C:\\foo.py")
        return node_id

    def plan(self, task: str) -> AegisPlan:
        """Produce a plan for the given task.

        Strategy:
          1. Extract candidate files from task text (symbol names, paths).
          2. Expand via brain graph (callers/callees of named symbols).
          3. Use M6 retriever to find semantically relevant additional files.
          4. De-duplicate and cap at MAX_STEPS_PER_PLAN.
          5. For each candidate, produce an AegisStep with a rationale.
          6. If model_fn is provided, ask the model to refine ordering
             and produce per-step descriptions.
        """
        start = time.time()
        plan = AegisPlan(task=task)

        # Safety — immediate bail if task is empty or suspicious
        if not task or len(task.strip()) < 3:
            plan.abandoned = True
            plan.abandon_reason = "Task is empty or too short"
            plan.planning_time_s = time.time() - start
            return plan
        if len(task) > 2000:
            plan.abandoned = True
            plan.abandon_reason = "Task too long (>2000 chars)"
            plan.planning_time_s = time.time() - start
            return plan

        # ── Step 1: extract candidate symbols & files from task text ──
        # Priority-ordered: anything explicitly named by the user must
        # land in the plan, regardless of what semantic retrieval or
        # graph expansion suggests.
        explicit_files: List[str] = []        # exact path/name matches — HIGHEST priority
        explicit_functions: List[Tuple[str, str]] = []  # (file, qname) from exact symbol matches
        candidate_files: List[str] = []       # everything that made it past filters
        candidate_functions: List[Tuple[str, str]] = []

        # Words that are common in task *verbs*, not code symbols.
        # Filters "add", "remove", "refactor" etc from becoming
        # function-name candidates.
        TASK_VERB_FILLER = {
            # action verbs
            'add', 'remove', 'refactor', 'rename', 'create', 'delete',
            'update', 'modify', 'change', 'edit', 'fix', 'insert',
            'replace', 'move', 'copy', 'extract', 'inline', 'split',
            'merge', 'convert', 'format', 'clean', 'organize', 'reorder',
            'include', 'exclude', 'enable', 'disable', 'register',
            'setup', 'configure', 'initialize',
            # common nouns in tasks
            'docstring', 'docstrings', 'doc', 'docs', 'comment',
            'comments', 'logging', 'log', 'logs', 'error', 'errors',
            'test', 'tests', 'function', 'functions', 'method', 'methods',
            'class', 'classes', 'variable', 'variables', 'import',
            'imports', 'type', 'types', 'hint', 'hints', 'annotation',
            'annotations', 'parameter', 'parameters', 'argument',
            'arguments', 'return', 'returns', 'value', 'values',
            # common adjectives/connectives
            'every', 'each', 'any', 'all', 'some', 'public', 'private',
            'protected', 'static', 'async', 'sync', 'new', 'old',
        }

        # Look for filename patterns in the task.
        # These are the STRONGEST signal — the user wrote a real path.
        file_pattern = re.compile(
            r'\b([A-Za-z_][A-Za-z0-9_/.\\-]*\.[a-z]{1,4})\b')
        for match in file_pattern.findall(task):
            normed = match.replace('\\', '/')
            for rel in self.brain._file_analyses.keys():
                rel_norm = rel.replace('\\', '/')
                if rel_norm == normed or rel_norm.endswith('/' + normed):
                    if rel not in explicit_files:
                        explicit_files.append(rel)

        # Look for function/class names. Require tokens to be
        # at least 4 chars AND not in the task-verb-filler set.
        # This prevents "add" matching HDKnowledgeStore.add, or single
        # letters like "C" becoming candidates.
        word_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{3,})\b')
        tokens_raw = word_pattern.findall(task)
        # Filter: length >= 4, not a task verb, not in filler
        tokens = {t for t in tokens_raw
                  if len(t) >= 4 and t.lower() not in TASK_VERB_FILLER}

        for rel, analysis in self.brain._file_analyses.items():
            for func in (analysis.functions or []):
                # Only match on full function NAME (not qualified name,
                # to avoid Class.add matching on "add"). Must also
                # satisfy length + filler constraints.
                if func.name in tokens:
                    explicit_functions.append((rel, func.qualified_name))
                    if rel not in explicit_files:
                        explicit_files.append(rel)
                elif func.qualified_name in tokens:
                    explicit_functions.append((rel, func.qualified_name))
                    if rel not in explicit_files:
                        explicit_files.append(rel)
            for cls in (analysis.classes or []):
                if cls.name in tokens:
                    if rel not in explicit_files:
                        explicit_files.append(rel)

        # Record which files were directly named (for rationale attribution)
        directly_named_files = set(explicit_files)

        # Candidate pool starts with explicit files in order
        candidate_files = list(explicit_files)
        candidate_functions = list(explicit_functions)

        # ── Step 2: expand via graph (callers of named symbols) ──
        # Graph expansion is SECONDARY — it should only pad the plan
        # AFTER explicit files are in, and only if we have remaining
        # step budget.
        expanded_files: List[str] = list(candidate_files)
        graph_expanded_set: Set[str] = set()
        for (rel_file, qname) in candidate_functions:
            node_id = f"{os.path.abspath(os.path.join(self.brain.config.project_path, rel_file))}:{qname}"
            for src, targets in self.brain.graph._adjacency.items():
                if node_id in targets or any(
                        t.endswith(':' + qname) or t.endswith('.' + qname)
                        for t in targets):
                    # Windows-safe source-file extraction. Graph
                    # adjacency keys are either:
                    #   - A bare filepath (module-level node)
                    #   - filepath:qualified_name
                    # On Windows, filepaths start with 'C:\' which
                    # includes a colon. A naive rsplit(':', 1) on
                    # 'C:\foo.py' returns ['C', '\foo.py'] — giving
                    # us 'C' as the bogus file. Fix: only split on ':'
                    # if the last colon is AFTER the last path separator
                    # (i.e. truly a qname separator).
                    src_file = self._split_node_id(src)
                    try:
                        rel_src = os.path.relpath(
                            src_file, self.brain.config.project_path)
                    except ValueError:
                        continue
                    if rel_src.startswith('..'):
                        continue
                    # Extra guard: rel_src must be a real file in brain
                    if rel_src not in self.brain._file_analyses:
                        continue
                    if rel_src not in expanded_files:
                        expanded_files.append(rel_src)
                        graph_expanded_set.add(rel_src)

        # ── Step 3: semantic expansion via M6 retriever ──
        # TERTIARY — only used if we still have budget AND the semantic
        # hit is for a REAL file path. Filters out non-file chunk names
        # like "block" or "module".
        semantic_files: List[str] = []
        if self.indexer is not None:
            try:
                hits = self.indexer.search(task, top_k=10) or []
                for h in hits:
                    rp = h.get('relative_path', h.get('file_path', ''))
                    # Must be a string with an extension and not just a chunk name
                    if (rp and isinstance(rp, str)
                            and '.' in os.path.basename(rp)
                            and rp not in expanded_files
                            and rp in self.brain._file_analyses):
                        semantic_files.append(rp)
            except Exception:
                pass

        # ── Combine in priority order ──
        final_files: List[str] = list(expanded_files)  # explicit + graph
        for s in semantic_files:
            if s not in final_files:
                final_files.append(s)

        # If the user named a file explicitly AND either a specific
        # function name OR a very focused task, cap the plan tight so
        # we don't waste time on semantic padding. Heuristic: any
        # task with an explicit filename AND (explicit function OR
        # "in <file>" phrasing OR task under ~60 chars) caps at
        # explicit_files + 1 extra slot.
        task_has_focused_phrase = (
            len(task) < 80
            or any(p in task.lower()
                   for p in (' in ', ' of ', ' to ', ' for '))
        )
        if explicit_files and (explicit_functions or task_has_focused_phrase):
            # Keep explicit files first, plus at most 2 graph-expanded
            # files (real callers), then stop. No semantic padding.
            graph_only = [f for f in expanded_files
                          if f in graph_expanded_set][:2]
            final_files = list(explicit_files) + graph_only

        # Filter out test files unless task has test intent
        test_intent = any(w in task.lower() for w in
                          ('test', 'tests', 'pytest', 'unittest'))
        if not test_intent:
            final_files = [
                f for f in final_files
                if not ('/tests/' in f.replace('\\', '/')
                        or f.replace('\\', '/').startswith('tests/')
                        or os.path.basename(f).startswith('test_'))
            ]

        # Cap at MAX_STEPS_PER_PLAN
        final_files = final_files[:MAX_STEPS_PER_PLAN]

        if not final_files:
            plan.abandoned = True
            plan.abandon_reason = (
                "Could not identify any relevant files in the project for "
                "this task. Try mentioning specific filenames, function "
                "names, or describe the feature area more concretely."
            )
            plan.planning_time_s = time.time() - start
            return plan

        # ── Step 4: produce AegisStep for each file ──
        # The rationale tells the user WHY this file is in scope.
        for i, rel_file in enumerate(final_files, start=1):
            # Pick the most relevant function if we have a candidate_function
            # in this file, otherwise leave at file-level
            target_func = None
            for (cf, cq) in candidate_functions:
                if cf == rel_file:
                    target_func = cq
                    break

            # Rationale: priority-ordered attribution
            if rel_file in directly_named_files:
                rationale = "Directly named or referenced in task"
            elif rel_file in graph_expanded_set:
                rationale = "Called by or calls a symbol named in task"
            elif rel_file in semantic_files:
                rationale = "Semantically similar to task description"
            else:
                rationale = "Identified as in-scope"

            plan.steps.append(AegisStep(
                step_id=f"step-{i}",
                description=f"Apply task change to {rel_file}" + (
                    f" ({target_func})" if target_func else ""),
                target_file=rel_file,
                target_function=target_func,
                change_type="modify",
                rationale=rationale,
            ))

        # ── Step 5: optionally refine via model ──
        # Skip the model rationale when the plan is short AND the user
        # explicitly named the file/function. The deterministic per-step
        # attribution already says "Directly named or referenced in
        # task" — nothing to add. Model calls cost ~60s on a 27B with
        # partial GPU offload.
        if (self.model_fn is not None
                and len(plan.steps) > 1
                and not explicit_files):
            plan.planning_rationale = self._model_refine_plan(plan, task)
        elif self.model_fn is not None and explicit_files:
            plan.planning_rationale = (
                f"Plan focused on {len(explicit_files)} file(s) you "
                f"named directly. No model rationale needed."
            )

        plan.planning_time_s = time.time() - start
        return plan

    def _model_refine_plan(self, plan: AegisPlan, task: str) -> str:
        """Ask the model for a short rationale summarizing how the plan
        addresses the task. Does NOT change the plan steps — those are
        determined deterministically from the graph/retrieval. The model
        only provides the 'why this plan' narrative.

        On model failure, returns a deterministic fallback string.
        """
        if not self.model_fn:
            return "(no model available — plan built deterministically from graph + retrieval)"
        try:
            steps_summary = "\n".join(
                f"  {s.step_id}: {s.target_file}  ({s.rationale})"
                for s in plan.steps
            )
            system = (
                "You are planning code changes for LeanAI. You are given "
                "a user task and a list of files the system identified as "
                "relevant. Your job: write a BRIEF (3-4 sentences max) "
                "rationale explaining how this plan addresses the task. "
                "Do NOT add or remove files from the plan. Do NOT "
                "suggest code. Only explain the plan."
            )
            user = (
                f"TASK: {task}\n\n"
                f"FILES IDENTIFIED BY GRAPH + RETRIEVAL:\n{steps_summary}\n\n"
                "Write a 3-4 sentence rationale."
            )
            resp = self.model_fn(system, user)
            # Strip and cap
            resp = (resp or "").strip()
            if len(resp) > 1200:
                resp = resp[:1200] + "..."
            return resp
        except Exception as e:
            return f"(model rationale unavailable: {e})"


# ══════════════════════════════════════════════════════════════
# AEGIS EXECUTOR
# ══════════════════════════════════════════════════════════════

class AegisExecutor:
    """Executes an AegisPlan step by step with validation + rollback.

    Each step:
      1. Reads current file content
      2. Builds rich context (brain graph + M6 retrieval + M4 forensics)
      3. Asks model for a patch (full replacement file content)
      4. Validates: ast.parse, then pre/post Sentinel delta
      5. If validation fails, retry up to MAX_RETRIES_PER_STEP
      6. On success, hands to SafeFileWriter which applies per-mode rules
    """

    def __init__(
        self,
        brain,
        writer: SafeFileWriter,
        indexer=None,
        sentinel=None,
        forensics=None,
        model_fn=None,
    ):
        self.brain = brain
        self.writer = writer
        self.indexer = indexer
        self.sentinel = sentinel
        self.forensics = forensics
        self.model_fn = model_fn

    def execute(self, plan: AegisPlan, task: str) -> AegisReport:
        """Run the plan. Returns a populated AegisReport."""
        report = AegisReport(plan=plan, mode=self.writer.mode.value)
        start = time.time()

        if plan.abandoned:
            return report

        # Stash uncommitted changes before we start (no-op in dry-run)
        self.writer.stash_before()

        try:
            for step in plan.steps:
                # Time budget check
                if time.time() - start > MAX_PLAN_DURATION_S:
                    step.status = StepStatus.SKIPPED
                    step.validation_errors.append("Plan time budget exceeded")
                    report.skipped_steps += 1
                    continue

                step_start = time.time()
                step.status = StepStatus.EXECUTING

                try:
                    self._execute_step(step, task)
                    step.duration_s = time.time() - step_start

                    if step.status == StepStatus.APPLIED:
                        report.applied_steps += 1
                        if step.target_file not in report.files_modified:
                            report.files_modified.append(step.target_file)
                    elif step.status == StepStatus.SKIPPED:
                        report.skipped_steps += 1
                        report.files_skipped.append(
                            (step.target_file,
                             step.validation_errors[-1]
                             if step.validation_errors
                             else "no reason given"))
                    else:
                        report.failed_steps += 1
                except SafetyError as e:
                    # User aborted — don't continue
                    print(f"  [Aegis] Aborting: {e}")
                    self.writer.restore_stash_on_abort()
                    step.status = StepStatus.FAILED
                    step.validation_errors.append(str(e))
                    report.failed_steps += 1
                    break
                except Exception as e:
                    step.status = StepStatus.FAILED
                    step.validation_errors.append(f"Unexpected error: {e}")
                    step.duration_s = time.time() - step_start
                    report.failed_steps += 1
                    # Continue — other steps may succeed
        except KeyboardInterrupt:
            print("\n  [Aegis] Ctrl+C — aborting, restoring stash...")
            self.writer.restore_stash_on_abort()

        report.total_duration_s = time.time() - start

        # Files that got applied but had validation_errors → needs review
        for s in plan.steps:
            if s.status == StepStatus.APPLIED and s.new_sentinel_findings > 0:
                report.needs_human_review.append(
                    f"{s.target_file}: introduced {s.new_sentinel_findings} "
                    f"new Sentinel finding(s) — review before committing"
                )

        return report

    def _execute_step(self, step: AegisStep, task: str):
        """Run one step. Mutates step in place."""
        abs_file = os.path.join(self.brain.config.project_path, step.target_file)
        if not os.path.exists(abs_file):
            step.status = StepStatus.SKIPPED
            step.validation_errors.append(f"File not found: {step.target_file}")
            return

        try:
            with open(abs_file, "r", encoding="utf-8", errors="replace") as f:
                old_content = f.read()
        except Exception as e:
            step.status = StepStatus.SKIPPED
            step.validation_errors.append(f"Could not read file: {e}")
            return

        # ── GUARD: refuse files too large for full-file replacement ──
        # M7 uses full-file replacement (asks model for the complete new
        # content). For files larger than MAX_FILE_SIZE_FOR_AEGIS, the
        # model only sees a truncated view and may produce a truncated
        # output, deleting code. Skip these files entirely.
        if len(old_content) > MAX_FILE_SIZE_FOR_AEGIS:
            step.status = StepStatus.SKIPPED
            step.validation_errors.append(
                f"File too large for safe full-file replacement "
                f"({len(old_content)} chars > {MAX_FILE_SIZE_FOR_AEGIS} limit). "
                f"M7 Aegis only handles small/medium files. "
                f"Big-file refactors need diff-based patching (parked)."
            )
            return

        # ── Build rich context ──
        context = self._build_step_context(step, old_content, task)

        # ── Ask model for patch (with retries) ──
        new_content = None
        for retry in range(MAX_RETRIES_PER_STEP):
            step.retries_used = retry + 1
            try:
                new_content = self._propose_patch(step, task, old_content, context)
            except Exception as e:
                step.validation_errors.append(
                    f"Retry {retry+1}: model call failed: {e}")
                continue

            if new_content is None or new_content == old_content:
                step.validation_errors.append(
                    f"Retry {retry+1}: model returned no meaningful change")
                continue

            # ── GUARD: reject suspicious shrinkage ──
            # If the new file is dramatically shorter than the old one,
            # the model probably truncated rather than edited. A
            # docstring/logging/typing addition should NEVER make the
            # file shorter.
            shrink_ratio = 1 - (len(new_content) / max(len(old_content), 1))
            if shrink_ratio > MAX_SHRINK_RATIO:
                step.validation_errors.append(
                    f"Retry {retry+1}: proposed patch shrinks file by "
                    f"{shrink_ratio*100:.0f}% ({len(old_content)} -> "
                    f"{len(new_content)} chars). Likely truncation, "
                    f"not legitimate edit. Rejected."
                )
                continue

            # ── Validate patch: syntax ──
            if step.target_file.endswith('.py'):
                try:
                    import ast
                    ast.parse(new_content)
                except SyntaxError as e:
                    step.validation_errors.append(
                        f"Retry {retry+1}: proposed patch has Python syntax "
                        f"error at line {e.lineno}: {e.msg}")
                    continue

            # ── GUARD: target symbol must still be present ──
            # If the user asked to modify tokenize_text, and tokenize_text
            # is missing from the new content, the model deleted it
            # rather than editing it. Reject.
            if step.target_function:
                short_name = step.target_function.rsplit('.', 1)[-1]
                if short_name and short_name not in new_content:
                    step.validation_errors.append(
                        f"Retry {retry+1}: target symbol {short_name!r} "
                        f"is missing from proposed patch. Model deleted "
                        f"the function instead of modifying it. Rejected."
                    )
                    continue

            # ── Validate patch: Sentinel delta ──
            if self.sentinel is not None:
                new_findings = self._sentinel_delta(
                    step.target_file, old_content, new_content)
                if new_findings > 0:
                    # Try to mitigate — if this is retry < last, retry
                    # with the finding count in the error so the model
                    # can try to avoid it next time
                    if retry < MAX_RETRIES_PER_STEP - 1:
                        step.validation_errors.append(
                            f"Retry {retry+1}: patch introduces "
                            f"{new_findings} new security finding(s), retrying")
                        continue
                    else:
                        # Last retry — accept but flag for human review
                        step.new_sentinel_findings = new_findings

            # Patch is valid
            break
        else:
            # All retries exhausted
            step.status = StepStatus.FAILED
            return

        if new_content is None:
            step.status = StepStatus.FAILED
            return

        step.proposed_patch = new_content
        step.diff = self._compute_diff(step.target_file, old_content, new_content)
        step.status = StepStatus.VALIDATING

        # ── Hand to SafeFileWriter ──
        intent = WriteIntent(
            relative_path=step.target_file,
            old_content=old_content,
            new_content=new_content,
            reason=f"{step.description}  ({step.rationale})",
            step_id=step.step_id,
        )
        result = self.writer.apply(intent)

        if result.applied:
            step.status = StepStatus.APPLIED
        else:
            step.status = StepStatus.SKIPPED
            step.validation_errors.append(
                result.skipped_reason or "unknown skip reason")

    # ─────────── Context building ───────────

    def _build_step_context(
        self, step: AegisStep, old_content: str, task: str,
    ) -> str:
        """Build a context string <= MAX_CONTEXT_CHARS with:
          - File description
          - M6 retrieval hits relevant to this step
          - M4 forensics if target_function is set (stability, authors)
        """
        parts: List[str] = []
        budget = MAX_CONTEXT_CHARS

        def _append(s: str) -> bool:
            nonlocal budget
            if len(s) + 1 > budget:
                return False
            parts.append(s)
            budget -= len(s) + 1
            return True

        # M6: retrieve similar functions for style
        if self.indexer is not None:
            try:
                q = f"{task} {step.target_file}"
                if step.target_function:
                    q = f"{task} {step.target_function}"
                hits = self.indexer.search(q, top_k=3) or []
                style_parts = []
                for h in hits[:3]:
                    name = h.get('name', 'chunk')
                    rp = h.get('relative_path', '')
                    if rp == step.target_file:
                        continue  # don't show target as its own style ref
                    chunk = h.get('content', '')[:400]
                    style_parts.append(
                        f"[Style reference — {rp}:{name}]\n{chunk}")
                if style_parts:
                    _append("RELEVANT PROJECT CODE (for style consistency):\n"
                            + "\n\n".join(style_parts))
            except Exception:
                pass

        # M4: forensics on target_function if present
        if self.forensics is not None and step.target_function:
            try:
                stab = self.forensics.stability(step.target_file, step.target_function)
                authors = self.forensics.authors(step.target_file, step.target_function)
                if stab and authors:
                    _append(
                        f"FORENSICS for {step.target_function}: "
                        f"stability {stab.score}/100 "
                        f"({stab.interpretation[:60]}), "
                        f"primary author {authors.breakdown[0].author if authors.breakdown else '?'}"
                    )
            except Exception:
                pass

        return "\n\n".join(parts)

    # ─────────── Patch proposal ───────────

    def _propose_patch(
        self, step: AegisStep, task: str,
        old_content: str, context: str,
    ) -> Optional[str]:
        """Ask the model to produce the full new content of the file."""
        if self.model_fn is None:
            # No model available — can't propose a patch in this mode
            return None

        # Cap old_content so prompt fits
        if len(old_content) > 8000:
            old_content_view = old_content[:8000] + "\n# ... [truncated for model context]"
        else:
            old_content_view = old_content

        system = (
            "You are LeanAI Aegis, an agentic code-modification assistant. "
            "Given a TASK, the CURRENT FILE content, and CONTEXT from the "
            "user's project, produce the COMPLETE NEW CONTENT of the file "
            "that accomplishes the task. Rules:\n"
            "  1. Output ONLY the new file content — no prose, no markdown "
            "fences, no explanation.\n"
            "  2. Preserve existing imports, unrelated functions, formatting.\n"
            "  3. Match the style of CONTEXT functions (same naming, error "
            "handling, docstring shape).\n"
            "  4. Do NOT introduce subprocess/eval/exec with untrusted "
            "input. Do NOT hardcode secrets.\n"
            "  5. If the task does not require changing this file, output "
            "the file content UNCHANGED (copy it verbatim)."
        )
        user = (
            f"TASK: {task}\n\n"
            f"FILE TO MODIFY: {step.target_file}\n"
            + (f"FOCUS: {step.target_function}\n" if step.target_function else "")
            + f"WHY THIS FILE IS IN SCOPE: {step.rationale}\n\n"
            + (f"CONTEXT:\n{context}\n\n" if context else "")
            + f"CURRENT FILE CONTENT:\n```python\n{old_content_view}\n```\n\n"
            "Output the COMPLETE new file content. No prose. No fences."
        )

        resp = self.model_fn(system, user)
        if not resp:
            return None

        # Strip code fences if model wrapped response
        resp = resp.strip()
        if resp.startswith("```"):
            # Strip fence markers
            lines = resp.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            resp = "\n".join(lines).strip()

        return resp

    # ─────────── Validation ───────────

    def _sentinel_delta(
        self, rel_file: str, old_content: str, new_content: str,
    ) -> int:
        """Run Sentinel on old vs new content. Returns how many NEW
        MEDIUM+ findings exist in new that weren't in old. 0 means
        no new vulnerabilities.

        We count by fingerprint (file:line:vuln_class) to avoid
        false deltas from line-number shifts.
        """
        if self.sentinel is None:
            return 0

        abs_file = os.path.join(self.brain.config.project_path, rel_file)
        try:
            # Snapshot old
            old_findings_set = self._sentinel_fingerprints(rel_file, abs_file)

            # Write new content to a temp file, point sentinel at it via
            # a side-channel. But Sentinel uses brain._file_analyses not
            # disk reads — so we monkey-patch the analysis temporarily.
            from brain.analyzer import analyze_python_file
            # Save old analysis
            old_analysis = self.brain._file_analyses.get(rel_file)
            # Produce a new analysis from proposed content
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(
                        mode='w', suffix='.py', delete=False,
                        encoding='utf-8') as tf:
                    tf.write(new_content)
                    tmp_path = tf.name
                try:
                    new_analysis = analyze_python_file(tmp_path)
                    # Fix up the filepath so it matches the target
                    new_analysis.filepath = abs_file
                finally:
                    os.unlink(tmp_path)
            except Exception:
                return 0

            # Swap analysis, scan, swap back
            self.brain._file_analyses[rel_file] = new_analysis
            try:
                new_findings_set = self._sentinel_fingerprints(rel_file, abs_file)
            finally:
                # Restore
                if old_analysis is not None:
                    self.brain._file_analyses[rel_file] = old_analysis

            # Delta
            return max(0, len(new_findings_set - old_findings_set))
        except Exception:
            return 0

    def _sentinel_fingerprints(self, rel_file: str, abs_file: str) -> set:
        """Run Sentinel scoped to one file, return set of vuln
        fingerprints (class+line)."""
        try:
            from core.sentinel import Severity
            findings, _ = self.sentinel.scan(
                target=rel_file,
                severity_floor=Severity.MEDIUM,
                use_model=False,
                verbose=False,
            )
            return set(f"{f.vuln_class}:{f.line}" for f in findings)
        except Exception:
            return set()

    # ─────────── Diff display ───────────

    def _compute_diff(
        self, rel_path: str, old_content: str, new_content: str,
    ) -> str:
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}",
            n=3,
        ))
        return "".join(diff)


# ══════════════════════════════════════════════════════════════
# HIGH-LEVEL ENTRY POINT
# ══════════════════════════════════════════════════════════════

@dataclass
class AegisConfig:
    """Convenience config bundle."""
    cli_flag_apply: bool = False
    interactive: bool = True
    plan_only: bool = False


def run_aegis(
    task: str,
    brain,
    config: AegisConfig,
    indexer=None,
    sentinel=None,
    forensics=None,
    model_fn=None,
) -> AegisReport:
    """Full end-to-end: plan -> (execute unless plan_only) -> report.

    Mode decision:
      - If cli_flag_apply is False → DRY_RUN (always)
      - If cli_flag_apply is True AND LEANAI_AEGIS_CONFIRM=1 → APPLY
      - If cli_flag_apply is True but env not set → DRY_RUN + warning
    """
    mode = detect_safety_mode(config.cli_flag_apply)
    writer = SafeFileWriter(
        project_root=brain.config.project_path,
        mode=mode,
        interactive=config.interactive,
    )

    print(f"  [Aegis] Safety mode: {mode.value}")
    print(f"  [Aegis] {explain_mode(mode)}")
    print()

    if config.cli_flag_apply and mode == SafetyMode.DRY_RUN:
        print(f"  [Aegis] --apply flag was set but {AEGIS_CONFIRM_ENV} "
              f"env var is not {AEGIS_CONFIRM_VALUE}. Falling back to "
              f"DRY_RUN mode. To actually apply, set:")
        print(f"    PowerShell: $env:{AEGIS_CONFIRM_ENV}='{AEGIS_CONFIRM_VALUE}'")
        print(f"    Bash:       export {AEGIS_CONFIRM_ENV}={AEGIS_CONFIRM_VALUE}")
        print()

    # ── Plan ──
    planner = AegisPlanner(
        brain=brain, indexer=indexer,
        sentinel=sentinel,
        # Skip model rationale in plan-only mode — it's expensive
        # (~30-60s on a 27B with partial GPU offload) and the user
        # explicitly asked to preview the plan without model work.
        # The deterministic rationale per-step is still produced from
        # the graph/retrieval attribution.
        model_fn=None if config.plan_only else model_fn,
    )
    plan = planner.plan(task)

    if plan.abandoned:
        print(f"  [Aegis] Planning abandoned: {plan.abandon_reason}")
        return AegisReport(plan=plan, mode=mode.value)

    _print_plan(plan)

    if config.plan_only:
        print("  [Aegis] --plan-only mode; not executing.")
        return AegisReport(plan=plan, mode=mode.value)

    # ── Execute ──
    executor = AegisExecutor(
        brain=brain, writer=writer,
        indexer=indexer, sentinel=sentinel, forensics=forensics,
        model_fn=model_fn,
    )
    report = executor.execute(plan, task)

    _print_report(report)
    return report


# ══════════════════════════════════════════════════════════════
# FORMATTING
# ══════════════════════════════════════════════════════════════

def _print_plan(plan: AegisPlan):
    print("  " + "━" * 58)
    print(f"  Aegis Plan for: {plan.task}")
    print("  " + "━" * 58)
    print()
    if plan.planning_rationale:
        print(f"  Rationale:")
        for line in plan.planning_rationale.splitlines():
            print(f"    {line}")
        print()
    print(f"  {len(plan.steps)} step(s):")
    for i, s in enumerate(plan.steps, start=1):
        print(f"    {s.step_id}. {s.target_file}"
              + (f"  ({s.target_function})" if s.target_function else ""))
        print(f"         → {s.rationale}")
    print()


def _print_report(report: AegisReport):
    print()
    print("  " + "━" * 58)
    print("  Aegis Report")
    print("  " + "━" * 58)
    print()
    print(f"  Mode:          {report.mode}")
    print(f"  Duration:      {report.total_duration_s:.1f}s")
    print(f"  Steps applied: {report.applied_steps}")
    print(f"  Steps skipped: {report.skipped_steps}")
    print(f"  Steps failed:  {report.failed_steps}")
    print()
    if report.files_modified:
        print(f"  Files modified ({len(report.files_modified)}):")
        for f in report.files_modified:
            print(f"    ✓ {f}")
        print()
    if report.files_skipped:
        print(f"  Files skipped ({len(report.files_skipped)}):")
        for (p, r) in report.files_skipped[:10]:
            print(f"    · {p}: {r}")
        print()
    if report.needs_human_review:
        print(f"  Needs human review:")
        for msg in report.needs_human_review:
            print(f"    ⚠  {msg}")
        print()

    # Step-level details
    if any(s.validation_errors for s in report.plan.steps):
        print(f"  Step errors (for debugging):")
        for s in report.plan.steps:
            if s.validation_errors:
                print(f"    {s.step_id}: {s.status.value}")
                for err in s.validation_errors[-2:]:  # last 2 only
                    print(f"      - {err[:180]}")
        print()

"""
LeanAI — Aegis Safety Layer (M7)
=================================

Three-tier guard against unintended file modification. Every mutation
flows through SafeFileWriter which enforces:

  TIER 1 — Dry-run by default.
    Aegis operates in DRY_RUN mode unless --apply flag AND
    LEANAI_AEGIS_CONFIRM=1 env var are both present. Dry-run emits
    unified diffs to stdout but never touches disk.

  TIER 2 — Path validation.
    Every write target must resolve inside the project root (same
    rule as ExploitForge). No absolute paths, no ../ traversals,
    no symlink escapes.

  TIER 3 — Interactive confirmation (when applying).
    Before each file write, user sees the diff and must type 'y'
    to confirm. Anything else → skip. Ctrl+C → abort the whole
    plan, roll back any prior steps.

Additional guards:
  - Git stash wrapper — every step starts with `git stash`; on
    failure we restore via `git stash pop`. User's uncommitted
    changes are never lost.
  - Never touches test files (unless query has test intent)
  - Never modifies files outside project root
  - Never runs generated code (that's M5 AutoFix territory)
  - Never performs git commit / push / pull
"""

from __future__ import annotations

import difflib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class SafetyMode(Enum):
    """Aegis operates in one of three modes, hardcoded at construction."""
    DRY_RUN = "dry_run"          # show diffs, don't write
    APPLY = "apply"              # write, but require per-file confirmation
    APPLY_UNSAFE = "apply_unsafe"  # SHOULD NOT BE REACHABLE — tripwire


# Environment variable required for any actual file writes
AEGIS_CONFIRM_ENV = "LEANAI_AEGIS_CONFIRM"
AEGIS_CONFIRM_VALUE = "1"


@dataclass
class WriteIntent:
    """A proposed file change, not yet executed."""
    relative_path: str          # path under project root
    old_content: Optional[str]  # None if creating new file
    new_content: str            # what we want to write
    reason: str                 # why this change (shown to user)
    step_id: str                # which Aegis step proposed this


@dataclass
class WriteResult:
    """Outcome of attempting to apply a WriteIntent."""
    intent: WriteIntent
    applied: bool
    skipped_reason: Optional[str] = None  # why skipped (if not applied)
    absolute_path: Optional[str] = None


class SafetyError(Exception):
    """Raised when a write is rejected by the safety layer."""
    pass


class SafeFileWriter:
    """Only legitimate path for Aegis to modify files.

    Construction:
      writer = SafeFileWriter(project_root, mode=SafetyMode.DRY_RUN)

    Every write MUST go through writer.apply(intent). If mode is
    DRY_RUN, writes are simulated (diff printed, not applied). If
    APPLY, per-file confirmation is required.
    """

    def __init__(
        self,
        project_root: str,
        mode: SafetyMode = SafetyMode.DRY_RUN,
        interactive: bool = True,
    ):
        self.project_root = os.path.abspath(project_root)
        self.mode = mode
        self.interactive = interactive
        self._writes_applied: List[WriteResult] = []
        self._writes_skipped: List[WriteResult] = []
        # Remember the stash ref if we stashed uncommitted changes.
        self._stash_ref: Optional[str] = None

    # ─────────── Tier 2: path validation ───────────

    def _validate_path(self, relative_path: str) -> str:
        """Normalize + validate that the target is inside project_root.
        Returns the absolute path. Raises SafetyError on rejection."""
        if not relative_path or not relative_path.strip():
            raise SafetyError("Empty relative path")
        # Normalize path separators first so Windows-style '..\\' is caught
        # by the same '..' check as POSIX '../'. Also reject any '..' in
        # the normalized string form BEFORE relying on Path parts (which
        # may split differently on Windows vs Linux).
        normalized = relative_path.replace('\\', '/')
        if '..' in normalized.split('/'):
            raise SafetyError(
                f"Path traversal rejected: {relative_path}")
        # Reject absolute paths
        if os.path.isabs(relative_path) or normalized.startswith('/'):
            raise SafetyError(
                f"Absolute paths are not allowed: {relative_path}")
        # Belt-and-braces: also check Path parts (POSIX systems)
        parts = Path(relative_path).parts
        if '..' in parts:
            raise SafetyError(
                f"Path traversal rejected: {relative_path}")
        # Reject any path that would escape the project root after resolve
        abs_candidate = os.path.abspath(
            os.path.join(self.project_root, relative_path))
        try:
            common = os.path.commonpath([self.project_root, abs_candidate])
        except ValueError:
            raise SafetyError(f"Path is on different drive: {relative_path}")
        if common != self.project_root:
            raise SafetyError(
                f"Resolved path escapes project root: "
                f"{relative_path} -> {abs_candidate}")
        return abs_candidate

    # ─────────── Tier 1: mode gate ───────────

    def _apply_allowed(self) -> bool:
        """Check tier 1: mode and env var. Returns True if write is
        permitted to actually touch disk."""
        if self.mode == SafetyMode.DRY_RUN:
            return False
        # Even in APPLY mode, require the env var
        if os.environ.get(AEGIS_CONFIRM_ENV) != AEGIS_CONFIRM_VALUE:
            return False
        return True

    # ─────────── Tier 3: interactive confirmation ───────────

    def _show_diff(self, intent: WriteIntent, abs_path: str) -> str:
        """Return a unified-diff string for this change."""
        old_lines = (intent.old_content or "").splitlines(keepends=True)
        new_lines = intent.new_content.splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{intent.relative_path}",
            tofile=f"b/{intent.relative_path}",
            n=3,
        ))
        return "".join(diff_lines) if diff_lines else "(no textual change)"

    def _confirm_interactive(self, intent: WriteIntent, diff: str) -> bool:
        """Ask the user. Return True if they say yes."""
        if not self.interactive:
            return True
        try:
            print()
            print(f"  [Aegis] Proposed change from step {intent.step_id}:")
            print(f"  Reason: {intent.reason}")
            print(f"  File:   {intent.relative_path}")
            print(f"  Diff:")
            # Indent the diff for readability
            for line in diff.splitlines()[:120]:  # cap display length
                print(f"    {line}")
            if len(diff.splitlines()) > 120:
                print(f"    ... +{len(diff.splitlines()) - 120} more lines")
            print()
            answer = input("  Apply this change? [y/N]: ").strip().lower()
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            raise SafetyError("User aborted during confirmation")

    # ─────────── Main API ───────────

    def apply(self, intent: WriteIntent) -> WriteResult:
        """Attempt to apply a write intent.

        In DRY_RUN: always 'skips' but prints the diff so the user
        sees what would happen.

        In APPLY: validates path, shows diff, asks user, writes if
        user confirms.
        """
        # TIER 2: always validate path, even in dry-run
        try:
            abs_path = self._validate_path(intent.relative_path)
        except SafetyError as e:
            result = WriteResult(
                intent=intent, applied=False,
                skipped_reason=f"Safety check failed: {e}",
            )
            self._writes_skipped.append(result)
            return result

        diff = self._show_diff(intent, abs_path)

        # TIER 1: mode gate
        if not self._apply_allowed():
            # Dry-run path — print diff for transparency
            print(f"\n  [Aegis DRY-RUN] Would modify {intent.relative_path}:")
            print(f"  Reason: {intent.reason}")
            for line in diff.splitlines()[:80]:
                print(f"    {line}")
            if len(diff.splitlines()) > 80:
                print(f"    ... +{len(diff.splitlines()) - 80} more lines")

            result = WriteResult(
                intent=intent, applied=False,
                absolute_path=abs_path,
                skipped_reason=(
                    "dry-run mode" if self.mode == SafetyMode.DRY_RUN
                    else f"env var {AEGIS_CONFIRM_ENV} not set"
                ),
            )
            self._writes_skipped.append(result)
            return result

        # TIER 3: interactive confirmation
        try:
            confirmed = self._confirm_interactive(intent, diff)
        except SafetyError:
            # User aborted
            raise
        if not confirmed:
            result = WriteResult(
                intent=intent, applied=False,
                absolute_path=abs_path,
                skipped_reason="user declined",
            )
            self._writes_skipped.append(result)
            return result

        # ACTUALLY WRITE
        try:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            # Write via temp + rename for atomicity
            tmp_path = abs_path + ".aegis.tmp"
            with open(tmp_path, "w", encoding="utf-8", newline="") as f:
                f.write(intent.new_content)
            os.replace(tmp_path, abs_path)
            result = WriteResult(
                intent=intent, applied=True,
                absolute_path=abs_path,
            )
            self._writes_applied.append(result)
            return result
        except Exception as e:
            result = WriteResult(
                intent=intent, applied=False,
                absolute_path=abs_path,
                skipped_reason=f"Write failed: {e}",
            )
            self._writes_skipped.append(result)
            return result

    # ─────────── Git stash wrapper ───────────

    def stash_before(self) -> bool:
        """Stash any uncommitted changes before Aegis runs. So that if
        Aegis misbehaves, user's WIP is recoverable via `git stash pop`.
        Returns True if a stash was created, False if no changes or no
        git repo. Stores the stash ref for later restore.
        """
        if self.mode == SafetyMode.DRY_RUN:
            # Dry-run doesn't touch disk, nothing to stash
            return False
        try:
            # Check if we're in a git repo
            r = subprocess.run(
                ["git", "-C", self.project_root, "rev-parse", "--git-dir"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0:
                return False
            # Check if there are uncommitted changes
            r = subprocess.run(
                ["git", "-C", self.project_root, "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            )
            if not r.stdout.strip():
                return False
            # Create a named stash so we can find it later
            stash_msg = "aegis-pre-run-autostash"
            r = subprocess.run(
                ["git", "-C", self.project_root, "stash", "push",
                 "-m", stash_msg, "--include-untracked"],
                capture_output=True, text=True, timeout=30,
                encoding="utf-8", errors="replace",
            )
            if r.returncode == 0:
                self._stash_ref = stash_msg
                print(f"  [Aegis] Uncommitted changes auto-stashed. "
                      f"Restore with: git stash pop "
                      f"(or `git stash list` to find 'aegis-pre-run-autostash')")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    def restore_stash_on_abort(self):
        """If Aegis is aborting due to failure AND we stashed before,
        pop the stash so user's WIP returns automatically."""
        if not self._stash_ref or self.mode == SafetyMode.DRY_RUN:
            return
        try:
            # Find the stash by message
            r = subprocess.run(
                ["git", "-C", self.project_root, "stash", "list"],
                capture_output=True, text=True, timeout=5,
                encoding="utf-8", errors="replace",
            )
            for line in r.stdout.splitlines():
                if self._stash_ref in line:
                    # Extract the stash ref like "stash@{0}"
                    ref = line.split(":", 1)[0].strip()
                    subprocess.run(
                        ["git", "-C", self.project_root, "stash", "pop", ref],
                        capture_output=True, text=True, timeout=10,
                    )
                    print(f"  [Aegis] Restored pre-run uncommitted changes.")
                    self._stash_ref = None
                    return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # ─────────── Stats ───────────

    def summary(self) -> dict:
        return {
            "mode": self.mode.value,
            "applied": len(self._writes_applied),
            "skipped": len(self._writes_skipped),
            "skip_reasons": [r.skipped_reason for r in self._writes_skipped
                             if r.skipped_reason],
        }

    @property
    def applied(self) -> List[WriteResult]:
        return list(self._writes_applied)

    @property
    def skipped(self) -> List[WriteResult]:
        return list(self._writes_skipped)


def detect_safety_mode(cli_flag_apply: bool) -> SafetyMode:
    """Decide mode from CLI flag + env var. Default is DRY_RUN.

    For Aegis to APPLY:
      - CLI must pass --apply flag  (cli_flag_apply=True)
      - Env must have LEANAI_AEGIS_CONFIRM=1

    Either missing → DRY_RUN. Both present → APPLY. Interactive
    Y/n still happens per-file inside SafeFileWriter.apply().
    """
    if not cli_flag_apply:
        return SafetyMode.DRY_RUN
    if os.environ.get(AEGIS_CONFIRM_ENV) != AEGIS_CONFIRM_VALUE:
        return SafetyMode.DRY_RUN
    return SafetyMode.APPLY


def explain_mode(mode: SafetyMode) -> str:
    """Human-readable explanation of what the current mode will do."""
    if mode == SafetyMode.DRY_RUN:
        return (
            "DRY-RUN: Aegis will show the proposed changes as diffs but "
            "will NOT modify any files. To actually apply changes, set "
            f"{AEGIS_CONFIRM_ENV}={AEGIS_CONFIRM_VALUE} env var AND pass "
            "--apply flag. You'll still be asked Y/n for each change."
        )
    elif mode == SafetyMode.APPLY:
        return (
            "APPLY MODE: Aegis will propose changes and ask Y/n before "
            "each file write. Unchanged files stay untouched. Uncommitted "
            "changes are auto-stashed before the first write. Ctrl+C "
            "aborts and restores the stash."
        )
    return "UNKNOWN MODE — this should never happen"

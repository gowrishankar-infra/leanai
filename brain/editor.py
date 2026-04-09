"""
LeanAI Phase 7d — Multi-File Coherent Editor
When you change a function signature, automatically finds and updates
every caller, import, and reference across the entire project.

Capabilities:
  - Rename function/class across all files
  - Update callers when function signature changes
  - Find all references to a symbol
  - Safe refactoring with preview before applying
"""

import os
import re
import ast
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set


@dataclass
class FileEdit:
    """A single edit to apply to a file."""
    filepath: str
    line_number: int
    old_text: str
    new_text: str
    context: str = ""  # surrounding line for preview

    def preview(self) -> str:
        return f"  {self.filepath}:{self.line_number}\n    - {self.old_text.strip()}\n    + {self.new_text.strip()}"


@dataclass
class RefactorPlan:
    """Plan for a multi-file refactoring operation."""
    operation: str          # "rename", "update_signature", "find_references"
    target: str             # what we're refactoring
    edits: List[FileEdit] = field(default_factory=list)
    files_affected: Set[str] = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)

    @property
    def num_edits(self) -> int:
        return len(self.edits)

    @property
    def num_files(self) -> int:
        return len(self.files_affected)

    def preview(self) -> str:
        lines = [
            f"Refactor: {self.operation} '{self.target}'",
            f"Files affected: {self.num_files}",
            f"Edits: {self.num_edits}",
            "",
        ]
        for edit in self.edits[:20]:  # limit preview
            lines.append(edit.preview())
        if self.num_edits > 20:
            lines.append(f"  ... and {self.num_edits - 20} more edits")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


@dataclass
class Reference:
    """A reference to a symbol in the codebase."""
    filepath: str
    line_number: int
    line_text: str
    ref_type: str  # "definition", "import", "call", "reference"


SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".pytest_cache", "dist", "build", ".eggs",
}


class MultiFileEditor:
    """
    Coherent multi-file editor for project-wide refactoring.
    
    Usage:
        editor = MultiFileEditor("/path/to/project")
        
        # Find all references
        refs = editor.find_references("handle_request")
        
        # Rename across entire project
        plan = editor.rename("old_name", "new_name")
        print(plan.preview())  # see what will change
        editor.apply(plan)     # apply the changes
        
        # Update when signature changes
        plan = editor.update_signature("func", old_args="a, b", new_args="a, b, c=None")
    """

    def __init__(self, project_path: str):
        self.project_path = os.path.abspath(project_path)

    def _get_python_files(self) -> List[str]:
        """Get all Python files in the project."""
        files = []
        for root, dirs, filenames in os.walk(self.project_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in filenames:
                if f.endswith(".py"):
                    files.append(os.path.join(root, f))
        return files

    def _read_file(self, filepath: str) -> List[str]:
        """Read file lines."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.readlines()
        except (FileNotFoundError, IOError):
            return []

    def _write_file(self, filepath: str, lines: List[str]):
        """Write file lines."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)

    # ── Find References ───────────────────────────────────────────

    def find_references(self, symbol: str) -> List[Reference]:
        """Find all references to a symbol across the project."""
        refs = []
        pattern = re.compile(r'\b' + re.escape(symbol) + r'\b')

        for filepath in self._get_python_files():
            lines = self._read_file(filepath)
            rel_path = os.path.relpath(filepath, self.project_path)

            for i, line in enumerate(lines, 1):
                if pattern.search(line):
                    # Determine reference type
                    stripped = line.strip()
                    if stripped.startswith(f"def {symbol}") or stripped.startswith(f"class {symbol}"):
                        ref_type = "definition"
                    elif "import" in stripped and symbol in stripped:
                        ref_type = "import"
                    elif f"{symbol}(" in stripped:
                        ref_type = "call"
                    else:
                        ref_type = "reference"

                    refs.append(Reference(
                        filepath=rel_path,
                        line_number=i,
                        line_text=stripped,
                        ref_type=ref_type,
                    ))
        return refs

    def find_references_summary(self, symbol: str) -> str:
        """Human-readable summary of all references."""
        refs = self.find_references(symbol)
        if not refs:
            return f"No references found for '{symbol}'"

        lines = [f"References to '{symbol}': {len(refs)} found"]
        by_type = {}
        for r in refs:
            by_type.setdefault(r.ref_type, []).append(r)

        for ref_type in ["definition", "import", "call", "reference"]:
            group = by_type.get(ref_type, [])
            if group:
                lines.append(f"\n  {ref_type.upper()} ({len(group)}):")
                for r in group[:10]:
                    lines.append(f"    {r.filepath}:{r.line_number}  {r.line_text[:80]}")
                if len(group) > 10:
                    lines.append(f"    ... and {len(group) - 10} more")
        return "\n".join(lines)

    # ── Rename ────────────────────────────────────────────────────

    def rename(self, old_name: str, new_name: str) -> RefactorPlan:
        """
        Plan a rename of a symbol across the entire project.
        Returns a RefactorPlan that can be previewed and applied.
        """
        plan = RefactorPlan(operation="rename", target=f"{old_name} → {new_name}")
        pattern = re.compile(r'\b' + re.escape(old_name) + r'\b')

        for filepath in self._get_python_files():
            lines = self._read_file(filepath)
            rel_path = os.path.relpath(filepath, self.project_path)

            for i, line in enumerate(lines, 1):
                if pattern.search(line):
                    new_line = pattern.sub(new_name, line)
                    if new_line != line:
                        plan.edits.append(FileEdit(
                            filepath=filepath,
                            line_number=i,
                            old_text=line,
                            new_text=new_line,
                            context=rel_path,
                        ))
                        plan.files_affected.add(rel_path)

        # Warnings
        if not plan.edits:
            plan.warnings.append(f"No occurrences of '{old_name}' found")
        if new_name in [ref.line_text for ref in self.find_references(new_name)]:
            plan.warnings.append(f"'{new_name}' already exists in the project — possible conflict")

        return plan

    # ── Update Signature ──────────────────────────────────────────

    def update_signature(self, func_name: str, old_args: str, new_args: str) -> RefactorPlan:
        """
        Plan updates when a function signature changes.
        Finds all callers and suggests updates.
        """
        plan = RefactorPlan(
            operation="update_signature",
            target=f"{func_name}({old_args}) → {func_name}({new_args})",
        )

        # Find definition and update it
        old_def = f"def {func_name}({old_args})"
        new_def = f"def {func_name}({new_args})"

        for filepath in self._get_python_files():
            lines = self._read_file(filepath)
            rel_path = os.path.relpath(filepath, self.project_path)

            for i, line in enumerate(lines, 1):
                if old_def in line:
                    new_line = line.replace(old_def, new_def)
                    plan.edits.append(FileEdit(
                        filepath=filepath, line_number=i,
                        old_text=line, new_text=new_line, context=rel_path,
                    ))
                    plan.files_affected.add(rel_path)

        # Find callers that might need updating
        call_pattern = re.compile(r'\b' + re.escape(func_name) + r'\s*\(')
        for filepath in self._get_python_files():
            lines = self._read_file(filepath)
            rel_path = os.path.relpath(filepath, self.project_path)

            for i, line in enumerate(lines, 1):
                if call_pattern.search(line) and f"def {func_name}" not in line:
                    plan.warnings.append(
                        f"Caller at {rel_path}:{i} may need updating: {line.strip()[:80]}"
                    )

        return plan

    # ── Apply Changes ─────────────────────────────────────────────

    def apply(self, plan: RefactorPlan) -> dict:
        """Apply a refactoring plan to the project files."""
        if not plan.edits:
            return {"applied": 0, "files": 0}

        # Group edits by file
        by_file: Dict[str, List[FileEdit]] = {}
        for edit in plan.edits:
            by_file.setdefault(edit.filepath, []).append(edit)

        files_modified = 0
        edits_applied = 0

        for filepath, edits in by_file.items():
            lines = self._read_file(filepath)
            if not lines:
                continue

            # Apply edits in reverse order (so line numbers stay valid)
            edits_sorted = sorted(edits, key=lambda e: e.line_number, reverse=True)
            for edit in edits_sorted:
                idx = edit.line_number - 1
                if 0 <= idx < len(lines):
                    lines[idx] = edit.new_text
                    edits_applied += 1

            self._write_file(filepath, lines)
            files_modified += 1

        return {"applied": edits_applied, "files": files_modified}

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        files = self._get_python_files()
        return {
            "project": self.project_path,
            "python_files": len(files),
        }

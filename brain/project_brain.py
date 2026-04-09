"""
LeanAI Phase 7a — Project Brain
The persistent, always-aware intelligence layer for your codebase.

Watches your project directory, auto-indexes on changes, maintains a full
dependency graph, and can answer structural questions instantly.

Usage:
    brain = ProjectBrain("/path/to/my/project")
    brain.scan()  # initial full scan
    
    # Query the brain
    brain.describe_file("src/api.py")
    brain.what_calls("handle_request")
    brain.what_depends_on("database.py")
    brain.impact_of_changing("models.py")
    brain.find_function("validate_token")
    brain.project_summary()
    
    # Start watching for changes (background)
    brain.start_watching()
"""

import os
import time
import json
import hashlib
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from brain.analyzer import analyze_file, FileAnalysis
from brain.dependency_graph import DependencyGraph, GraphNode


# File extensions to analyze
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
    ".java", ".c", ".cpp", ".h", ".hpp", ".cs",
    ".rb", ".php", ".swift", ".kt",
}

# Directories to skip
SKIP_DIRS = {
    "__pycache__", ".git", ".svn", "node_modules", ".venv", "venv",
    "env", ".env", ".tox", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".eggs", "*.egg-info", ".idea", ".vscode",
    "target", "bin", "obj",
}


@dataclass
class FileState:
    """Tracks the state of a file for change detection."""
    filepath: str
    last_modified: float
    content_hash: str
    size: int


@dataclass
class BrainConfig:
    """Configuration for the Project Brain."""
    project_path: str
    data_dir: str = ""
    watch_interval: float = 2.0  # seconds between file change checks
    auto_watch: bool = False
    max_file_size: int = 1_000_000  # skip files larger than 1MB
    deep_analysis: bool = True  # AST analysis (Python only for now)


class ProjectBrain:
    """
    Persistent, always-aware intelligence layer for a codebase.
    
    Scans the project, builds a dependency graph, and provides
    instant answers about project structure and relationships.
    """

    def __init__(self, project_path: str, config: Optional[BrainConfig] = None):
        self.config = config or BrainConfig(project_path=project_path)
        self.config.project_path = os.path.abspath(project_path)

        if not self.config.data_dir:
            self.config.data_dir = os.path.join(
                str(Path.home()), ".leanai", "brains",
                hashlib.md5(self.config.project_path.encode()).hexdigest()[:12],
            )
        os.makedirs(self.config.data_dir, exist_ok=True)

        # Core components
        self.graph = DependencyGraph()
        self._file_analyses: Dict[str, FileAnalysis] = {}
        self._file_states: Dict[str, FileState] = {}
        self._scan_time: float = 0
        self._last_scan: float = 0

        # Watcher
        self._watching = False
        self._watch_thread: Optional[threading.Thread] = None

        # Load cached state
        self._load_cache()

    # ── Scanning ──────────────────────────────────────────────────

    def scan(self, force: bool = False) -> dict:
        """
        Scan the entire project directory.
        Only re-analyzes files that changed since last scan.
        Returns scan statistics.
        """
        start = time.time()
        project = self.config.project_path

        if not os.path.isdir(project):
            return {"error": f"Not a directory: {project}"}

        files_found = 0
        files_analyzed = 0
        files_unchanged = 0
        files_new = 0

        # Walk the directory
        for root, dirs, files in os.walk(project):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, project)

                # Skip large files
                try:
                    size = os.path.getsize(filepath)
                    if size > self.config.max_file_size:
                        continue
                except OSError:
                    continue

                files_found += 1

                # Check if file changed
                mtime = os.path.getmtime(filepath)
                old_state = self._file_states.get(rel_path)

                if not force and old_state and old_state.last_modified >= mtime:
                    files_unchanged += 1
                    continue

                # Analyze the file
                analysis = analyze_file(filepath)
                self._file_analyses[rel_path] = analysis
                self.graph.add_file_analysis(analysis)

                # Update file state
                content_hash = self._hash_file(filepath)
                self._file_states[rel_path] = FileState(
                    filepath=rel_path,
                    last_modified=mtime,
                    content_hash=content_hash,
                    size=size,
                )

                if old_state is None:
                    files_new += 1
                files_analyzed += 1

        # Resolve cross-file references
        self.graph.resolve_references()

        elapsed = time.time() - start
        self._scan_time = elapsed
        self._last_scan = time.time()

        # Save cache
        self._save_cache()

        return {
            "project": self.config.project_path,
            "files_found": files_found,
            "files_analyzed": files_analyzed,
            "files_unchanged": files_unchanged,
            "files_new": files_new,
            "scan_time_ms": round(elapsed * 1000),
            "graph": self.graph.stats(),
        }

    def rescan_file(self, filepath: str) -> Optional[FileAnalysis]:
        """Re-analyze a single file (called on file change)."""
        if not os.path.exists(filepath):
            rel = os.path.relpath(filepath, self.config.project_path)
            self._file_analyses.pop(rel, None)
            self._file_states.pop(rel, None)
            return None

        analysis = analyze_file(filepath)
        rel = os.path.relpath(filepath, self.config.project_path)
        self._file_analyses[rel] = analysis
        self.graph.add_file_analysis(analysis)
        self.graph.resolve_references()

        mtime = os.path.getmtime(filepath)
        self._file_states[rel] = FileState(
            filepath=rel,
            last_modified=mtime,
            content_hash=self._hash_file(filepath),
            size=os.path.getsize(filepath),
        )
        return analysis

    def _hash_file(self, filepath: str) -> str:
        """Compute content hash of a file."""
        try:
            with open(filepath, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except (FileNotFoundError, IOError):
            return ""

    # ── File Watching ─────────────────────────────────────────────

    def start_watching(self):
        """Start watching for file changes in background."""
        if self._watching:
            return
        self._watching = True
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()

    def stop_watching(self):
        """Stop the file watcher."""
        self._watching = False

    def _watch_loop(self):
        """Background loop that checks for file changes."""
        while self._watching:
            self._check_for_changes()
            time.sleep(self.config.watch_interval)

    def _check_for_changes(self):
        """Check all tracked files for modifications."""
        project = self.config.project_path
        changes = []

        for rel_path, state in list(self._file_states.items()):
            filepath = os.path.join(project, rel_path)
            if not os.path.exists(filepath):
                changes.append(("deleted", rel_path))
                continue
            mtime = os.path.getmtime(filepath)
            if mtime > state.last_modified:
                changes.append(("modified", rel_path))

        for change_type, rel_path in changes:
            filepath = os.path.join(project, rel_path)
            if change_type == "deleted":
                self._file_analyses.pop(rel_path, None)
                self._file_states.pop(rel_path, None)
            else:
                self.rescan_file(filepath)

    # ── Query Interface ───────────────────────────────────────────

    def describe_file(self, filepath: str) -> str:
        """Get a human-readable description of a file's contents."""
        rel = self._resolve_path(filepath)
        analysis = self._file_analyses.get(rel)
        if not analysis:
            return f"File not indexed: {filepath}"

        lines = [f"File: {rel} ({analysis.total_lines} lines)"]
        if analysis.docstring:
            lines.append(f"Purpose: {analysis.docstring[:200]}")
        if analysis.classes:
            lines.append(f"Classes: {', '.join(analysis.class_names)}")
        if analysis.functions:
            for f in analysis.functions:
                sig = f"{f.qualified_name}({', '.join(f.args)})"
                if f.return_type:
                    sig += f" -> {f.return_type}"
                doc = f" — {f.docstring[:80]}" if f.docstring else ""
                lines.append(f"  def {sig}{doc}")
        if analysis.imports:
            mods = list(set(analysis.import_modules))[:10]
            lines.append(f"Imports: {', '.join(mods)}")
        return "\n".join(lines)

    def what_calls(self, function_name: str) -> str:
        """Find what calls a given function."""
        chain = self.graph.get_call_chain(function_name, max_depth=3)
        if not chain:
            return f"Function '{function_name}' not found in project."
        lines = [f"Call chain from {function_name}:"]
        for node_id, depth in chain:
            node = self.graph.nodes.get(node_id)
            if node:
                indent = "  " * depth
                lines.append(f"{indent}→ {node.name} ({os.path.basename(node.filepath)})")
        return "\n".join(lines)

    def what_depends_on(self, filepath: str) -> str:
        """Find all files/functions that depend on a given file."""
        rel = self._resolve_path(filepath)
        dependents = self.graph.get_dependents(rel)
        if not dependents:
            return f"Nothing depends on {filepath}"
        lines = [f"Dependents of {rel}:"]
        for dep_id in dependents:
            if not dep_id.startswith("__"):
                node = self.graph.nodes.get(dep_id)
                name = node.name if node else dep_id
                lines.append(f"  ← {name}")
        return "\n".join(lines)

    def impact_of_changing(self, filepath: str) -> str:
        """Analyze impact of changing a file."""
        rel = self._resolve_path(filepath)
        impact = self.graph.impact_analysis(rel)
        if not impact:
            return f"Changing {filepath} has no detected downstream impact."
        lines = [f"Impact of changing {rel}:"]
        for node_id, distance in sorted(impact.items(), key=lambda x: x[1]):
            node = self.graph.nodes.get(node_id)
            name = node.name if node else node_id
            level = "DIRECT" if distance == 1 else f"depth {distance}"
            lines.append(f"  [{level}] {name}")
        return "\n".join(lines)

    def find_function(self, name: str) -> str:
        """Find a function by name and show its details."""
        node = self.graph.find_function(name)
        if not node:
            return f"Function '{name}' not found."
        lines = [f"Function: {node.name}"]
        lines.append(f"File: {node.filepath}")
        meta = node.metadata
        if meta.get("args"):
            lines.append(f"Args: {', '.join(meta['args'])}")
        if meta.get("return_type"):
            lines.append(f"Returns: {meta['return_type']}")
        if meta.get("docstring"):
            lines.append(f"Docs: {meta['docstring'][:200]}")
        if meta.get("complexity"):
            lines.append(f"Complexity: {meta['complexity']}")
        lines.append(f"Lines: {meta.get('line_start', '?')}-{meta.get('line_end', '?')}")
        return "\n".join(lines)

    def project_summary(self) -> str:
        """Get a high-level summary of the project."""
        stats = self.graph.stats()
        total_lines = sum(a.total_lines for a in self._file_analyses.values())
        languages = {}
        for a in self._file_analyses.values():
            languages[a.language] = languages.get(a.language, 0) + 1

        lines = [
            f"Project: {os.path.basename(self.config.project_path)}",
            f"Path: {self.config.project_path}",
            f"Files: {stats['files']} indexed",
            f"Functions: {stats['functions']}",
            f"Classes: {stats['classes']}",
            f"Total lines: {total_lines:,}",
            f"Edges: {stats['edges']}",
        ]
        if languages:
            lang_str = ", ".join(f"{k}({v})" for k, v in languages.items())
            lines.append(f"Languages: {lang_str}")
        if self._scan_time:
            lines.append(f"Last scan: {self._scan_time*1000:.0f}ms")
        return "\n".join(lines)

    def get_context_for_query(self, query: str) -> str:
        """
        Build rich context for an AI query about the project.
        This is injected into the model prompt for project-aware responses.
        """
        query_lower = query.lower()

        context_parts = [f"Project: {os.path.basename(self.config.project_path)}"]

        # If query mentions a specific file
        for rel_path, analysis in self._file_analyses.items():
            fname = os.path.basename(rel_path).lower()
            if fname in query_lower or rel_path.lower() in query_lower:
                context_parts.append(f"\n{self.describe_file(rel_path)}")

        # If query mentions a function
        for func_name, func_id in self.graph._function_lookup.items():
            if func_name.lower() in query_lower:
                node = self.graph.nodes.get(func_id)
                if node:
                    context_parts.append(f"\n{self.find_function(func_name)}")
                break

        # Always include project summary
        context_parts.append(f"\n{self.project_summary()}")

        return "\n".join(context_parts)

    def _resolve_path(self, filepath: str) -> str:
        """Resolve a filepath to match the keys used in the index.
        Handles both / and \\ separators on all platforms."""
        if os.path.isabs(filepath):
            rel = os.path.relpath(filepath, self.config.project_path)
        else:
            rel = filepath
        # Normalize separators — try both forms
        rel_fwd = rel.replace("\\", "/")
        rel_back = rel.replace("/", "\\")
        # Check which form exists in our index
        if rel in self._file_analyses:
            return rel
        if rel_fwd in self._file_analyses:
            return rel_fwd
        if rel_back in self._file_analyses:
            return rel_back
        # Try os.path.normpath
        normed = os.path.normpath(rel)
        if normed in self._file_analyses:
            return normed
        return rel  # return as-is, caller handles "not found"

    # ── Persistence ───────────────────────────────────────────────

    def _cache_path(self) -> str:
        return os.path.join(self.config.data_dir, "brain_cache.json")

    def _save_cache(self):
        """Save file states to disk for incremental scanning."""
        cache = {
            "project": self.config.project_path,
            "last_scan": self._last_scan,
            "file_states": {
                k: {"filepath": v.filepath, "last_modified": v.last_modified,
                     "content_hash": v.content_hash, "size": v.size}
                for k, v in self._file_states.items()
            },
        }
        try:
            with open(self._cache_path(), "w") as f:
                json.dump(cache, f)
        except Exception:
            pass

    def _load_cache(self):
        """Load cached file states for incremental scanning."""
        path = self._cache_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                cache = json.load(f)
            if cache.get("project") != self.config.project_path:
                return  # different project
            self._last_scan = cache.get("last_scan", 0)
            for k, v in cache.get("file_states", {}).items():
                self._file_states[k] = FileState(**v)
        except (json.JSONDecodeError, Exception):
            pass

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "project": self.config.project_path,
            "files_indexed": len(self._file_analyses),
            "graph": self.graph.stats(),
            "watching": self._watching,
            "last_scan_ms": round(self._scan_time * 1000),
        }

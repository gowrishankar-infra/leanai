"""
M10/M11 — Incremental Sentinel.

M10 re-scanned each changed file on its own and reconciled findings. M11 adds
an optional CROSS-FILE mode: when a file changes, its 1-hop import neighbours
(files it imports + files that import it) are pulled into a single multi-file
scan, so taint paths that cross between those files are detected too — which a
per-file scan structurally cannot do (a path source->sink that spans two files
needs both files in the same scan scope).

Pure orchestration: drives ``SentinelEngine.scan(...)`` and ``MemoryForge`` and
holds no persistent state. Never raises — exceptions are summarised into
``IncrementalResult.error`` (mirrors Watchguard's BatchResult contract).

Modes
-----
* cross_file=False (M10 default): each changed file is scanned alone via
  ``scan(target=path)``; same-file flows only.
* cross_file=True  (M11):         the changed files plus their 1-hop import
  neighbours are scanned together via ``scan(targets=[...])``; cross-file taint
  within that neighbourhood is caught. Neighbour resolution reuses the brain's
  import graph (``brain.graph._file_imports``) and is therefore heuristic, like
  ChainBreaker's reachability — it matches on module/name, not a perfect
  file-path resolution.

Reconciliation (per scanned scope)
----------------------------------
* fingerprint in scan but not on disk -> new       (scan persisted its JSON)
* fingerprint on disk and in scan     -> unchanged
* fingerprint on disk (for a scoped file) but not in scan -> resolved
  (delete JSON + memory_forge.forget_finding -> prune row/edge + log event)

Incremental never invokes the model (use_model=False) — pure-static, cheap
enough to run on every save.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


def _norm(p: str) -> str:
    return str(p).replace("\\", "/") if p else ""


def _fingerprint(filepath: str, line: Any, vuln_class: str) -> str:
    """Reproduce Sentinel's fingerprint exactly: md5(file:line:class)[:10]."""
    return hashlib.md5(
        f"{filepath}:{line}:{vuln_class}".encode()
    ).hexdigest()[:10]


def _module_candidates(rel_norm: str) -> Set[str]:
    """Module/name spellings that an import of this file might use.

    'core/x.py' -> {'x', 'core.x', 'core/x', 'x.py'}  (heuristic — mirrors the
    way ChainBreaker matches imports, since the import graph stores names, not
    resolved file paths)."""
    rel_norm = _norm(rel_norm)
    no_ext = rel_norm[:-3] if rel_norm.endswith(".py") else rel_norm
    stem = no_ext.rsplit("/", 1)[-1]
    dotted = no_ext.replace("/", ".")
    cands = {stem, dotted, no_ext, rel_norm}
    # also the dotted form minus a leading package segment (e.g. 'x' from 'core.x')
    if "." in dotted:
        cands.add(dotted.split(".", 1)[1])
    return {c for c in cands if c}


@dataclass
class IncrementalResult:
    files_scanned: int = 0
    findings_added: int = 0
    findings_resolved: int = 0
    findings_unchanged: int = 0
    cross_file_files: int = 0     # extra neighbour files pulled in (M11)
    skipped_over_cap: bool = False
    error: Optional[str] = None

    @property
    def changed(self) -> bool:
        return bool(self.findings_added or self.findings_resolved)

    def summary_line(self) -> str:
        if self.skipped_over_cap:
            return "sentinel skipped (batch too large)"
        if self.error:
            return "sentinel error"
        if not self.changed:
            return ""
        base = f"sec +{self.findings_added}/-{self.findings_resolved}"
        if self.cross_file_files:
            base += f" (xf+{self.cross_file_files})"
        return base


class IncrementalSentinel:
    """Per-batch incremental scan + graph reconciliation.

    Parameters
    ----------
    sentinel:      SentinelEngine-like (scan, vuln_dir, project_root,
                   _resolve_to_rel; and .brain for cross-file).
    memory_forge:  MemoryForge-like (forget_finding; optional vuln_dir).
    max_files:     scope cap; a larger scope is skipped (suggest /sentinel).
    cross_file:    enable M11 1-hop neighbour expansion + multi-file scan.
    max_hops:      neighbour BFS depth when cross_file (default 1).
    """

    def __init__(self, sentinel: Any, memory_forge: Any, *,
                 max_files: int = 25, cross_file: bool = False,
                 max_hops: int = 1):
        self.sentinel = sentinel
        self.memory_forge = memory_forge
        self.max_files = int(max_files)
        self.cross_file = bool(cross_file)
        self.max_hops = max(1, int(max_hops))

    # ── public entry ────────────────────────────────────────────────
    def process_batch(self, paths: List[str]) -> IncrementalResult:
        result = IncrementalResult()
        try:
            changed: List[str] = []
            seen = set()
            for p in paths:
                if p and p not in seen:
                    seen.add(p)
                    changed.append(p)
            if not changed:
                return result

            if self.cross_file:
                scope, extra = self._expand_neighbours(changed)
                result.cross_file_files = extra
                if self.max_files and len(scope) > self.max_files:
                    result.skipped_over_cap = True
                    return result
                self._process_scope(scope, result)
            else:
                if self.max_files and len(changed) > self.max_files:
                    result.skipped_over_cap = True
                    return result
                for path in changed:
                    self._process_one(path, result)
        except Exception as e:
            result.error = str(e)
        return result

    # ── M10 per-file path (unchanged) ────────────────────────────────
    def _process_one(self, path: str, result: IncrementalResult) -> None:
        rel = _norm(self._resolve_rel(path))
        before = self._fingerprints_on_disk_for({rel})
        findings, _ = self.sentinel.scan(target=path, use_model=False, verbose=False)
        result.files_scanned += 1
        current = {_fingerprint(f.filepath, f.line, f.vuln_class): f.vuln_id
                   for f in findings}
        self._reconcile(before, current, result)

    # ── M11 multi-file scope path ─────────────────────────────────────
    def _process_scope(self, scope: List[str], result: IncrementalResult) -> None:
        scope_rels = {_norm(self._resolve_rel(p)) for p in scope}
        before = self._fingerprints_on_disk_for(scope_rels)
        findings, _ = self.sentinel.scan(targets=scope, use_model=False, verbose=False)
        result.files_scanned += len(scope)
        current = {_fingerprint(f.filepath, f.line, f.vuln_class): f.vuln_id
                   for f in findings}
        self._reconcile(before, current, result)

    def _reconcile(self, before: Dict[str, str], current: Dict[str, str],
                   result: IncrementalResult) -> None:
        for fp in current:
            if fp in before:
                result.findings_unchanged += 1
            else:
                result.findings_added += 1
        for fp, vuln_id in before.items():
            if fp not in current:
                self._resolve(vuln_id)
                result.findings_resolved += 1

    # ── neighbour expansion (M11, M11.1 abs/rel-key fix) ──────────────
    def _expand_neighbours(self, changed: List[str]) -> Tuple[List[str], int]:
        """Return (scope_paths, extra_count). scope = changed + 1-hop import
        neighbours (both dependencies and dependents). Falls back to
        changed-only if no import graph.

        The brain's ``_file_imports`` may be keyed by ABSOLUTE paths (real
        ProjectBrain) or relative ones; we normalise every key to a project
        relative path up front so both lookup directions work regardless.
        (M11 shipped comparing the changed file's *relative* path against
        possibly-absolute keys, which silently resolved only dependents —
        never dependencies. Fixed here.)"""
        graph = getattr(getattr(self.sentinel, "brain", None), "graph", None)
        file_imports = getattr(graph, "_file_imports", None)
        if not file_imports:
            return list(changed), 0

        # Normalise every known file to a project-relative key, once.
        imports_by_rel: Dict[str, Set[str]] = {}
        cands_by_rel: Dict[str, Set[str]] = {}
        for raw_key, imported in file_imports.items():
            rel = _norm(self._resolve_rel(raw_key))
            imports_by_rel[rel] = set(imported)
            cands_by_rel[rel] = _module_candidates(rel)

        changed_rels = {_norm(self._resolve_rel(p)) for p in changed}
        scope_rels: Set[str] = set(changed_rels)

        frontier = deque((r, 0) for r in changed_rels)
        while frontier:
            cur, depth = frontier.popleft()
            if depth >= self.max_hops:
                continue
            cur_imports = imports_by_rel.get(cur, set())
            cur_cands = _module_candidates(cur)

            for rel, f_cands in cands_by_rel.items():
                if rel in scope_rels:
                    continue
                f_imports = imports_by_rel.get(rel, set())
                is_dependency = bool(cur_imports & f_cands)   # cur imports rel
                is_dependent = bool(f_imports & cur_cands)    # rel imports cur
                if is_dependency or is_dependent:
                    scope_rels.add(rel)
                    frontier.append((rel, depth + 1))

        extra = len(scope_rels) - len(changed_rels)
        # Preserve original changed paths; append neighbour rels (scan()'s
        # _resolve_to_rel handles either form).
        scope = list(changed) + [r for r in scope_rels if r not in changed_rels]
        return scope, max(0, extra)

    # ── helpers ──────────────────────────────────────────────────────
    def _resolve_rel(self, path: str) -> str:
        resolver = getattr(self.sentinel, "_resolve_to_rel", None)
        if callable(resolver):
            try:
                return resolver(path)
            except Exception:
                pass
        root = getattr(self.sentinel, "project_root", "") or ""
        try:
            return os.path.relpath(os.path.abspath(path), root)
        except Exception:
            return path

    def _vuln_dir(self) -> Optional[str]:
        return (getattr(self.sentinel, "vuln_dir", None)
                or getattr(self.memory_forge, "vuln_dir", None))

    def _fingerprints_on_disk_for(self, rel_norms: Set[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        vuln_dir = self._vuln_dir()
        if not vuln_dir or not os.path.isdir(vuln_dir):
            return out
        for fpath in glob.glob(os.path.join(vuln_dir, "VULN-*.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            if _norm(data.get("filepath", "")) not in rel_norms:
                continue
            fp = data.get("fingerprint") or _fingerprint(
                data.get("filepath", ""), data.get("line", 0),
                data.get("vuln_class", ""))
            vid = data.get("vuln_id") or os.path.basename(fpath)[:-5]
            out[fp] = vid
        return out

    def _resolve(self, vuln_id: str) -> None:
        vuln_dir = self._vuln_dir()
        if vuln_dir:
            jp = os.path.join(vuln_dir, vuln_id + ".json")
            try:
                if os.path.exists(jp):
                    os.remove(jp)
            except Exception:
                pass
        forget = getattr(self.memory_forge, "forget_finding", None)
        if callable(forget):
            try:
                forget(vuln_id)
            except Exception:
                pass

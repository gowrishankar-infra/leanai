"""
M10 — Incremental Sentinel.

Re-scans only the files changed in a Watchguard batch, reconciles the result
against the existing VULN-*.json store + MemoryForge graph, and reports what
changed. Pure orchestration: it drives ``SentinelEngine.scan(target=...)`` and
``MemoryForge`` and holds no persistent state of its own, so it is trivially
unit-testable with mock collaborators.

Why this exists
---------------
M9 Watchguard keeps the brain + symbol graph fresh on save but deliberately
does NOT re-run Sentinel (a full scan was judged too costly per-save on a
4GB-VRAM box). So findings drift: a vuln you just fixed lingers in
``/memory facts`` and a vuln you just introduced goes unflagged until the next
manual ``/sentinel``. This module closes that drift — for the changed files
only.

What it catches / what it does not
----------------------------------
``SentinelEngine.scan(target=path)`` runs source/sink discovery and taint
tracing over *only that file*. So:

  * Direct pattern vulns, file-level vulns, and **same-file** taint flows are
    detected.
  * **Cross-file** taint paths (source in file A, sink in file B) are NOT
    detected here when only one side changes. That remains the job of a full
    ``/sentinel``.

Incremental never invokes the model (``use_model=False``) — it stays
pure-static so it is cheap enough to run on every save.

Reconciliation contract
------------------------
For each changed file, comparing the set of finding fingerprints stored on
disk *before* the scan against the set produced *by* the scan:

  * fingerprint in scan but not on disk  -> **new**       (scan already wrote
    its VULN-*.json; the subsequent ``memory_forge.sync()`` ingests it)
  * fingerprint on disk and in scan      -> **unchanged** (left as-is)
  * fingerprint on disk but not in scan  -> **resolved**  (delete its
    VULN-*.json and call ``memory_forge.forget_finding`` to prune the graph
    row + edges and log a ``vuln_resolved`` event)

A deleted source file scans to zero findings, so all of its findings resolve —
which is the correct behavior.

Never raises: the contract mirrors Watchguard's BatchResult — all exceptions
are caught and summarized into ``IncrementalResult.error``.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _norm(p: str) -> str:
    """Forward-slash normalize a path (graph + findings use forward slash)."""
    return str(p).replace("\\", "/") if p else ""


def _fingerprint(filepath: str, line: Any, vuln_class: str) -> str:
    """Reproduce Sentinel's fingerprint exactly: md5(file:line:class)[:10].

    Must stay byte-identical to ``SentinelEngine._persist`` /
    ``_assign_ids`` or the diff would never match.
    """
    return hashlib.md5(
        f"{filepath}:{line}:{vuln_class}".encode()
    ).hexdigest()[:10]


@dataclass
class IncrementalResult:
    """Outcome of reconciling one Watchguard batch. Counters are additive
    across the files in the batch."""
    files_scanned: int = 0
    findings_added: int = 0       # new fingerprints (scan persisted them)
    findings_resolved: int = 0    # fingerprints that disappeared (fixed/file gone)
    findings_unchanged: int = 0
    skipped_over_cap: bool = False
    error: Optional[str] = None

    @property
    def changed(self) -> bool:
        return bool(self.findings_added or self.findings_resolved)

    def summary_line(self) -> str:
        """Short fragment for the Watchguard batch summary. Empty string when
        nothing security-relevant happened, so the caller can omit it."""
        if self.skipped_over_cap:
            return "sentinel skipped (batch too large)"
        if self.error:
            return f"sentinel error"
        if not self.changed:
            return ""
        return f"sec +{self.findings_added}/-{self.findings_resolved}"


class IncrementalSentinel:
    """Drives per-file Sentinel scans + graph reconciliation for a batch.

    Parameters
    ----------
    sentinel:
        A ``SentinelEngine`` (or any object exposing ``scan(target=...)``,
        ``vuln_dir``, ``project_root``, and ``_resolve_to_rel``).
    memory_forge:
        A ``MemoryForge`` exposing ``forget_finding(finding_id)`` and,
        optionally, ``vuln_dir``.
    max_files:
        If a single batch touches more than this many files, the incremental
        pass is skipped (the caller should suggest a full ``/sentinel``). This
        keeps a bulk operation like ``git checkout`` from stalling the worker.
        Set to 0 to disable the cap.
    """

    def __init__(self, sentinel: Any, memory_forge: Any, *, max_files: int = 25):
        self.sentinel = sentinel
        self.memory_forge = memory_forge
        self.max_files = int(max_files)

    # ── public entry ────────────────────────────────────────────────
    def process_batch(self, paths: List[str]) -> IncrementalResult:
        """Scan the changed files and reconcile findings. Never raises."""
        result = IncrementalResult()
        try:
            unique: List[str] = []
            seen = set()
            for p in paths:
                if p and p not in seen:
                    seen.add(p)
                    unique.append(p)

            if not unique:
                return result

            if self.max_files and len(unique) > self.max_files:
                result.skipped_over_cap = True
                return result

            for path in unique:
                self._process_one(path, result)
        except Exception as e:  # defensive — contract is never-raise
            result.error = str(e)
        return result

    # ── per file ─────────────────────────────────────────────────────
    def _process_one(self, path: str, result: IncrementalResult) -> None:
        rel = _norm(self._resolve_rel(path))

        before = self._fingerprints_on_disk_for(rel)   # {fingerprint: vuln_id}

        # scan(target=...) re-persists current findings with stable ids.
        findings, _stats = self.sentinel.scan(
            target=path, use_model=False, verbose=False,
        )
        result.files_scanned += 1

        current: Dict[str, str] = {}
        for f in findings:
            fp = _fingerprint(f.filepath, f.line, f.vuln_class)
            current[fp] = f.vuln_id

        for fp in current:
            if fp in before:
                result.findings_unchanged += 1
            else:
                result.findings_added += 1

        for fp, vuln_id in before.items():
            if fp not in current:
                self._resolve(vuln_id)
                result.findings_resolved += 1

    # ── helpers ──────────────────────────────────────────────────────
    def _resolve_rel(self, path: str) -> str:
        """Resolve a changed path to the brain-keyed relative path Sentinel
        uses, so the before/after fingerprint sets share a key space."""
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
        return (
            getattr(self.sentinel, "vuln_dir", None)
            or getattr(self.memory_forge, "vuln_dir", None)
        )

    def _fingerprints_on_disk_for(self, rel_norm: str) -> Dict[str, str]:
        """Map {fingerprint: vuln_id} for findings stored for this file."""
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
            if _norm(data.get("filepath", "")) != rel_norm:
                continue
            fp = data.get("fingerprint")
            if not fp:
                # Recompute if an older JSON lacks the field.
                fp = _fingerprint(
                    data.get("filepath", ""),
                    data.get("line", 0),
                    data.get("vuln_class", ""),
                )
            vid = data.get("vuln_id") or os.path.basename(fpath)[:-5]
            out[fp] = vid
        return out

    def _resolve(self, vuln_id: str) -> None:
        """A finding disappeared: delete its JSON and prune it from the graph."""
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

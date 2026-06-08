#!/usr/bin/env python3
"""
diagnose_crossfile.py — read-only check of M11 cross-file neighbour resolution.

The cross-file incremental scanner (M11) decides, when a file changes, which
1-hop import neighbours to pull into the same scan. That resolution is heuristic
(it matches on module/name, because the brain's import graph stores names, not
resolved file paths). This script shows you EXACTLY what that heuristic resolves
to on your real project, so you can eyeball whether it's right — before trusting
it on live edits.

It is strictly read-only: it builds/scans the brain (same as `/brain .`), then
calls the *real* IncrementalSentinel._expand_neighbours. It never runs a vuln
scan, never writes VULN-*.json, and never touches the memory graph DB.

Usage:
    python tools/diagnose_crossfile.py [PROJECT_PATH] [FILE ...]

Examples:
    # default: project = '.', sample a few files
    python tools/diagnose_crossfile.py

    # check specific files
    python tools/diagnose_crossfile.py . core/sentinel_incremental.py core/watchguard.py
"""

import os
import sys
import tempfile

# Make the repo importable whether run from root or tools/
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from brain.project_brain import ProjectBrain                       # noqa: E402
from core.sentinel import SentinelEngine                           # noqa: E402
from core.sentinel_incremental import (                            # noqa: E402
    IncrementalSentinel, _module_candidates, _norm,
)


class _NoOpForge:
    """Satisfies IncrementalSentinel's interface; never invoked here."""
    vuln_dir = None
    def forget_finding(self, finding_id, *, source_tool="diagnostic"):
        return False


def _classify(changed_rel, neighbour_rel, imports_by_rel, cands_by_rel):
    """Explain WHY a neighbour was pulled in (dependency vs dependent),
    using rel-normalised import data (matches the fixed scanner)."""
    changed_imports = imports_by_rel.get(_norm(changed_rel), set())
    neigh_imports = imports_by_rel.get(_norm(neighbour_rel), set())
    changed_cands = cands_by_rel.get(_norm(changed_rel)) or _module_candidates(changed_rel)
    neigh_cands = cands_by_rel.get(_norm(neighbour_rel)) or _module_candidates(neighbour_rel)

    reasons = []
    hit = changed_imports & neigh_cands
    if hit:
        reasons.append(f"dependency — this file imports {sorted(hit)[:3]}")
    hit2 = neigh_imports & changed_cands
    if hit2:
        reasons.append(f"dependent — it imports {sorted(hit2)[:3]}")
    return "; ".join(reasons) or "matched (heuristic)"


def main():
    args = sys.argv[1:]
    project = args[0] if args else "."
    targets = args[1:]

    project = os.path.abspath(project)
    print(f"[diagnose] Project: {project}")
    print(f"[diagnose] Building brain (read-only)...")
    brain = ProjectBrain(project)
    brain.scan()

    file_imports = getattr(getattr(brain, "graph", None), "_file_imports", None)
    known = sorted(brain._file_analyses.keys())
    print(f"[diagnose] Files indexed: {len(known)}")
    if not file_imports:
        print("[diagnose] WARNING: brain.graph._file_imports is empty — "
              "cross-file mode would fall back to single-file. Stop here.")
        return
    print(f"[diagnose] Files with recorded imports: {len(file_imports)}\n")

    # Real scanner (read-only path); temp vuln_dir so we touch nothing real.
    sentinel = SentinelEngine(brain, vuln_dir=tempfile.mkdtemp())
    incr = IncrementalSentinel(sentinel, _NoOpForge(), cross_file=True)

    # Rel-normalised import maps (keys may be absolute in the real brain).
    imports_by_rel = {}
    cands_by_rel = {}
    for raw_key, imported in file_imports.items():
        rel = _norm(incr._resolve_rel(raw_key))
        imports_by_rel[rel] = set(imported)
        cands_by_rel[rel] = _module_candidates(rel)

    if not targets:
        # Pick a few files that actually have internal neighbours, for signal.
        targets = known[:5]
        print(f"[diagnose] No files given — sampling: "
              f"{', '.join(targets)}\n")

    for tgt in targets:
        rel = _norm(incr._resolve_rel(tgt))
        print(f"=== {tgt} ===")
        print(f"  resolved rel: {rel}")
        if rel not in {_norm(k) for k in brain._file_analyses}:
            print(f"  (not in brain index — would be skipped)\n")
            continue

        scope, extra = incr._expand_neighbours([tgt])
        scope_rels = [_norm(incr._resolve_rel(p)) for p in scope]
        neighbours = [r for r in scope_rels if r != rel]

        if not neighbours:
            print(f"  1-hop neighbours: none (no internal imports resolved)\n")
            continue

        print(f"  1-hop neighbours pulled into scope ({extra}):")
        for n in neighbours:
            why = _classify(rel, n, imports_by_rel, cands_by_rel)
            print(f"    + {n}")
            print(f"        {why}")
        print(f"  -> scan scope would be {len(scope_rels)} file(s)\n")

    print("[diagnose] Done. Nothing was scanned or written.")
    print("[diagnose] Eyeball check: are the neighbours above the files you'd")
    print("           expect taint to flow to/from? Wrong picks = heuristic")
    print("           over-matching on a common module/name.")


if __name__ == "__main__":
    main()

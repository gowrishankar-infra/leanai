"""
LeanAI — Detection Quality Eval
═══════════════════════════════
Runs Sentinel against a fixed corpus of known-vulnerable and known-clean
files (evals/corpus/) and diffs the findings against ground truth
(evals/expected.json).

Why this exists: the SQL-injection detector was once completely dead
while 900+ unit tests stayed green. Unit tests check that code runs;
this eval checks that the scanner DETECTS. Green suite != correct.

Pass criteria (any failure -> exit code 1):
  - RECALL  100% on known-positives: every expected (file, class) pair is
    found, at or above its minimum count.
  - PRECISION 100% on the corpus: clean files produce zero findings, and
    vulnerable files produce no findings outside their expected classes.
  - SOURCES: framework source-check files (Django/FastAPI/Tornado)
    register at least min_sources input sources.

Usage (from the repo root):
    python evals\\run_eval.py            # human-readable report
    python evals\\run_eval.py --json     # also dump machine-readable JSON
    python evals\\run_eval.py -v         # list every finding

Safe by construction: runs against a throwaway LEANAI_HOME and a
throwaway vuln_dir — it never touches ~/.leanai or persisted findings.
"""

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict

# Windows consoles/pipes may use legacy codecs (cp1252) that can't encode
# every character (e.g. in Sentinel finding descriptions). Never let the
# REPORT crash the eval: replace unencodable chars instead.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

# ── Bootstrap: make the repo root importable no matter where we're run from ──
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.join(EVALS_DIR, "corpus")
DEFAULT_EXPECTED = os.path.join(EVALS_DIR, "expected.json")


def _norm(rel_path: str) -> str:
    """Normalize a finding filepath to its basename for corpus matching."""
    return str(rel_path).replace("\\", "/").split("/")[-1]


def run_eval(corpus_dir: str, expected_path: str, verbose: bool = False):
    """Scan the corpus and diff against ground truth.

    Returns (ok: bool, report: dict).
    """
    with open(expected_path, "r", encoding="utf-8") as f:
        expected = json.load(f)["files"]

    # Throwaway home so the eval NEVER touches real ~/.leanai state.
    tmp_home = tempfile.mkdtemp(prefix="leanai-eval-home-")
    prev_home = os.environ.get("LEANAI_HOME")
    os.environ["LEANAI_HOME"] = tmp_home
    try:
        from brain.project_brain import ProjectBrain
        from core.sentinel import SentinelEngine, Severity

        brain = ProjectBrain(corpus_dir)
        brain.scan()

        vuln_dir = os.path.join(tmp_home, "vulns")
        engine = SentinelEngine(brain, vuln_dir=vuln_dir)

        # One full-corpus scan for findings.
        findings, stats = engine.scan(severity_floor=Severity.LOW, verbose=False)

        # Group findings: {basename: {vuln_class: [findings]}}
        by_file = defaultdict(lambda: defaultdict(list))
        for v in findings:
            by_file[_norm(v.filepath)][v.vuln_class].append(v)

        # Per-file source counts for the source_check fixtures.
        source_counts = {}
        for fname, spec in expected.items():
            if spec.get("kind") == "source_check":
                _, fstats = engine.scan(
                    target=fname, severity_floor=Severity.LOW, verbose=False
                )
                source_counts[fname] = fstats.sources_found
    finally:
        if prev_home is not None:
            os.environ["LEANAI_HOME"] = prev_home
        else:
            os.environ.pop("LEANAI_HOME", None)

    # ── Diff against ground truth ──
    misses = []          # recall failures: expected but not found
    false_positives = [] # precision failures: found but not expected
    source_failures = [] # source_check files that registered no sources
    per_class = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})

    corpus_files = set(expected.keys())

    for fname, spec in expected.items():
        kind = spec.get("kind")
        found_here = by_file.get(fname, {})

        if kind == "vulnerable":
            want = spec.get("expected", {})
            # Recall: every expected class present at >= min count.
            for cls, min_count in want.items():
                got = len(found_here.get(cls, []))
                if got >= min_count:
                    per_class[cls]["tp"] += got
                else:
                    per_class[cls]["tp"] += got
                    per_class[cls]["fn"] += (min_count - got)
                    misses.append(
                        f"{fname}: expected >= {min_count} x {cls}, found {got}"
                    )
            # Precision: nothing outside the expected classes.
            for cls, items in found_here.items():
                if cls not in want:
                    per_class[cls]["fp"] += len(items)
                    for v in items:
                        false_positives.append(
                            f"{fname}:{v.line} unexpected {cls} "
                            f"({v.severity}) - {v.description}"
                        )

        elif kind in ("clean", "source_check"):
            # Precision: zero findings of any class.
            for cls, items in found_here.items():
                per_class[cls]["fp"] += len(items)
                for v in items:
                    false_positives.append(
                        f"{fname}:{v.line} FALSE POSITIVE {cls} "
                        f"({v.severity}) - {v.description}"
                    )
            if kind == "source_check":
                need = spec.get("min_sources", 1)
                got = source_counts.get(fname, 0)
                if got < need:
                    source_failures.append(
                        f"{fname}: expected >= {need} input source(s), "
                        f"registered {got} - SOURCE_PATTERNS regressed"
                    )

    # Findings in files we don't even track (shouldn't happen) count as FPs.
    for fname, classes in by_file.items():
        if fname not in corpus_files:
            for cls, items in classes.items():
                per_class[cls]["fp"] += len(items)
                for v in items:
                    false_positives.append(
                        f"{fname}:{v.line} finding outside tracked corpus: {cls}"
                    )

    ok = not misses and not false_positives and not source_failures

    # ── Report ──
    print("=" * 64)
    print(" LeanAI Detection Eval - Sentinel vs ground-truth corpus")
    print("=" * 64)
    print(f" Corpus:   {corpus_dir}")
    print(f" Files:    {stats.files_scanned} scanned, "
          f"{stats.functions_analyzed} functions")
    print(f" Findings: {len(findings)} total")
    print("-" * 64)
    print(f" {'class':<26}{'TP':>5}{'FN':>5}{'FP':>5}{'recall':>9}{'prec':>8}")
    for cls in sorted(per_class):
        c = per_class[cls]
        rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 1.0
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 1.0
        print(f" {cls:<26}{c['tp']:>5}{c['fn']:>5}{c['fp']:>5}"
              f"{rec:>8.0%}{prec:>8.0%}")
    print("-" * 64)
    for fname, got in sorted(source_counts.items()):
        print(f" sources  {fname:<32} registered: {got}")
    print("-" * 64)

    if verbose and findings:
        for v in sorted(findings, key=lambda x: (_norm(x.filepath), x.line)):
            print(f"   {_norm(v.filepath)}:{v.line:<4} {v.vuln_class:<24} "
                  f"{str(v.severity):<8} {v.cwe}")
        print("-" * 64)

    if misses:
        print(" RECALL FAILURES (detector went blind - fix before shipping):")
        for m in misses:
            print(f"   x {m}")
    if false_positives:
        print(" PRECISION FAILURES (false positives on clean code):")
        for fp in false_positives:
            print(f"   x {fp}")
    if source_failures:
        print(" SOURCE-DETECTION FAILURES:")
        for sf in source_failures:
            print(f"   x {sf}")

    print("=" * 64)
    print(f" RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 64)

    report = {
        "ok": ok,
        "total_findings": len(findings),
        "per_class": {k: dict(v) for k, v in per_class.items()},
        "misses": misses,
        "false_positives": false_positives,
        "source_failures": source_failures,
        "source_counts": source_counts,
    }
    return ok, report


def main():
    ap = argparse.ArgumentParser(description="LeanAI Sentinel detection eval")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--expected", default=DEFAULT_EXPECTED)
    ap.add_argument("--json", dest="json_out", nargs="?", const="-",
                    help="write JSON report to a path, or '-' for stdout")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="list every finding")
    args = ap.parse_args()

    ok, report = run_eval(args.corpus, args.expected, verbose=args.verbose)

    if args.json_out:
        payload = json.dumps(report, indent=2)
        if args.json_out == "-":
            print(payload)
        else:
            with open(args.json_out, "w", encoding="utf-8") as f:
                f.write(payload)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

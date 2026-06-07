"""
M10 tests — Incremental Sentinel.

Covers:
  * MemoryForge.forget_finding: prunes row + edges, logs vuln_resolved, returns
    True/False correctly (run against a real temp SQLite graph).
  * IncrementalSentinel diff: new / unchanged / resolved across batches, the
    batch-size cap, and the never-raise contract (run with lightweight fakes).
  * Watchguard: incremental is off by default and a no-op when disabled;
    enable_incremental fails cleanly with no brain/engine.

No watchdog, no model, no real brain required.
"""

import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest

# Make the repo root importable (tests/ -> repo root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.memory_forge import MemoryForge
from core.sentinel_incremental import (
    IncrementalSentinel, IncrementalResult, _fingerprint, _norm,
)


# ── Lightweight fakes ────────────────────────────────────────────────────

class _FakeVuln:
    """Duck-types the fields IncrementalSentinel reads off a Vulnerability."""
    def __init__(self, vuln_id, filepath, line, vuln_class):
        self.vuln_id = vuln_id
        self.filepath = filepath
        self.line = line
        self.vuln_class = vuln_class


class _FakeSentinel:
    """Mimics SentinelEngine just enough: per-file scan that persists
    VULN-*.json exactly like the real _persist (file:line:class fingerprint)
    and returns the finding objects. The test sets `self.results[rel]` to the
    list of findings the next scan of that file should return."""

    def __init__(self, vuln_dir, project_root="/proj"):
        self.vuln_dir = vuln_dir
        self.project_root = project_root
        self.results = {}          # rel_path -> List[_FakeVuln]
        self.scan_calls = []
        self.raise_on_scan = False

    def _resolve_to_rel(self, target):
        # Tests pass rel paths already normalized.
        return _norm(target)

    def scan(self, target=None, severity_floor=None, use_model=False, verbose=True):
        self.scan_calls.append(dict(target=target, use_model=use_model))
        if self.raise_on_scan:
            raise RuntimeError("boom")
        rel = self._resolve_to_rel(target)
        findings = list(self.results.get(rel, []))
        # Persist exactly like the real engine: one JSON per finding.
        for f in findings:
            fp = _fingerprint(f.filepath, f.line, f.vuln_class)
            data = {
                "vuln_id": f.vuln_id, "vuln_class": f.vuln_class,
                "severity": "MEDIUM", "confidence": 0.6,
                "filepath": f.filepath, "function_name": "fn",
                "line": f.line, "fingerprint": fp, "timestamp": time.time(),
            }
            with open(os.path.join(self.vuln_dir, f.vuln_id + ".json"), "w") as fh:
                json.dump(data, fh)
        return findings, object()


class _SpyForge:
    """Records forget_finding calls without a real DB."""
    def __init__(self, vuln_dir):
        self.vuln_dir = vuln_dir
        self.forgotten = []

    def forget_finding(self, finding_id, *, source_tool="sentinel_incremental"):
        self.forgotten.append(finding_id)
        return True


def _write_vuln_json(vuln_dir, vuln_id, filepath, line, vuln_class):
    fp = _fingerprint(filepath, line, vuln_class)
    data = {
        "vuln_id": vuln_id, "vuln_class": vuln_class, "severity": "HIGH",
        "confidence": 0.8, "filepath": filepath, "function_name": "fn",
        "line": line, "fingerprint": fp, "timestamp": time.time(),
    }
    with open(os.path.join(vuln_dir, vuln_id + ".json"), "w") as fh:
        json.dump(data, fh)
    return fp


# ── MemoryForge.forget_finding ───────────────────────────────────────────

class TestForgetFinding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["LEANAI_HOME"] = self.tmp
        self.db = os.path.join(self.tmp, "graph.db")
        self.forge = MemoryForge(project_path=self.tmp, db_path=self.db)

    def _seed_finding_with_edge(self):
        """Insert a symbol, a finding, and a found_in edge directly."""
        now = time.time()
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO symbols(name, kind, filepath, line, last_sync) "
                "VALUES (?,?,?,?,?)",
                ("core/x.py::foo", "function", "core/x.py", 10, now),
            )
            sym_id = c.execute("SELECT id FROM symbols").fetchone()[0]
            c.execute(
                "INSERT INTO findings"
                "(finding_id, kind, category, severity, confidence, filepath, "
                " line, fingerprint, created_at, last_sync) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("VULN-2026-0001", "vuln", "sql_injection", "HIGH", 0.8,
                 "core/x.py", 10, "abc123", now, now),
            )
            fid = c.execute("SELECT id FROM findings").fetchone()[0]
            c.execute(
                "INSERT INTO relations"
                "(src_kind, src_id, dst_kind, dst_id, relation, weight, last_sync) "
                "VALUES ('finding', ?, 'symbol', ?, 'found_in', 1.0, ?)",
                (fid, sym_id, now),
            )
        return fid

    def test_forget_removes_row_edge_and_logs_event(self):
        self._seed_finding_with_edge()
        ok = self.forge.forget_finding("VULN-2026-0001")
        self.assertTrue(ok)
        with sqlite3.connect(self.db) as c:
            self.assertEqual(
                c.execute("SELECT COUNT(*) FROM findings").fetchone()[0], 0)
            self.assertEqual(
                c.execute("SELECT COUNT(*) FROM relations "
                          "WHERE relation='found_in'").fetchone()[0], 0)
            ev = c.execute(
                "SELECT kind, source_tool FROM events "
                "WHERE kind='vuln_resolved'").fetchone()
        self.assertIsNotNone(ev)
        self.assertEqual(ev[1], "sentinel_incremental")

    def test_forget_unknown_returns_false(self):
        self.assertFalse(self.forge.forget_finding("VULN-9999-9999"))

    def test_forget_does_not_touch_other_findings(self):
        self._seed_finding_with_edge()
        now = time.time()
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO findings"
                "(finding_id, kind, category, severity, confidence, filepath, "
                " line, fingerprint, created_at, last_sync) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("VULN-2026-0002", "vuln", "xss", "LOW", 0.5,
                 "core/y.py", 5, "def456", now, now),
            )
        self.forge.forget_finding("VULN-2026-0001")
        with sqlite3.connect(self.db) as c:
            remaining = c.execute(
                "SELECT finding_id FROM findings").fetchall()
        self.assertEqual([r[0] for r in remaining], ["VULN-2026-0002"])


# ── IncrementalSentinel diff/reconcile ───────────────────────────────────

class TestIncrementalDiff(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vuln_dir = os.path.join(self.tmp, "vulns")
        os.makedirs(self.vuln_dir)
        self.sentinel = _FakeSentinel(self.vuln_dir)
        self.forge = _SpyForge(self.vuln_dir)
        self.incr = IncrementalSentinel(self.sentinel, self.forge, max_files=25)

    def test_new_finding_counted_added(self):
        self.sentinel.results["core/a.py"] = [
            _FakeVuln("VULN-2026-0001", "core/a.py", 12, "sql_injection")]
        r = self.incr.process_batch(["core/a.py"])
        self.assertEqual(r.findings_added, 1)
        self.assertEqual(r.findings_resolved, 0)
        self.assertEqual(r.findings_unchanged, 0)
        self.assertEqual(r.files_scanned, 1)

    def test_unchanged_finding_not_recounted(self):
        # Pre-existing JSON on disk for this file (already-known finding).
        _write_vuln_json(self.vuln_dir, "VULN-2026-0001", "core/a.py", 12, "sql_injection")
        self.sentinel.results["core/a.py"] = [
            _FakeVuln("VULN-2026-0001", "core/a.py", 12, "sql_injection")]
        r = self.incr.process_batch(["core/a.py"])
        self.assertEqual(r.findings_added, 0)
        self.assertEqual(r.findings_unchanged, 1)
        self.assertEqual(r.findings_resolved, 0)
        self.assertEqual(self.forge.forgotten, [])

    def test_resolved_finding_pruned_and_json_deleted(self):
        # Known finding on disk; scan now returns nothing (vuln fixed).
        _write_vuln_json(self.vuln_dir, "VULN-2026-0007", "core/a.py", 30, "weak_crypto")
        self.sentinel.results["core/a.py"] = []   # fixed
        r = self.incr.process_batch(["core/a.py"])
        self.assertEqual(r.findings_resolved, 1)
        self.assertEqual(r.findings_added, 0)
        self.assertEqual(self.forge.forgotten, ["VULN-2026-0007"])
        self.assertFalse(os.path.exists(
            os.path.join(self.vuln_dir, "VULN-2026-0007.json")))

    def test_mixed_new_and_resolved_same_file(self):
        _write_vuln_json(self.vuln_dir, "VULN-2026-0001", "core/a.py", 12, "sql_injection")
        # old one (line 12) gone, a new one (line 40) appears
        self.sentinel.results["core/a.py"] = [
            _FakeVuln("VULN-2026-0009", "core/a.py", 40, "path_traversal")]
        r = self.incr.process_batch(["core/a.py"])
        self.assertEqual(r.findings_added, 1)
        self.assertEqual(r.findings_resolved, 1)
        self.assertEqual(self.forge.forgotten, ["VULN-2026-0001"])

    def test_other_files_findings_are_not_resolved(self):
        # A finding for a DIFFERENT file must not be touched when a.py changes.
        _write_vuln_json(self.vuln_dir, "VULN-2026-0050", "core/b.py", 3, "xss")
        self.sentinel.results["core/a.py"] = []
        r = self.incr.process_batch(["core/a.py"])
        self.assertEqual(r.findings_resolved, 0)
        self.assertEqual(self.forge.forgotten, [])
        self.assertTrue(os.path.exists(
            os.path.join(self.vuln_dir, "VULN-2026-0050.json")))

    def test_batch_cap_skips(self):
        incr = IncrementalSentinel(self.sentinel, self.forge, max_files=2)
        r = incr.process_batch(["a.py", "b.py", "c.py"])
        self.assertTrue(r.skipped_over_cap)
        self.assertEqual(r.files_scanned, 0)
        self.assertEqual(self.sentinel.scan_calls, [])

    def test_never_raises_on_scan_error(self):
        self.sentinel.raise_on_scan = True
        self.sentinel.results["core/a.py"] = []
        r = self.incr.process_batch(["core/a.py"])
        self.assertIsNotNone(r.error)   # captured, not raised

    def test_summary_line_omitted_when_no_change(self):
        self.sentinel.results["core/a.py"] = []
        r = self.incr.process_batch(["core/a.py"])
        self.assertEqual(r.summary_line(), "")

    def test_summary_line_present_on_change(self):
        self.sentinel.results["core/a.py"] = [
            _FakeVuln("VULN-2026-0001", "core/a.py", 12, "sql_injection")]
        r = self.incr.process_batch(["core/a.py"])
        self.assertIn("+1", r.summary_line())


# ── Watchguard integration: off by default ───────────────────────────────

class _FakeBrain:
    def rescan_file(self, path):
        return None


class _FakeForgeWG:
    def __init__(self):
        self.synced = 0
        self.vuln_dir = tempfile.mkdtemp()

    def set_brain(self, b):
        pass

    def sync(self, verbose=False):
        self.synced += 1

    def stats(self):
        return {"symbols": 0, "findings": 0, "relations": 0}

    def forget_finding(self, fid, *, source_tool="sentinel_incremental"):
        return True


class TestWatchguardIncrementalOffByDefault(unittest.TestCase):
    def setUp(self):
        from core.watchguard import Watchguard
        self.tmp = tempfile.mkdtemp()
        self.wg = Watchguard(
            project_path=self.tmp,
            brain=_FakeBrain(),
            memory_forge=_FakeForgeWG(),
            leanai_home=self.tmp,
        )

    def test_disabled_by_default(self):
        self.assertFalse(self.wg.incremental_enabled)

    def test_batch_does_not_scan_when_disabled(self):
        # Without a sentinel and disabled, the batch should still process the
        # rescan + sync and never touch incremental state.
        res = self.wg._handle_events_batch([os.path.join(self.tmp, "x.py")])
        self.assertEqual(self.wg.status().incr_files_scanned, 0)
        self.assertEqual(res.incr_summary, "")

    def test_enable_without_brain_or_engine_fails_clean(self):
        from core.watchguard import Watchguard
        wg = Watchguard(
            project_path=self.tmp, brain=None,
            memory_forge=_FakeForgeWG(), leanai_home=self.tmp,
        )
        self.assertFalse(wg.enable_incremental(True))
        self.assertFalse(wg.incremental_enabled)


if __name__ == "__main__":
    unittest.main(verbosity=2)

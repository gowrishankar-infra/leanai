"""
M11 tests — three independent features:

  A) IncrementalSentinel cross-file mode: 1-hop import-neighbour expansion and
     multi-file scope scan/reconcile.
  B) FindingsReport: rollup, dedup, SARIF, markdown.
  C) LeanAIConfig: defaults, validation, persistence, corrupt-file recovery.

Lightweight fakes; no watchdog / model / real brain.
"""

import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.sentinel_incremental import (
    IncrementalSentinel, _module_candidates, _fingerprint, _norm,
)
from core.findings_report import FindingsReport, Rollup
from core.leanai_config import LeanAIConfig, DEFAULTS


# ── shared fakes ─────────────────────────────────────────────────────────

class _FakeVuln:
    def __init__(self, vuln_id, filepath, line, vuln_class):
        self.vuln_id = vuln_id
        self.filepath = filepath
        self.line = line
        self.vuln_class = vuln_class


class _FakeGraph:
    def __init__(self, file_imports):
        self._file_imports = file_imports   # filepath -> set(import name strings)


class _FakeBrain:
    def __init__(self, file_imports):
        self.graph = _FakeGraph(file_imports)
        self._file_analyses = {k: object() for k in file_imports}


class _FakeSentinel:
    """scan(targets=[...]) multi-file OR scan(target=...) single; persists JSON
    like the real engine and returns the configured findings for the scope."""
    def __init__(self, vuln_dir, brain=None, project_root="/proj"):
        self.vuln_dir = vuln_dir
        self.brain = brain
        self.project_root = project_root
        self.results = {}            # rel -> [ _FakeVuln ]
        self.scan_scopes = []        # records each scan's scope

    def _resolve_to_rel(self, target):
        return _norm(target)

    def _persist(self, findings):
        for f in findings:
            fp = _fingerprint(f.filepath, f.line, f.vuln_class)
            with open(os.path.join(self.vuln_dir, f.vuln_id + ".json"), "w") as fh:
                json.dump({
                    "vuln_id": f.vuln_id, "vuln_class": f.vuln_class,
                    "severity": "HIGH", "confidence": 0.8,
                    "filepath": f.filepath, "function_name": "fn",
                    "line": f.line, "fingerprint": fp, "timestamp": time.time(),
                }, fh)

    def scan(self, target=None, targets=None, severity_floor=None,
             use_model=False, verbose=True):
        if targets is not None:
            rels = [self._resolve_to_rel(t) for t in targets]
        else:
            rels = [self._resolve_to_rel(target)]
        self.scan_scopes.append(sorted(set(rels)))
        findings = []
        for rel in set(rels):
            findings.extend(self.results.get(rel, []))
        self._persist(findings)
        return findings, object()


class _SpyForge:
    def __init__(self, vuln_dir):
        self.vuln_dir = vuln_dir
        self.forgotten = []

    def forget_finding(self, finding_id, *, source_tool="sentinel_incremental"):
        self.forgotten.append(finding_id)
        return True


def _write_vuln(vuln_dir, vuln_id, filepath, line, vuln_class):
    fp = _fingerprint(filepath, line, vuln_class)
    with open(os.path.join(vuln_dir, vuln_id + ".json"), "w") as fh:
        json.dump({
            "vuln_id": vuln_id, "vuln_class": vuln_class, "severity": "HIGH",
            "confidence": 0.8, "filepath": filepath, "function_name": "fn",
            "line": line, "fingerprint": fp, "timestamp": time.time(),
        }, fh)
    return fp


# ── A: cross-file ────────────────────────────────────────────────────────

class TestCrossFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vuln_dir = os.path.join(self.tmp, "vulns")
        os.makedirs(self.vuln_dir)
        # b.py imports a (dependency of b); c.py imports nothing relevant.
        self.imports = {
            "core/a.py": set(),
            "core/b.py": {"a", "core.a"},     # b imports a
            "core/c.py": {"os"},
        }
        self.brain = _FakeBrain(self.imports)
        self.sentinel = _FakeSentinel(self.vuln_dir, brain=self.brain)
        self.forge = _SpyForge(self.vuln_dir)

    def test_module_candidates(self):
        c = _module_candidates("core/a.py")
        self.assertIn("a", c)
        self.assertIn("core.a", c)

    def test_expansion_pulls_in_dependents(self):
        # a.py changes; b.py imports a -> b should be pulled into scope.
        incr = IncrementalSentinel(self.sentinel, self.forge, cross_file=True)
        incr.process_batch(["core/a.py"])
        scope = self.sentinel.scan_scopes[-1]
        self.assertIn("core/a.py", scope)
        self.assertIn("core/b.py", scope)   # dependent
        self.assertNotIn("core/c.py", scope)

    def test_expansion_pulls_in_dependencies(self):
        # b.py changes; b imports a -> a should be pulled in as a dependency.
        incr = IncrementalSentinel(self.sentinel, self.forge, cross_file=True)
        incr.process_batch(["core/b.py"])
        scope = self.sentinel.scan_scopes[-1]
        self.assertIn("core/b.py", scope)
        self.assertIn("core/a.py", scope)

    def test_cross_file_off_scans_single_file(self):
        incr = IncrementalSentinel(self.sentinel, self.forge, cross_file=False)
        incr.process_batch(["core/a.py"])
        scope = self.sentinel.scan_scopes[-1]
        self.assertEqual(scope, ["core/a.py"])   # no neighbours

    def test_cross_file_detects_new_finding_across_scope(self):
        # A taint vuln that only exists when a+b scanned together.
        self.sentinel.results["core/b.py"] = [
            _FakeVuln("VULN-2026-0100", "core/b.py", 20, "sql_injection")]
        incr = IncrementalSentinel(self.sentinel, self.forge, cross_file=True)
        r = incr.process_batch(["core/a.py"])
        self.assertEqual(r.findings_added, 1)
        self.assertGreaterEqual(r.cross_file_files, 1)

    def test_no_import_graph_falls_back(self):
        s = _FakeSentinel(self.vuln_dir, brain=None)
        incr = IncrementalSentinel(s, self.forge, cross_file=True)
        incr.process_batch(["core/a.py"])
        self.assertEqual(s.scan_scopes[-1], ["core/a.py"])

    def test_scope_cap_skips(self):
        incr = IncrementalSentinel(self.sentinel, self.forge,
                                   cross_file=True, max_files=1)
        r = incr.process_batch(["core/b.py"])   # expands to 2 -> over cap
        self.assertTrue(r.skipped_over_cap)
        self.assertEqual(self.sentinel.scan_scopes, [])


# ── B: findings report ───────────────────────────────────────────────────

class TestFindingsReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vuln_dir = os.path.join(self.tmp, "vulns")
        os.makedirs(self.vuln_dir)
        _write_vuln(self.vuln_dir, "VULN-1", "a.py", 10, "sql_injection")
        _write_vuln(self.vuln_dir, "VULN-2", "a.py", 20, "xss")
        _write_vuln(self.vuln_dir, "VULN-3", "b.py", 5, "sql_injection")

    def test_rollup_counts(self):
        r = FindingsReport(self.vuln_dir).load().rollup()
        self.assertEqual(r.total, 3)
        self.assertEqual(r.by_category["sql_injection"], 2)
        self.assertEqual(r.by_file["a.py"], 2)

    def test_dedup_collapses_identical(self):
        # second JSON with identical fingerprint key
        _write_vuln(self.vuln_dir, "VULN-1-dup", "a.py", 10, "sql_injection")
        r = FindingsReport(self.vuln_dir).load()
        roll = r.rollup()
        self.assertEqual(roll.total, 3)            # not 4
        self.assertEqual(roll.duplicates_collapsed, 1)

    def test_sarif_structure(self):
        sarif = FindingsReport(self.vuln_dir).load().to_sarif()
        self.assertEqual(sarif["version"], "2.1.0")
        run = sarif["runs"][0]
        self.assertEqual(run["tool"]["driver"]["name"], "LeanAI Sentinel")
        self.assertEqual(len(run["results"]), 3)
        self.assertEqual(run["results"][0]["level"], "error")  # HIGH -> error
        self.assertIn("startLine",
                      run["results"][0]["locations"][0]["physicalLocation"]["region"])

    def test_markdown_contains_sections(self):
        md = FindingsReport(self.vuln_dir).load().to_markdown()
        self.assertIn("# LeanAI Security Findings Report", md)
        self.assertIn("By severity", md)
        self.assertIn("sql_injection", md)

    def test_write_files(self):
        rep = FindingsReport(self.vuln_dir).load()
        out = os.path.join(self.tmp, "reports")
        md_path = rep.write(out, "markdown")
        sarif_path = rep.write(out, "sarif")
        self.assertTrue(md_path.endswith(".md") and os.path.exists(md_path))
        self.assertTrue(sarif_path.endswith(".sarif") and os.path.exists(sarif_path))
        with open(sarif_path) as fh:
            json.load(fh)   # valid JSON

    def test_empty_dir(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        self.assertEqual(FindingsReport(empty).load().rollup().total, 0)


# ── C: config ─────────────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "config.json")

    def test_defaults_when_missing(self):
        c = LeanAIConfig(self.path)
        self.assertEqual(c.get("snippet_limit"), DEFAULTS["snippet_limit"])
        self.assertFalse(c.get("incremental_cross_file"))

    def test_set_and_persist(self):
        c = LeanAIConfig(self.path)
        self.assertTrue(c.set("snippet_limit", 12000))
        self.assertTrue(os.path.exists(self.path))
        c2 = LeanAIConfig(self.path)
        self.assertEqual(c2.get("snippet_limit"), 12000)

    def test_validation_rejects_bad(self):
        c = LeanAIConfig(self.path)
        self.assertFalse(c.set("snippet_limit", 5))         # below min
        self.assertFalse(c.set("snippet_limit", "huge"))    # wrong type
        self.assertFalse(c.set("unknown_key", 1))           # unknown
        self.assertEqual(c.get("snippet_limit"), DEFAULTS["snippet_limit"])

    def test_bool_setting(self):
        c = LeanAIConfig(self.path)
        self.assertTrue(c.set("incremental_cross_file", True))
        self.assertTrue(LeanAIConfig(self.path).get("incremental_cross_file"))

    def test_corrupt_file_recovers(self):
        with open(self.path, "w") as fh:
            fh.write("{ this is not json ")
        c = LeanAIConfig(self.path)   # must not raise
        self.assertEqual(c.get("snippet_limit"), DEFAULTS["snippet_limit"])

    def test_all_returns_copy(self):
        c = LeanAIConfig(self.path)
        d = c.all()
        d["snippet_limit"] = 999
        self.assertNotEqual(c.get("snippet_limit"), 999)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
M12 tests — SecurityAudit (unified audit + combined report).

Exercises load/dedup, summary counts, two-run SARIF, Markdown structure,
severity normalisation, write(), and empty-input behaviour. Pure file I/O —
no brain/model/engine.
"""

import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.security_audit import SecurityAudit, AuditSummary, _norm_sev


def _write_vuln(vdir, vid, filepath, line, cls, sev="HIGH"):
    with open(os.path.join(vdir, vid + ".json"), "w") as fh:
        json.dump({
            "vuln_id": vid, "vuln_class": cls, "severity": sev,
            "confidence": 0.8, "filepath": filepath, "function_name": "fn",
            "line": line, "fingerprint": f"{filepath}:{line}:{cls}"[:10],
            "timestamp": time.time(),
        }, fh)


def _write_chain(cdir, cid, entry_vuln, capability, sev="Severity.CRITICAL",
                 steps=None, conf=0.7):
    steps = steps or [
        {"function_id": "a", "function_name": "handler", "filepath": "core/a.py",
         "line": 10, "stage": "entry", "indicators": ["user_input"]},
        {"function_id": "b", "function_name": "run", "filepath": "core/b.py",
         "line": 40, "stage": "rce", "indicators": ["os.system"]},
    ]
    with open(os.path.join(cdir, cid + ".json"), "w") as fh:
        json.dump({
            "chain_id": cid, "entry_vuln_id": entry_vuln, "severity": sev,
            "confidence": conf, "capability": capability, "steps": steps,
            "narrative": f"Attacker reaches {capability} via {len(steps)} hops.",
            "impact_summary": f"Remote {capability}.",
            "fix_recommendation": "Validate and sanitise input.",
            "fingerprint": "abc123", "timestamp": time.time(),
        }, fh)


class TestSecurityAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vdir = os.path.join(self.tmp, "vulns")
        self.cdir = os.path.join(self.tmp, "chains")
        os.makedirs(self.vdir)
        os.makedirs(self.cdir)
        _write_vuln(self.vdir, "VULN-1", "core/a.py", 10, "sql_injection", "HIGH")
        _write_vuln(self.vdir, "VULN-2", "core/b.py", 40, "command_injection", "CRITICAL")
        _write_chain(self.cdir, "CHAIN-2026-0001", "VULN-2", "rce")

    def test_norm_sev(self):
        self.assertEqual(_norm_sev("Severity.HIGH"), "HIGH")
        self.assertEqual(_norm_sev("critical"), "CRITICAL")

    def test_summary_counts(self):
        s = SecurityAudit(self.vdir, self.cdir).load().summary()
        self.assertEqual(s.vuln_total, 2)
        self.assertEqual(s.chain_total, 1)
        self.assertEqual(s.vuln_by_severity["CRITICAL"], 1)
        self.assertEqual(s.chain_by_capability["rce"], 1)
        self.assertEqual(s.chain_by_severity["CRITICAL"], 1)

    def test_chain_dedup_by_id(self):
        # Re-writing same chain id must not double-count.
        _write_chain(self.cdir, "CHAIN-2026-0001", "VULN-2", "rce")
        s = SecurityAudit(self.vdir, self.cdir).load().summary()
        self.assertEqual(s.chain_total, 1)

    def test_sarif_two_runs(self):
        sarif = SecurityAudit(self.vdir, self.cdir).load().to_sarif()
        self.assertEqual(sarif["version"], "2.1.0")
        self.assertEqual(len(sarif["runs"]), 2)
        names = [r["tool"]["driver"]["name"] for r in sarif["runs"]]
        self.assertIn("LeanAI Sentinel", names)
        self.assertIn("LeanAI ChainBreaker", names)
        chain_run = sarif["runs"][1]
        self.assertEqual(len(chain_run["results"]), 1)
        res = chain_run["results"][0]
        self.assertEqual(res["level"], "error")          # CRITICAL -> error
        self.assertEqual(res["properties"]["capability"], "rce")
        # second step becomes a relatedLocation
        self.assertEqual(len(res["relatedLocations"]), 1)
        self.assertEqual(
            res["locations"][0]["physicalLocation"]["artifactLocation"]["uri"],
            "core/a.py")

    def test_markdown_sections(self):
        md = SecurityAudit(self.vdir, self.cdir).load().to_markdown()
        self.assertIn("# LeanAI Security Audit", md)
        self.assertIn("Executive summary", md)
        self.assertIn("Exploit chains", md)
        self.assertIn("CHAIN-2026-0001", md)
        self.assertIn("Vulnerabilities", md)
        self.assertIn("VULN-2", md)

    def test_write_both(self):
        out = os.path.join(self.tmp, "reports")
        paths = SecurityAudit(self.vdir, self.cdir).load().write(out, "both")
        self.assertEqual(len(paths), 2)
        sarif_path = [p for p in paths if p.endswith(".sarif")][0]
        with open(sarif_path) as fh:
            json.load(fh)   # valid JSON
        self.assertTrue(any(p.endswith(".md") for p in paths))

    def test_write_single_format(self):
        out = os.path.join(self.tmp, "reports2")
        paths = SecurityAudit(self.vdir, self.cdir).load().write(out, "sarif")
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith(".sarif"))

    def test_empty_inputs(self):
        e = os.path.join(self.tmp, "empty")
        os.makedirs(os.path.join(e, "vulns"))
        os.makedirs(os.path.join(e, "chains"))
        a = SecurityAudit(os.path.join(e, "vulns"),
                          os.path.join(e, "chains")).load()
        s = a.summary()
        self.assertEqual(s.vuln_total, 0)
        self.assertEqual(s.chain_total, 0)
        # still produces a valid report
        self.assertIn("No exploit chains", a.to_markdown())
        self.assertEqual(len(a.to_sarif()["runs"]), 2)

    def test_missing_chain_dir(self):
        a = SecurityAudit(self.vdir, os.path.join(self.tmp, "nope")).load()
        self.assertEqual(a.summary().chain_total, 0)
        self.assertEqual(a.summary().vuln_total, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

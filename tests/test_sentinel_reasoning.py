"""
Tests for Sentinel's model-primary reasoning pass (scan(reason=True)).

Covers: _parse_reasoning robustness, false-positive suppression, attaching
verdict/reasoning/fix, confidence adjustment, default-unchanged behavior, and
graceful no-op / never-raise contracts. No real model — a stub model_fn.
"""

import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.sentinel import SentinelEngine, Severity


def _make_project(root):
    with open(os.path.join(root, "svc.py"), "w") as f:
        f.write(textwrap.dedent("""\
            import os
            def run_user_cmd(cmd):
                os.system("sh -c " + cmd)            # exploitable: arg from caller
            def safe_cleanup():
                os.system("rm -rf /tmp/leanai_cache") # constant arg: not exploitable
        """))


class TestParseReasoning(unittest.TestCase):
    def test_full_structured(self):
        p = SentinelEngine._parse_reasoning(
            "VERDICT: exploitable\nCONFIDENCE: 0.9\n"
            "REASONING: tainted input reaches os.system.\nFIX: use shell=False.")
        self.assertEqual(p["verdict"], "exploitable")
        self.assertAlmostEqual(p["confidence"], 0.9)
        self.assertIn("tainted", p["reasoning"])
        self.assertIn("shell=False", p["fix"])

    def test_not_exploitable_variants(self):
        for txt in ["VERDICT: not_exploitable\n", "VERDICT: no, this is safe\n",
                    "not exploitable — constant argument"]:
            self.assertEqual(SentinelEngine._parse_reasoning(txt)["verdict"],
                             "not_exploitable")

    def test_confidence_percent_normalised(self):
        p = SentinelEngine._parse_reasoning("VERDICT: exploitable\nCONFIDENCE: 85")
        self.assertAlmostEqual(p["confidence"], 0.85)

    def test_empty_and_garbage(self):
        self.assertEqual(SentinelEngine._parse_reasoning("")["verdict"], "")
        # garbage shouldn't raise
        SentinelEngine._parse_reasoning("\u2603 random text with no fields")


class TestReasoningPass(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["LEANAI_HOME"] = os.path.join(self.tmp, "home")
        self.proj = os.path.join(self.tmp, "proj")
        os.makedirs(self.proj)
        _make_project(self.proj)
        from brain.project_brain import ProjectBrain
        self.brain = ProjectBrain(self.proj)
        self.brain.scan()

    def _engine(self, model_fn=None):
        return SentinelEngine(
            self.brain, model_fn=model_fn,
            vuln_dir=os.path.join(self.tmp, "vulns"))

    def test_default_scan_unchanged(self):
        f, _ = self._engine().scan(severity_floor=Severity.LOW, verbose=False)
        self.assertTrue(len(f) >= 2)
        self.assertTrue(all(not v.model_verdict for v in f))  # no model fields set

    def test_false_positive_suppressed_and_real_kept(self):
        def model_fn(prompt):
            if "safe_cleanup" in prompt:
                return ("VERDICT: not_exploitable\nCONFIDENCE: 0.95\n"
                        "REASONING: constant argument, not user input.\nFIX: none.")
            return ("VERDICT: exploitable\nCONFIDENCE: 0.9\n"
                    "REASONING: cmd flows unsanitized to os.system.\n"
                    "FIX: subprocess.run([...], shell=False).")
        f, _ = self._engine(model_fn).scan(
            severity_floor=Severity.LOW, reason=True, verbose=False)
        names = {v.function_name for v in f}
        self.assertNotIn("safe_cleanup", names)         # FP dropped
        real = [v for v in f if v.function_name == "run_user_cmd"]
        self.assertEqual(len(real), 1)
        self.assertEqual(real[0].model_verdict, "exploitable")
        self.assertGreaterEqual(real[0].confidence, 0.85)
        self.assertIn("shell=False", real[0].fix_suggestion)   # contextual fix used
        self.assertTrue(real[0].model_reasoning)

    def test_no_model_is_noop(self):
        # reason=True but no model_fn -> behaves like a normal scan, nothing dropped
        f, _ = self._engine(None).scan(
            severity_floor=Severity.LOW, reason=True, verbose=False)
        self.assertTrue(len(f) >= 2)
        self.assertTrue(all(not v.model_verdict for v in f))

    def test_model_exception_keeps_findings(self):
        def boom(prompt):
            raise RuntimeError("model down")
        f, _ = self._engine(boom).scan(
            severity_floor=Severity.LOW, reason=True, verbose=False)
        self.assertTrue(len(f) >= 2)   # nothing dropped; never raised

    def test_uncertain_leaves_finding(self):
        def model_fn(prompt):
            return "VERDICT: uncertain\nREASONING: needs review.\nFIX: review manually."
        f, _ = self._engine(model_fn).scan(
            severity_floor=Severity.LOW, reason=True, verbose=False)
        self.assertTrue(len(f) >= 2)   # uncertain not dropped
        self.assertTrue(any(v.model_verdict == "uncertain" for v in f))

    def test_context_includes_function_and_callers(self):
        eng = self._engine(lambda p: "VERDICT: uncertain")
        f, _ = eng.scan(severity_floor=Severity.LOW, verbose=False)
        ctx = eng._build_finding_context(f[0])
        self.assertIn("FUNCTION", ctx)
        self.assertIn("os.system", ctx)   # the function body is present


if __name__ == "__main__":
    unittest.main(verbosity=2)

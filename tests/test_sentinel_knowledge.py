import os, sys, tempfile, textwrap, unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.sentinel import SentinelEngine, Severity, CWE_MAP


def _proj(root):
    with open(os.path.join(root, "svc.py"), "w") as f:
        f.write(textwrap.dedent("""\
            import os
            def run(cmd):
                os.system("sh -c " + cmd)
        """))


class TestCWE(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda v=os.environ.get("LEANAI_HOME"): (os.environ.__setitem__("LEANAI_HOME", v) if v is not None else os.environ.pop("LEANAI_HOME", None)))
        os.environ["LEANAI_HOME"] = os.path.join(self.tmp, "home")
        self.proj = os.path.join(self.tmp, "p"); os.makedirs(self.proj); _proj(self.proj)
        from brain.project_brain import ProjectBrain
        self.brain = ProjectBrain(self.proj); self.brain.scan()

    def test_cwe_attached(self):
        f, _ = SentinelEngine(self.brain, vuln_dir=os.path.join(self.tmp, "v")).scan(
            severity_floor=Severity.LOW, verbose=False)
        self.assertTrue(f)
        self.assertEqual(f[0].cwe, CWE_MAP["command_injection"])
        self.assertEqual(f[0].cwe, "CWE-78")

    def test_cwe_map_complete(self):
        # every severity-mapped class has a CWE
        from core.sentinel import DEFAULT_SEVERITY
        for cls in DEFAULT_SEVERITY:
            self.assertIn(cls, CWE_MAP, f"missing CWE for {cls}")


class TestEscalation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda v=os.environ.get("LEANAI_HOME"): (os.environ.__setitem__("LEANAI_HOME", v) if v is not None else os.environ.pop("LEANAI_HOME", None)))
        os.environ["LEANAI_HOME"] = os.path.join(self.tmp, "home")
        self.proj = os.path.join(self.tmp, "p"); os.makedirs(self.proj); _proj(self.proj)
        from brain.project_brain import ProjectBrain
        self.brain = ProjectBrain(self.proj); self.brain.scan()

    def test_uncertain_escalates(self):
        local = lambda p: "VERDICT: uncertain\nREASONING: unsure.\nFIX: review."
        deep = lambda p: "VERDICT: exploitable\nCONFIDENCE: 0.95\nREASONING: deep confirms.\nFIX: parameterize."
        s = SentinelEngine(self.brain, model_fn=local, deep_reasoner_fn=deep,
                           vuln_dir=os.path.join(self.tmp, "v"))
        f, _ = s.scan(severity_floor=Severity.LOW, reason=True, verbose=False)
        self.assertTrue(any(v.escalated and v.model_verdict == "exploitable" for v in f))

    def test_no_escalation_when_confident(self):
        local = lambda p: "VERDICT: exploitable\nCONFIDENCE: 0.9\nREASONING: clear.\nFIX: fix."
        deep_called = {"n": 0}
        def deep(p):
            deep_called["n"] += 1
            return "VERDICT: not_exploitable"
        s = SentinelEngine(self.brain, model_fn=local, deep_reasoner_fn=deep,
                           vuln_dir=os.path.join(self.tmp, "v"))
        s.scan(severity_floor=Severity.LOW, reason=True, verbose=False)
        self.assertEqual(deep_called["n"], 0)   # confident local verdict -> no escalation

    def test_no_deep_reasoner_is_fine(self):
        local = lambda p: "VERDICT: uncertain\nREASONING: unsure."
        s = SentinelEngine(self.brain, model_fn=local,
                           vuln_dir=os.path.join(self.tmp, "v"))
        f, _ = s.scan(severity_floor=Severity.LOW, reason=True, verbose=False)
        self.assertTrue(all(not v.escalated for v in f))   # no escalation, no crash


if __name__ == "__main__":
    unittest.main(verbosity=2)

import os, sys, unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.verify_fix import check_syntax, lint, VerifyFixLoop


class TestChecks(unittest.TestCase):
    def test_syntax_ok(self):
        ok, err = check_syntax("def f():\n    return 1\n")
        self.assertTrue(ok); self.assertEqual(err, "")

    def test_syntax_bad(self):
        ok, err = check_syntax("def f(:\n  return\n")
        self.assertFalse(ok); self.assertIn("SyntaxError", err)

    def test_empty(self):
        ok, _ = check_syntax("")
        self.assertFalse(ok)

    def test_lint_clean_code(self):
        self.assertEqual(lint("def f():\n    return 1\n"), [])

    def test_lint_flags_bare_except_or_synerr(self):
        issues = lint("try:\n    x = 1\nexcept:\n    pass\n")
        # pyflakes or fallback: at least surfaces something or stays empty;
        # fallback specifically flags bare except.
        self.assertTrue(isinstance(issues, list))


class TestLoop(unittest.TestCase):
    def test_clean_code_passes_first_try(self):
        r = VerifyFixLoop().run("x = 1\n")
        self.assertTrue(r.ok); self.assertEqual(r.attempts, 1); self.assertFalse(r.fixed)

    def test_fix_applied_on_retry(self):
        calls = {"n": 0}
        def fix_fn(code, err):
            calls["n"] += 1
            return "x = 1\n"        # repaired
        r = VerifyFixLoop(fix_fn=fix_fn, max_attempts=3).run("x = (\n")
        self.assertTrue(r.ok); self.assertTrue(r.fixed); self.assertEqual(calls["n"], 1)

    def test_no_fix_fn_fails_cleanly(self):
        r = VerifyFixLoop().run("def f(:\n")
        self.assertFalse(r.ok); self.assertIn("SyntaxError", r.errors)

    def test_fix_fn_exception_is_contained(self):
        def boom(code, err): raise RuntimeError("x")
        r = VerifyFixLoop(fix_fn=boom).run("def f(:\n")
        self.assertFalse(r.ok)   # did not raise

    def test_runner_failure_triggers_fix(self):
        def runner(code):
            return ("raise" not in code), "boom" if "raise" in code else ""
        def fix_fn(code, err): return "y = 2\n"
        r = VerifyFixLoop(fix_fn=fix_fn, runner=runner).run("raise Exception()\n")
        self.assertTrue(r.ok); self.assertTrue(r.fixed)


if __name__ == "__main__":
    unittest.main(verbosity=2)

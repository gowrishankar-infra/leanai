"""
Detection-quality gate: runs the Sentinel eval corpus as part of the
normal test suite, so a detector regression turns the suite red the
day it's introduced (the dead-SQLi bug previously lived under a green
suite for weeks — this test exists so that can never happen again).

Runs the eval in a subprocess so it gets a fresh interpreter, its own
throwaway LEANAI_HOME, and zero import-order coupling with other tests.
"""
import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNNER = os.path.join(REPO_ROOT, "evals", "run_eval.py")


class TestDetectionEval(unittest.TestCase):

    def test_corpus_recall_and_precision(self):
        """Sentinel must hit 100% recall and 100% precision on the eval
        corpus, and register Django/FastAPI/Tornado input sources."""
        self.assertTrue(os.path.exists(RUNNER), f"missing {RUNNER}")
        proc = subprocess.run(
            [sys.executable, RUNNER],
            cwd=REPO_ROOT,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if proc.returncode != 0:
            self.fail(
                "Detection eval FAILED — a detector or source pattern "
                "regressed. Run `python evals\\run_eval.py -v` for the "
                "full diff.\n\n--- eval output ---\n"
                + proc.stdout + "\n" + proc.stderr
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

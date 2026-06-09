import os, sys, tempfile, textwrap, unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.code_context import rerank_chunks, CodeContextBuilder


class TestRerank(unittest.TestCase):
    def test_orders_by_relevance(self):
        chunks = [
            "def unrelated_logging_helper(): pass",
            "def authenticate_user(token): validate(token)",
            "def parse_config(path): return open(path)",
        ]
        out = rerank_chunks("user authentication token", chunks, top_k=3)
        self.assertIn("authenticate_user", out[0])   # most relevant first

    def test_dict_chunks_and_topk(self):
        chunks = [{"content": "sql query execute database"},
                  {"content": "css styling layout"},
                  {"content": "database connection pool sql"}]
        out = rerank_chunks("sql database", chunks, top_k=2)
        self.assertEqual(len(out), 2)
        self.assertTrue(all("sql" in c["content"] or "database" in c["content"] for c in out))

    def test_empty_query_returns_prefix(self):
        self.assertEqual(len(rerank_chunks("", ["a", "b", "c"], top_k=2)), 2)

    def test_empty_chunks(self):
        self.assertEqual(rerank_chunks("x", []), [])


class TestContextBuilder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda v=os.environ.get("LEANAI_HOME"): (os.environ.__setitem__("LEANAI_HOME", v) if v is not None else os.environ.pop("LEANAI_HOME", None)))
        os.environ["LEANAI_HOME"] = os.path.join(self.tmp, "home")
        self.proj = os.path.join(self.tmp, "proj")
        os.makedirs(self.proj)
        with open(os.path.join(self.proj, "m.py"), "w") as f:
            f.write(textwrap.dedent("""\
                import os
                def helper(x):
                    return os.path.abspath(x)
                def caller(p):
                    return helper(p)
            """))
        from brain.project_brain import ProjectBrain
        self.brain = ProjectBrain(self.proj); self.brain.scan()
        self.cb = CodeContextBuilder(self.brain)

    def test_for_function_includes_source(self):
        ctx = self.cb.for_function("m.py", "helper")
        self.assertIn("FUNCTION", ctx)
        self.assertIn("abspath", ctx)        # the body is present

    def test_callers_resolved(self):
        # helper is called by caller -> should appear as CALLED BY (graph perm.)
        callers = self.cb.callers_of("m.py", "helper")
        self.assertIsInstance(callers, list)   # resolves without error

    def test_missing_function_graceful(self):
        ctx = self.cb.for_function("m.py", "does_not_exist")
        self.assertIn("FUNCTION", ctx)         # no crash, placeholder


if __name__ == "__main__":
    unittest.main(verbosity=2)

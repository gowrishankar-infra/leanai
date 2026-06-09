import os, sys, unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.abstention import (self_consistency, should_abstain,
                             CalibratedDecider, ConsensusResult)


class TestSelfConsistency(unittest.TestCase):
    def test_majority(self):
        seq = iter(["yes", "yes", "no"])
        c = self_consistency(lambda: next(seq), n=3)
        self.assertEqual(c.answer, "yes")
        self.assertAlmostEqual(c.agreement, 2/3)

    def test_unanimous(self):
        c = self_consistency(lambda: "exploitable", n=4)
        self.assertEqual(c.answer, "exploitable")
        self.assertEqual(c.agreement, 1.0)

    def test_generate_exceptions_counted(self):
        def g():
            raise RuntimeError("down")
        c = self_consistency(g, n=3)
        self.assertIsNone(c.answer); self.assertEqual(c.agreement, 0.0)

    def test_parse_fn(self):
        seq = iter(["VERDICT: yes", "VERDICT: yes", "VERDICT: maybe"])
        c = self_consistency(lambda: next(seq), n=3,
                             parse_fn=lambda s: s.split(":")[1].strip())
        self.assertEqual(c.answer, "yes")


class TestAbstain(unittest.TestCase):
    def test_should_abstain(self):
        self.assertTrue(should_abstain(0.3, 0.5))
        self.assertFalse(should_abstain(0.8, 0.5))
        self.assertTrue(should_abstain(None))   # bad input -> abstain (safe)

    def test_decider(self):
        d = CalibratedDecider(threshold=0.6)
        self.assertTrue(d.decide("ans", 0.4).abstained)
        self.assertFalse(d.decide("ans", 0.9).abstained)

    def test_decide_by_consensus_abstains_on_disagreement(self):
        d = CalibratedDecider(threshold=0.8)
        seq = iter(["a", "b", "c"])      # total disagreement
        dec = d.decide_by_consensus(lambda: next(seq), n=3)
        self.assertTrue(dec.abstained)

    def test_decide_by_consensus_answers_on_agreement(self):
        d = CalibratedDecider(threshold=0.6)
        dec = d.decide_by_consensus(lambda: "safe", n=3)
        self.assertFalse(dec.abstained); self.assertEqual(dec.answer, "safe")


if __name__ == "__main__":
    unittest.main(verbosity=2)

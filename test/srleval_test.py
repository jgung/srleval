import unittest

import pkg_resources

from srleval.srleval import evaluate


class TestChunk(unittest.TestCase):

    def test_all_correct(self):
        self._evaluate("resources/gold.txt", "resources/pred1.txt", 3, 8, 25, 0, 0)

    def test_wrong_spans(self):
        self._evaluate("resources/gold.txt", "resources/pred2.txt", 3, 8, 22, 3, 3)

    def test_wrong_labels(self):
        self._evaluate("resources/gold.txt", "resources/pred3.txt", 3, 8, 23, 2, 2)

    def test_extra_spans(self):
        self._evaluate("resources/gold.txt", "resources/pred4.txt", 3, 8, 25, 0, 1)

    def _evaluate(self, gold, pred, ns, ntargets, okay, missed, excess):
        gold_path = pkg_resources.resource_filename(__name__, gold)
        pred_path = pkg_resources.resource_filename(__name__, pred)
        eval_results = evaluate(gold_path, pred_path)
        self.assertEqual(ns, eval_results.ns)
        self.assertEqual(ntargets, eval_results.ntargets)
        self.assertEqual(okay, eval_results.evaluation.ok)
        self.assertEqual(missed, eval_results.evaluation.ms)
        self.assertEqual(excess, eval_results.evaluation.op)

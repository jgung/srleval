import unittest

import pkg_resources

from srleval.srleval import conll_iterator, evaluate


class TestChunk(unittest.TestCase):

    def test_all_correct(self):
        self._evaluate("resources/gold.txt", "resources/pred1.txt", 3, 8, 17, 0, 0)

    def test_wrong_spans(self):
        self._evaluate("resources/gold.txt", "resources/pred2.txt", 3, 8, 14, 3, 3)

    def test_wrong_labels(self):
        self._evaluate("resources/gold.txt", "resources/pred3.txt", 3, 8, 15, 2, 2)

    def test_extra_spans(self):
        self._evaluate("resources/gold.txt", "resources/pred4.txt", 3, 8, 17, 0, 1)

    def _evaluate(self, gold, pred, ns, ntargets, okay, missed, excess):
        gold_path = pkg_resources.resource_filename(__name__, gold)
        pred_path = pkg_resources.resource_filename(__name__, pred)
        gold_props, pred_props = conll_iterator(gold_path), conll_iterator(pred_path)
        eval_results = evaluate(gold_props, pred_props)
        self.assertEqual(ns, eval_results.ns)
        self.assertEqual(ntargets, eval_results.ntargets)
        self.assertEqual(okay, eval_results.evaluation.ok)
        self.assertEqual(missed, eval_results.evaluation.ms)
        self.assertEqual(excess, eval_results.evaluation.op)

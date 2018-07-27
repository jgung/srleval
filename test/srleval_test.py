import unittest

import pkg_resources

from srleval.srleval import conll_iterator, evaluate, get_perl_output, options


class TestEval(unittest.TestCase):

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

    def _get_script_output(self, args, gold, pred, expected):
        gold_path = pkg_resources.resource_filename(__name__, gold)
        pred_path = pkg_resources.resource_filename(__name__, pred)
        expected_path = pkg_resources.resource_filename(__name__, expected)
        _opts = options([*args, '--gold', gold_path, '--pred', pred_path])
        output = get_perl_output(_opts.gold, _opts.pred, _opts.latex, _opts.confusions)
        with open(expected_path, 'r') as f:
            for expected, actual in zip(f, output.split('\n')):
                self.assertEquals(expected.strip(), actual.strip())

    def test_all_correct(self):
        self._evaluate("resources/gold.txt", "resources/pred.perfect.txt", 3, 8, 17, 0, 0)

    def test_wrong_spans(self):
        self._evaluate("resources/gold.txt", "resources/pred.wrong.spans.txt", 3, 8, 14, 3, 3)

    def test_wrong_labels(self):
        self._evaluate("resources/gold.txt", "resources/pred.wrong.labels.txt", 3, 8, 15, 2, 2)

    def test_extra_spans(self):
        self._evaluate("resources/gold.txt", "resources/pred.extra.txt", 3, 8, 17, 0, 1)

    def test_default_output(self):
        self._get_script_output([], 'resources/1.test.wsj', 'resources/2.test.wsj', 'resources/default.output.txt')

    def test_latex_confusions(self):
        self._get_script_output(['--latex', '-C'], 'resources/1.test.wsj', 'resources/2.test.wsj',
                                'resources/latex.confusions.txt')

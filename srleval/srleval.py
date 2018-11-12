import argparse
import re
from collections import Counter, defaultdict

NON_TARGET_TOKEN = '-'
SE_START = "("
SE_CONT = "*"
SE_END = ")"
CONTINUATION_PATTERN = re.compile("^C-")
START_TAG_PATTERN = re.compile("^\(([^*(]+)")
END_TAG_PATTERN = re.compile("^([^)]*)\)")
OKAY_KEY = "ok"
EXCESS_KEY = "op"
MISS_KEY = "ms"
NONE = "-NONE-"
VERB = "V"


class Evaluation(object):
    def __init__(self):
        self.ok = 0
        self.op = 0
        self.ms = 0
        self.types = defaultdict(Counter)
        self.excluded = defaultdict(Counter)
        self.ptv = 0
        self.confusions = defaultdict(Counter)

    def prec_rec_f1(self):
        """
        Compute the precision, recall and F1 score for this evaluation.
        :return: tuple containing (precision, recall, F1)
        """
        return Evaluation.precrecf1(self.ok, self.op, self.ms)

    def increment_ok(self):
        self.ok += 1

    def increment_op(self):
        self.op += 1

    def increment_ms(self):
        self.ms += 1

    def update_confusion_matrix(self, gprop, pprop):
        ok, ms, op, eq = SrlProp.discriminate_args(gprop, pprop, False)
        for pred, gold in zip(ok, eq):
            self.confusions[gold.label][pred.label] += 1
        for pred in ms:
            self.confusions[pred.label][NONE] += 1
        for pred in op:
            self.confusions[NONE][pred.label] += 1

    def display_confusion_matrix(self):
        uok, uop, ums, uacc = 0, 0, 0, 0
        for gold_label, label_counter in self.confusions.items():
            if gold_label in [NONE, VERB]:
                continue
            for pred_label in [item for item in self.confusions.keys() if item not in [NONE, VERB]]:
                uok += label_counter[pred_label]
            uacc += label_counter[gold_label]
            ums += label_counter[NONE]
        for pred_label in [item for item in self.confusions[NONE].keys() if item not in [NONE, VERB]]:
            uop += self.confusions[NONE][pred_label]
        lines = ["--------------------------------------------------------------------",
                 "{:>10}   {:>6}  {:>6}  {:>6}   {:>6}  {:>6}  {:>6}  {:>6}".format(
                     "", "corr.", "excess", "missed", "prec.", "rec.", "F1", "lAcc"),
                 "{:>10}   {:>6}  {:>6}  {:>6}   {:>6.2f}  {:>6.2f}  {:>6.2f}  {:>6.2f}".format(
                     "Unlabeled", uok, uop, ums, *Evaluation.precrecf1(uok, uop, ums), 100 * uacc / uok),
                 "--------------------------------------------------------------------",
                 "\n---- Confusion Matrix: (one row for each correct role, with the distribution of predictions)"]

        all_keys = set(self.confusions.keys())
        for val in self.confusions.values():
            all_keys = all_keys.union(val.keys())
        keys = sorted(all_keys)

        vals = ["            "]
        for i, gold_label in enumerate(keys):
            vals.append("{:>4}".format(i - 1))
        lines.append(" ".join(vals))
        for i, gold_label in enumerate(keys):
            vals = ["{:>2}: {:<8}".format(i - 1, gold_label)]
            for pred_label in keys:
                vals.append("{:>4}".format(self.confusions[gold_label][pred_label]))
            lines.append(" ".join(vals))
        return "\n".join(lines)

    def __str__(self) -> str:
        max_len = len(max({*self.types.keys(), *self.excluded.keys()}, key=lambda x: len(x)))

        def _format(val, ok, op, ms, prec, rec, f1):
            return '{:>{max_len}}   {:>6}  {:>6}  {:>6}   {:>6.2f}  {:>6.2f}  {:>6.2f}'.format(val, ok, op, ms, prec, rec, f1,
                                                                                               max_len=max_len)

        summary = _format("Overall", self.ok, self.op, self.ms, *self.prec_rec_f1())
        line = '-' * len(summary)
        lines = [
            '{:>{max_len}}   {:>6}  {:>6}  {:>6}   {:>6}  {:>6}  {:>6}'.format("", "corr.", "excess", "missed", "prec.", "rec.",
                                                                               "F1", max_len=max_len),
            line,
            summary,
            '-' * max_len
        ]

        for label in sorted(self.types.keys()):
            count = self.types[label]
            lines.append(_format(label, count[OKAY_KEY], count[EXCESS_KEY], count[MISS_KEY],
                                 *Evaluation.precrecf1(count[OKAY_KEY], count[EXCESS_KEY], count[MISS_KEY])))
        lines.append(line)
        for label in sorted(self.excluded.keys()):
            count = self.excluded[label]
            lines.append(_format(label, count[OKAY_KEY], count[EXCESS_KEY], count[MISS_KEY],
                                 *Evaluation.precrecf1(count[OKAY_KEY], count[EXCESS_KEY], count[MISS_KEY])))
        lines.append(line)

        return '\n'.join(lines)

    @staticmethod
    def precrecf1(ok, op, ms):
        """
        Compute the precision, recall and F1 for given TP, FP, and FN.
        :param ok: true positives
        :param op: false positives
        :param ms: false negatives
        :return: tuple containing (precision, recall, F1)
        """
        precision = 100 * ok / (ok + op) if ok + op > 0 else 0
        recall = 100 * ok / (ok + ms) if ok + ms > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1


class SrlEvaluation(object):
    def __init__(self, ns=0, ntargets=0, evaluation=Evaluation()):
        self.ns = ns
        self.ntargets = ntargets
        self.evaluation = evaluation

    def confusion_matrix(self):
        return self.evaluation.display_confusion_matrix()

    def latex(self):
        def _format_latex(val, prec, rec, f1):
            return '{:<10} & {:>6.2f}\\% & {:>6.2f}\\% & {:>6.2f}\\\\'.format(val, prec, rec, f1)

        lines = ['\\begin{table}[t]',
                 '\\centering',
                 '\\begin{tabular}{|l|r|r|r|}\\cline{2-4}',
                 '\\multicolumn{1}{l|}{}',
                 '           & Precision & Recall & F$_{\\beta=1}$\\\\',
                 '\\hline',
                 _format_latex('Overall', *self.evaluation.prec_rec_f1()),
                 '\\hline'
                 ]
        for label, label_counts in sorted(self.evaluation.types.items(), key=lambda item: item[0]):
            lines.append(_format_latex(label, *self.evaluation.precrecf1(label_counts[OKAY_KEY], label_counts[EXCESS_KEY],
                                                                         label_counts[MISS_KEY])))
        lines.append('\\hline')
        if self.evaluation.excluded:
            lines.append('\\hline')
            for label, label_counts in sorted(self.evaluation.excluded.items(), key=lambda item: item[0]):
                lines.append(_format_latex(label, *self.evaluation.precrecf1(label_counts[OKAY_KEY], label_counts[EXCESS_KEY],
                                                                             label_counts[MISS_KEY])))
            lines.append('\\hline')
        lines.append('\\end{tabular}')
        lines.append('\\end{table}')
        return '\n'.join(lines)

    def __str__(self) -> str:
        lines = ['Number of Sentences    :      {:>6}'.format(self.ns),
                 'Number of Propositions :      {:>6}'.format(self.ntargets),
                 "Percentage of perfect props : {:>6.2f}".format(
                     100 * self.evaluation.ptv / self.ntargets if self.ntargets > 0 else 0), '',
                 str(self.evaluation)]
        return '\n'.join(lines)


class SrlSentence(object):

    def __init__(self, sid):
        self.sid = sid
        self.words = None
        self.gold = {}
        self.pred = {}
        self.chunks = None
        self.clauses = None
        self.tree = None
        self.ne = None

    def length(self):
        return self.words if isinstance(self.words, int) else len(self.words)

    def word(self, i):
        return self.words[i]

    @staticmethod
    def load_props(sent_id, columns):
        targets = [target for target in columns[0] if target != NON_TARGET_TOKEN]
        if len(targets) != len(columns) - 1:
            raise RuntimeError('Sentence {}: mismatch in number of targets and arg columns: {} vs. {}'
                               .format(sent_id, len(targets), len(columns) - 1))
        results = {}
        for i, target in zip(range(1, len(columns)), targets):
            column = columns[i]
            prop = SrlProp(target, i).load_se_tagging(column)
            results[i - 1] = prop
        return results


class SrlProp(object):
    def __init__(self, verb, position, sense=None, args=None):
        if not args:
            args = []

        self.verb = verb
        self.position = position
        self.sense = sense
        self.args = args

    def load_se_tagging(self, tags):
        phrase_set = PhraseSet().load_se_tagging(tags)

        args = {}  # store args per type, to be able to continue them
        # add each phrase as an argument, with special treatment for multi-phrase arguments (A C-A C-A)
        for a in phrase_set.phrases():
            continuation_match = CONTINUATION_PATTERN.match(a.label)
            # the phrase continues a started arg
            if continuation_match:
                label = continuation_match.string[continuation_match.end():]
                if label in args:
                    pc = a
                    a = args[label]
                    if a.single:
                        # create the head phrase, considered arg until now
                        a.add_phrases(SrlPhrase(start=a.start, end=a.end, label=a.label))
                    a.add_phrases(pc)
                    a.end = pc.end
                else:
                    # turn the phrase into an arg
                    a = SrlArg(a.start, a.end, label)
                    # push @{$prop->[3]}, $a;
                    self.args.append(a)
                    args[a.label] = a
            else:
                # turn the phrase into an arg
                a = SrlArg(a.start, a.end, a.label)
                # push @{$prop->[3]}, $a;
                self.args.append(a)
                args[a.label] = a
        return self

    @staticmethod
    def discriminate_args(pa, pb, check_type=True):
        """
        Discriminates the args of prop $pb wrt the args of prop $pa, returning intersection(a^b), a-b and b-a returns a tuple
        containing three lists (ok, ms, op):
        ok : args in $pa and $pb
        ms : args in $pa and not in $pb
        op : args in $pb and not in $pa
        """
        args = defaultdict(dict)
        eq = []
        ok = []
        ms = []
        op = []
        for a in pa.args:
            args[a.start][a.end] = a
        for a in pb.args:
            s = a.start
            e = a.end
            gold = args[s].get(e)
            if not gold:
                op.append(a)
            elif gold.single() and a.single():
                if not check_type or gold.label == a.label:
                    if not check_type:
                        eq.append(gold)
                    ok.append(a)
                    del args[s][e]
                else:
                    op.append(a)
            elif not gold.single() and a.single():
                op.append(a)
            elif gold.single() and not a.single():
                op.append(a)
            else:
                # Check phrases of arg
                okay = not check_type or gold.label == a.label
                phrase_dict = {}
                if okay:
                    for gph in gold.phrases:
                        phrase_dict['{}.{}'.format(gph.start, gph.end)] = 1
                for p in a.phrases:
                    pkey = '{}.{}'.format(p.start, p.end)
                    if pkey in phrase_dict:
                        del phrase_dict[pkey]
                    else:
                        okay = False
                        break
                if okay and len(phrase_dict) == 0:
                    if not check_type:
                        eq.append(gold)
                    ok.append(a)
                    del args[s][e]
                else:
                    op.append(a)
        for s in args.keys():
            for a in args[s].values():
                ms.append(a)
        return ok, ms, op, eq

    def __str__(self) -> str:
        return "[{}@{}: {} ]".format(self.verb, self.position, " ".join([str(arg) for arg in self.args]))


class PhraseSet(object):
    phrase_types = None

    def __init__(self, phrases=None):
        self._phrases = defaultdict(dict)
        self.sentence_length = 0
        if phrases:
            self.add_phrases(phrases)

    def load_se_tagging(self, tags, phrase_types=None):
        """
        Adds phrases represented in Start-End tagging. Receives a list of Start-End tags (one per word in the sentence).
        Creates a phrase object for each phrase in the tagging and modifies the set so that the phrases are part of it.
        :param tags: list of SRL labels
        :param phrase_types: phrase labels to consider
        :return: list of phrases
        """
        current = []
        for wid, token in enumerate(tags):
            while not token.startswith(SE_CONT):
                match = START_TAG_PATTERN.match(token)
                if not match:
                    raise RuntimeError("Opening nodes -- bad format in {} at {}-th position!".format(token, wid))
                label = match.group(1)  # corresponds to type in original script
                token = match.string[match.end():]
                if not phrase_types or phrase_types[label]:
                    current.append(SrlPhrase(start=wid, label=label))

            token = token.replace(SE_CONT, "")
            while token:
                match = END_TAG_PATTERN.match(token)
                if not match:
                    raise RuntimeError("Closing phrases -- bad format in {}!".format(token))
                label = match.group(1)
                token = match.string[match.end():]
                if not label or not phrase_types or phrase_types[label]:
                    a = current.pop()
                    if label and label != a.label:
                        raise RuntimeError("Types do not match: {} vs. {}".format(label, a.label))
                    a.end = wid
                    self.add_phrases(a)

        if current:
            raise RuntimeError("Some phrases are unclosed!")
        return self

    def add_phrases(self, phrase):
        for phrase in phrase.dfs():
            self._phrases[phrase.start][phrase.end] = phrase
            if phrase.end >= self.sentence_length:
                self.sentence_length = phrase.end + 1

    def size(self):
        """
        Returns the number of phrases in the set.
        """
        n = 0
        for val in self._phrases.values():
            n += len(val)
        return n

    def phrase(self, start, end):
        """
        Returns the phrase starting at word position start and ending at end, or None if it doesn't exist.
        :param start: starting word position
        :param end: ending word position
        :return: phrase with start and end positions or None if not found
        """
        return self._phrases[start].get(end)

    def phrases(self, start=0, end=None):
        """
        Returns phrases in the set, recursively in depth first search order that is, if a phrase is returned, all its sub phrases
        are also returned. If no parameters, returns all phrases. If a pair of positions is given (start, end), returns phrases
        included within the start and end positions.
        """
        results = []
        if not end:
            end = self.sentence_length
        for i in range(start, end):
            if i in self._phrases:
                for j in range(end, start - 1, -1):
                    phrase = self._phrases[i].get(j)
                    if phrase:
                        results.append(phrase)
        return results

    def top_phrases(self, start=0, end=None):
        """
        Returns phrases in the set, non-recursively in sequential order that is, if a phrase is returned, its subphrases are not
        returned. If no parameters, returns all phrases. If a pair of positions is given (start, end), returns phrases included
        within the start and end positions.
        """
        results = []
        if not end:
            end = self.sentence_length
        i = start
        while i < end:
            for j in range(end, start - 1):
                if i in self._phrases and j in self._phrases[i]:
                    results.append(self._phrases[i][j])
                    i = j
                    break
            i += 1
        return results

    def __str__(self) -> str:
        return " ".join([str(phrase) for phrase in self.top_phrases()])


class SrlPhrase(object):

    def __init__(self, start, end=-1, label=None):
        super().__init__()
        # start word index
        self.start = start
        # end word index
        self.end = end
        # phrase type
        self.label = label
        # sub phrases
        self.phrases = []

    def add_phrases(self, phrases):
        self.phrases.append(phrases)

    def dfs(self):
        """
        Returns the phrases rooted in the current phrase in DFS order.
        """
        return _dfs(self, lambda val: val.phrases)

    def __str__(self) -> str:
        result = "({}{}{})".format(self.start,
                                   self.phrases and " " + " ".join([str(phrase) for phrase in self.phrases]) + " " or " ",
                                   self.end)

        if self.label:
            result += '_{}'.format(self.label)
        return result


def _dfs(root, child_func):
    """
    Returns a list of elements in DFS order from a given root node.
    :param root: root element
    :param child_func: mapping to children of each element
    :return: DFS ordered list of elements including root
    """
    visited, stack = set(), [root]
    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            stack.extend([child for child in child_func(current) if child not in visited])
    return list(visited)


class SrlArg(SrlPhrase):

    def __init__(self, start, end=-1, label=None):
        super().__init__(start, end, label)

    def single(self):
        return len(self.phrases) == 0


def conll_iterator(conll_path):
    """
    Generator-based iterator over a CoNLL formatted file (line per token, blank lines separating sentences).
    :param conll_path:  CoNLL formatted file
    :return: iterator over sentences in file
    """
    with open(conll_path, 'r') as lines:
        current = defaultdict(list)
        for line in lines:
            line = line.strip()
            if not line:
                if current:
                    yield current
                    current = defaultdict(list)
                continue
            for i, val in enumerate(line.split()):
                current[i].append(val)
        if current:  # read last instance if there is no newline at end of file
            yield current


def _evaluate_proposition(gprop, pprop, exclusions=None):
    if exclusions is None:
        exclusions = {}

    def excluded(val):
        return val in exclusions

    ok, ms, op, eq = SrlProp.discriminate_args(gprop, pprop)
    e = Evaluation()

    def update_counts(counts, count_key, incr_func):
        for a in counts:
            if not excluded(a.label):
                incr_func()
                e.types[a.label][count_key] += 1
            else:
                e.excluded[a.label][count_key] += 1

    update_counts(ok, OKAY_KEY, e.increment_ok)
    update_counts(ms, MISS_KEY, e.increment_ms)
    update_counts(op, EXCESS_KEY, e.increment_op)

    e.ptv = 1 if not e.op and not e.ms else 0
    return e


def evaluate(gold, pred):
    """
    Produce CoNLL 2005 SRL evaluation results for a provided sentences given as lists of tags, with the first list being a
    list of targets. For example, [['-', 'love', '-'], ['(A0*)', '(V*)', '(A1*)']].
    :param gold: gold labels
    :param pred: predicted labels
    :return: SRL evaluation results
    """
    ntargets, ns, e = 0, 0, Evaluation()

    for sent_id, (gold_sent, pred_sent) in enumerate(zip(gold, pred)):
        gold_targets = gold_sent[0]
        pred_targets = pred_sent[0]
        if len(gold_targets) != len(pred_targets):
            raise RuntimeError('Sentence {}: gold and pred sentences do not align correctly!'.format(sent_id))
        sent = SrlSentence(sent_id)
        sent.gold = SrlSentence.load_props(sent_id, gold_sent)
        sent.pred = SrlSentence.load_props(sent_id, pred_sent)
        sent.words = len(gold_targets)

        for i in range(len(sent.gold)):
            gprop = sent.gold.get(i)
            pprop = sent.pred.get(i)
            if pprop and not gprop:
                print("Warning : sentence {} : verb {} at position {} : found predicted prop without its gold reference! "
                      "Skipping prop!".format(sent_id, pprop.verb, pprop.position))
            elif gprop:
                if not pprop:
                    print("Warning : sentence {} : verb {} at position {} : missing predicted prop! Counting all arguments as "
                          "missed!".format(sent_id, gprop.verb, gprop.position))
                elif gprop.verb != pprop.verb:
                    print("Warning : sentence {} : props do not match : expecting {} at position {}, found {} at position {}! "
                          "Counting all gold arguments as missed!".format(sent_id, gprop.verb, gprop.position, pprop.verb,
                                                                          pprop.position))
            ntargets += 1
            results = _evaluate_proposition(gprop, pprop, exclusions=['V'])
            e.ok += results.ok
            e.op += results.op
            e.ms += results.ms
            e.ptv += results.ptv

            for key, val in results.types.items():
                e.types[key][OKAY_KEY] += val[OKAY_KEY]
                e.types[key][EXCESS_KEY] += val[EXCESS_KEY]
                e.types[key][MISS_KEY] += val[MISS_KEY]
            for key, val in results.excluded.items():
                e.excluded[key][OKAY_KEY] += val[OKAY_KEY]
                e.excluded[key][EXCESS_KEY] += val[EXCESS_KEY]
                e.excluded[key][MISS_KEY] += val[MISS_KEY]

            e.update_confusion_matrix(gprop, pprop)
        ns += 1
    return SrlEvaluation(ns, ntargets, e)


def eval_from_files(gold_path, pred_path):
    """
    Produce CoNLL 2005 SRL evaluation results for given gold/predicted SRL tags, provided as paths.
    :param gold_path: path to gold file
    :param pred_path: path to predictions file
    :return: CoNLL 2005 SRL evaluation results
    """
    gold_props, pred_props = conll_iterator(gold_path), conll_iterator(pred_path)
    return evaluate(gold=gold_props, pred=pred_props)


def get_perl_output(gold_path, pred_path, latex=False, confusions=False):
    """
    Return a string equivalent to original CoNLL 2005 perl script output.
    :param gold_path: path to gold props
    :param pred_path: path to predicted props
    :param latex: if `True`, output in LaTeX format
    :param confusions: if `True`, output confusion matrix
    :return: evaluation output string
    """
    evaluation_results = eval_from_files(gold_path=gold_path, pred_path=pred_path)
    results = []
    if latex:
        results.append(evaluation_results.latex())
    else:
        results.append(str(evaluation_results))
    if confusions:
        results.append(evaluation_results.confusion_matrix())
    return '\n'.join(results)


def options(args=None):
    parser = argparse.ArgumentParser(description="Evaluation program for the CoNLL-2005 Shared Task")
    parser.add_argument('--gold', type=str, required=True, help='Path to file containing gold propositions.')
    parser.add_argument('--pred', type=str, required=True, help='Path to file containing predicted propositions.')
    parser.add_argument('--latex', dest='latex', action='store_true',
                        help='Produce a results table in LaTeX')
    parser.add_argument('-C', dest='confusions', action='store_true',
                        help='Produce a confusion matrix of gold vs. predicted arguments, wrt. their role')
    parser.set_defaults(confusions=False)
    parser.set_defaults(latex=False)
    return parser.parse_args(args)


def main():
    _opts = options()
    print(get_perl_output(_opts.gold, _opts.pred, _opts.latex, _opts.confusions))


if __name__ == '__main__':
    main()

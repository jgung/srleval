# CoNLL-2005 Semantic Role Labeling Evaluation in Python

The [official CoNLL-2005 Shared Task SRL evaluation script](http://www.lsi.upc.es/~srlconll/srl-eval.pl) is implemented in Perl.
It provides very useful functionality, such as producing confusion matrices and `LaTeX` tables.


This project provides a Python implementation of this script.
This may be useful for tracking validation performance during training in cloud environments that don't support Perl without extra configuration (such as [Google Cloud's ML Engine](https://cloud.google.com/ml-engine/)).

Note, the official evaluation script should be used for any published results.

## Usage
This script takes slightly different command line arguments than the original Perl script, favoring named instead of positional arguments for gold and predicted props.

```
usage: srleval [-h] --gold GOLD --pred PRED [--latex] [-C]

Evaluation program for the CoNLL-2005 Shared Task

optional arguments:
  -h, --help   show this help message and exit
  --gold GOLD  Path to file containing gold propositions.
  --pred PRED  Path to file containing predicted propositions.
  --latex      Produce a results table in LaTeX
  -C           Produce a confusion matrix of gold vs. predicted arguments,
               wrt. their role
```


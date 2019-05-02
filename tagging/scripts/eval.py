"""Evaulate a tagger.

Usage:
  eval.py -i <file> [-c]
  eval.py -h | --help

Options:
  -c            Show confusion matrix.
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import sys
from collections import defaultdict

from tagging.ancora import SimpleAncoraCorpusReader


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-3.0.1es/', files)
    sents = list(corpus.tagged_sents())

    # tag and evaluate
    # WORK HERE!!

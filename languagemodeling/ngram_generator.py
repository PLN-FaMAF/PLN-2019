from collections import defaultdict
import random


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self._n = model._n

        # compute the probabilities
        probs = defaultdict(dict)
        # WORK HERE!!

        self._probs = dict(probs)

        # sort in descending order for efficient sampling
        self._sorted_probs = sorted_probs = {}
        # WORK HERE!!

    def generate_sent(self):
        """Randomly generate a sentence."""
        # WORK HERE!!

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # WORK HERE!!

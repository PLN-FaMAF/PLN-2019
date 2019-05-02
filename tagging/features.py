from collections import namedtuple

from featureforge.feature import Feature


# sent -- the whole sentence.
# prev_tags -- a tuple with the n previous tags.
# i -- the position to be tagged.
History = namedtuple('History', 'sent prev_tags i')


def word_lower(h):
    """Feature: current lowercased word.

    h -- a history.
    """
    # WORK HERE!! USE STRING METHOD lower()


def prev_tags(h):
    """Feature: previous tags tuple.

    h -- a history.
    """
    # WORK HERE!!


def word_istitle(h):
    """Feature: is the current word titlecased?

    h -- a history.
    """
    # WORK HERE!! USE STRING METHOD istitle()


def word_isupper(h):
    """Feature: is the current word in uppercase?

    h -- a history.
    """
    # WORK HERE!! USE STRING METHOD isupper()


def word_isdigit(h):
    """Feature: is the current word all digits?

    h -- a history.
    """
    # WORK HERE!! USE STRING METHOD isdigit()


class NPrevTags(Feature):

    def __init__(self, n):
        """Feature: n previous tags tuple.

        n -- number of previous tags to consider.
        """
        self._n = n

    def _evaluate(self, h):
        """Feature: n previous tags tuple.

        h -- a history.
        """
        # WORK HERE!!


class PrevWord(Feature):

    def __init__(self, f):
        """Apply a feature to the previous word.

        f -- the feature.
        """
        self._f = f

    def _evaluate(self, h):
        """Apply a feature to the previous word.

        h -- a history.
        """
        i = h.i
        if i > 0:
            return str(self._f(History(h.sent, h.prev_tags, i - 1)))
        else:
            return 'BOS'  # beginning of sentence


class NextWord(Feature):

    def __init__(self, f):
        """Apply a feature to the next word.

        f -- the feature.
        """
        self._f = f

    def _evaluate(self, h):
        """Apply a feature to the next word.

        h -- a history.
        """
        sent, i = h.sent, h.i
        if i < len(sent) - 1:
            return str(self._f(History(sent, h.prev_tags, i + 1)))
        else:
            return 'EOS'  # end of sentence

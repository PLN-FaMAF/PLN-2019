"""Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt
from collections import defaultdict

from tagging.ancora import SimpleAncoraCorpusReader


class POSStats:
    """Several statistics for a POS tagged corpus.
    """

    def __init__(self, tagged_sents):
        """
        tagged_sents -- corpus (list/iterable/generator of tagged sentences)
        """
        # WORK HERE!!
        # COLLECT REQUIRED STATISTICS INTO DICTIONARIES.

    def sent_count(self):
        """Total number of sentences."""
        # WORK HERE!!

    def token_count(self):
        """Total number of tokens."""
        # WORK HERE!!

    def words(self):
        """Vocabulary (set of word types)."""
        # WORK HERE!!

    def word_count(self):
        """Vocabulary size."""
        # WORK HERE!!

    def word_freq(self, w):
        """Frequency of word w."""
        # WORK HERE!!

    def unambiguous_words(self):
        """List of words with only one observed POS tag."""
        # WORK HERE!!

    def ambiguous_words(self, n):
        """List of words with n different observed POS tags.

        n -- number of tags.
        """
        # WORK HERE!!

    def tags(self):
        """POS Tagset."""
        # WORK HERE!!

    def tag_count(self):
        """POS tagset size."""
        # WORK HERE!!

    def tag_freq(self, t):
        """Frequency of tag t."""
        # WORK HERE!!

    def tag_word_dict(self, t):
        """Dictionary of words and their counts for tag t."""
        return dict(self._tcount[t])


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = SimpleAncoraCorpusReader('ancora/ancora-3.0.1es/')
    sents = corpus.tagged_sents()

    # compute the statistics
    stats = POSStats(sents)

    print('Basic Statistics')
    print('================')
    print('sents: {}'.format(stats.sent_count()))
    token_count = stats.token_count()
    print('tokens: {}'.format(token_count))
    word_count = stats.word_count()
    print('words: {}'.format(word_count))
    print('tags: {}'.format(stats.tag_count()))
    print('')

    print('Most Frequent POS Tags')
    print('======================')
    tags = [(t, stats.tag_freq(t)) for t in stats.tags()]
    sorted_tags = sorted(tags, key=lambda t_f: -t_f[1])
    print('tag\tfreq\t%\ttop')
    for t, f in sorted_tags[:10]:
        words = stats.tag_word_dict(t).items()
        sorted_words = sorted(words, key=lambda w_f: -w_f[1])
        top = [w for w, _ in sorted_words[:5]]
        print('{0}\t{1}\t{2:2.2f}\t({3})'.format(t, f, f * 100 / token_count, ', '.join(top)))
    print('')

    print('Word Ambiguity Levels')
    print('=====================')
    print('n\twords\t%\ttop')
    for n in range(1, 10):
        words = list(stats.ambiguous_words(n))
        m = len(words)

        # most frequent words:
        sorted_words = sorted(words, key=lambda w: -stats.word_freq(w))
        top = sorted_words[:5]
        print('{0}\t{1}\t{2:2.2f}\t({3})'.format(n, m, m * 100 / word_count, ', '.join(top)))

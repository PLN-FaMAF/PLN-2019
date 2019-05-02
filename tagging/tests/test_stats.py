# https://docs.python.org/3/library/unittest.html
from unittest import TestCase

from tagging.scripts.stats import POSStats


class TestStats(TestCase):

    def setUp(self):
        self.tagged_sents = [
            list(zip('el gato come pescado .'.split(),
                 'D N V N P'.split())),
            list(zip('la gata come salmón .'.split(),
                 'D N V N P'.split())),
        ]

    def test_basic_stats(self):
        stats = POSStats(self.tagged_sents)

        self.assertEqual(stats.sent_count(), 2)
        self.assertEqual(stats.token_count(), 10)
        self.assertEqual(stats.word_count(), 8)
        self.assertEqual(stats.tag_count(), 4)

    def test_words(self):
        stats = POSStats(self.tagged_sents)

        words = {'el', 'gato', 'come', 'pescado', 'la', 'gata', 'salmón', '.'}

        self.assertEqual(stats.words(), words)

    def test_word_freq(self):
        stats = POSStats(self.tagged_sents)

        freqs = {
            'el': 1,
            'gato': 1,
            'come': 2,
            'pescado': 1,
            'la': 1,
            'gata': 1,
            'salmón': 1,
            '.': 2,
        }

        for tag, freq in freqs.items():
            self.assertEqual(stats.word_freq(tag), freq)

    def test_tag_freq(self):
        stats = POSStats(self.tagged_sents)

        freqs = {
            'D': 2,
            'N': 4,
            'V': 2,
            'P': 2,
        }

        for tag, freq in freqs.items():
            self.assertEqual(stats.tag_freq(tag), freq)

    def test_tag_word_dict(self):
        stats = POSStats(self.tagged_sents)

        word_dicts = {
            'D': {'el': 1, 'la': 1},
            'N': {'salmón': 1, 'gata': 1, 'gato': 1, 'pescado': 1},
            'V': {'come': 2},
            'P': {'.': 2},
        }

        for tag, word_dict in word_dicts.items():
            self.assertEqual(stats.tag_word_dict(tag), word_dict)

    def test_unambiguous_words(self):
        stats = POSStats(self.tagged_sents)

        self.assertEqual(
            set(stats.unambiguous_words()),
            {'salmón', 'come', 'pescado', 'el', 'la', '.', 'gato', 'gata'}
        )

    def test_ambiguous_words(self):
        stats = POSStats(self.tagged_sents)

        self.assertEqual(set(stats.ambiguous_words(2)), set())
        self.assertEqual(set(stats.ambiguous_words(3)), set())
        self.assertEqual(set(stats.ambiguous_words(4)), set())
        self.assertEqual(set(stats.ambiguous_words(5)), set())

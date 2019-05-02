# https://docs.python.org/3/library/unittest.html
from unittest import TestCase

from tagging.features import (History, word_lower, word_istitle, word_isupper,
                              word_isdigit, prev_tags, NPrevTags, PrevWord)


class TestHistory(TestCase):

    def test_eq(self):
        h1 = History('el gato come pescado .'.split(), ('<s>', '<s>'), 0)
        h2 = History('el gato come pescado .'.split(), ('<s>', '<s>'), 0)

        self.assertEqual(h1, h2)

        h3 = History('la gata come salmón .'.split(), ('<s>', '<s>'), 0)

        self.assertNotEqual(h1, h3)


class TestFeatures(TestCase):

    def setUp(self):
        self.tagged_sents = [
            list(zip('el gato come pescado .'.split(),
                 'D N V N P'.split())),
            list(zip('la gata come salmón .'.split(),
                 'D N V N P'.split())),
        ]

    def test_word_lower(self):
        sent0 = 'El gato come pescado .'.split()
        sent1 = 'La gata come salmón .'.split()
        feature_values = [
            (History(sent0, ('<s>', '<s>'), 0), 'el'),
            (History(sent0, ('<s>', 'D'), 1), 'gato'),
            (History(sent0, ('D', 'N'), 2), 'come'),
            (History(sent0, ('N', 'V'), 3), 'pescado'),
            (History(sent0, ('V', 'N'), 4), '.'),
            (History(sent1, ('<s>', '<s>'), 0), 'la'),
            (History(sent1, ('<s>', 'D'), 1), 'gata'),
            (History(sent1, ('D', 'N'), 2), 'come'),
            (History(sent1, ('N', 'V'), 3), 'salmón'),
            (History(sent1, ('V', 'N'), 4), '.'),
        ]
        for h, v in feature_values:
            self.assertEqual(word_lower(h), v)

    def test_prev_tags(self):
        sent0 = 'El gato come pescado .'.split()
        sent1 = 'La gata come salmón .'.split()
        feature_values = [
            (History(sent0, ('<s>', '<s>'), 0), ('<s>', '<s>')),
            (History(sent0, ('<s>', 'D'), 1), ('<s>', 'D')),
            (History(sent0, ('D', 'N'), 2), ('D', 'N')),
            (History(sent0, ('N', 'V'), 3), ('N', 'V')),
            (History(sent0, ('V', 'N'), 4), ('V', 'N')),
            (History(sent1, ('<s>', '<s>'), 0), ('<s>', '<s>')),
            (History(sent1, ('<s>', 'D'), 1), ('<s>', 'D')),
            (History(sent1, ('D', 'N'), 2), ('D', 'N')),
            (History(sent1, ('N', 'V'), 3), ('N', 'V')),
            (History(sent1, ('V', 'N'), 4), ('V', 'N')),
        ]
        for h, v in feature_values:
            self.assertEqual(prev_tags(h), v)

    def test_one_prev_tags(self):
        n_prev_tags = NPrevTags(1)

        sent0 = 'El gato come pescado .'.split()
        sent1 = 'La gata come salmón .'.split()
        feature_values = [
            (History(sent0, ('<s>', '<s>'), 0), ('<s>',)),
            (History(sent0, ('<s>', 'D'), 1), ('D',)),
            (History(sent0, ('D', 'N'), 2), ('N',)),
            (History(sent0, ('N', 'V'), 3), ('V',)),
            (History(sent0, ('V', 'N'), 4), ('N',)),
            (History(sent1, ('<s>', '<s>'), 0), ('<s>',)),
            (History(sent1, ('<s>', 'D'), 1), ('D',)),
            (History(sent1, ('D', 'N'), 2), ('N',)),
            (History(sent1, ('N', 'V'), 3), ('V',)),
            (History(sent1, ('V', 'N'), 4), ('N',)),
        ]
        for h, v in feature_values:
            self.assertEqual(n_prev_tags(h), v)

    def test_two_prev_tags(self):
        n_prev_tags = NPrevTags(2)

        sent0 = 'El gato come pescado .'.split()
        sent1 = 'La gata come salmón .'.split()
        feature_values = [
            (History(sent0, ('<s>', '<s>'), 0), ('<s>', '<s>')),
            (History(sent0, ('<s>', 'D'), 1), ('<s>', 'D')),
            (History(sent0, ('D', 'N'), 2), ('D', 'N')),
            (History(sent0, ('N', 'V'), 3), ('N', 'V')),
            (History(sent0, ('V', 'N'), 4), ('V', 'N')),
            (History(sent1, ('<s>', '<s>'), 0), ('<s>', '<s>')),
            (History(sent1, ('<s>', 'D'), 1), ('<s>', 'D')),
            (History(sent1, ('D', 'N'), 2), ('D', 'N')),
            (History(sent1, ('N', 'V'), 3), ('N', 'V')),
            (History(sent1, ('V', 'N'), 4), ('V', 'N')),
        ]
        for h, v in feature_values:
            self.assertEqual(n_prev_tags(h), v)

    def test_word_istitle(self):
        sent0 = 'EL gato come pescado .'.split()
        sent1 = 'La gata come SALMÓN .'.split()
        feature_values = [
            (History(sent0, ('<s>', '<s>'), 0), False),
            (History(sent0, ('<s>', 'D'), 1), False),
            (History(sent0, ('D', 'N'), 2), False),
            (History(sent0, ('N', 'V'), 3), False),
            (History(sent0, ('V', 'N'), 4), False),
            (History(sent1, ('<s>', '<s>'), 0), True),
            (History(sent1, ('<s>', 'D'), 1), False),
            (History(sent1, ('D', 'N'), 2), False),
            (History(sent1, ('N', 'V'), 3), False),
            (History(sent1, ('V', 'N'), 4), False),
        ]
        for h, v in feature_values:
            self.assertEqual(word_istitle(h), v)

    def test_word_isupper(self):
        sent0 = 'EL gato come pescado .'.split()
        sent1 = 'La gata come SALMÓN .'.split()
        feature_values = [
            (History(sent0, ('<s>', '<s>'), 0), True),
            (History(sent0, ('<s>', 'D'), 1), False),
            (History(sent0, ('D', 'N'), 2), False),
            (History(sent0, ('N', 'V'), 3), False),
            (History(sent0, ('V', 'N'), 4), False),
            (History(sent1, ('<s>', '<s>'), 0), False),
            (History(sent1, ('<s>', 'D'), 1), False),
            (History(sent1, ('D', 'N'), 2), False),
            (History(sent1, ('N', 'V'), 3), True),
            (History(sent1, ('V', 'N'), 4), False),
        ]
        for h, v in feature_values:
            self.assertEqual(word_isupper(h), v, h)

    def test_word_isdigit(self):
        sent0 = 'El gato come 3 pescados .'.split()
        sent1 = 'Las 10 gatas c0m3n salmón .'.split()
        feature_values = [
            (History(sent0, ('<s>', '<s>'), 0), False),
            (History(sent0, ('<s>', 'D'), 1), False),
            (History(sent0, ('D', 'N'), 2), False),
            (History(sent0, ('N', 'V'), 3), True),
            (History(sent0, ('V', 'C'), 4), False),
            (History(sent0, ('C', 'N'), 5), False),
            (History(sent1, ('<s>', '<s>'), 0), False),
            (History(sent1, ('<s>', 'D'), 1), True),
            (History(sent1, ('D', 'C'), 2), False),
            (History(sent1, ('C', 'N'), 3), False),
            (History(sent1, ('N', 'V'), 4), False),
            (History(sent1, ('V', 'N'), 5), False),
        ]
        for h, v in feature_values:
            self.assertEqual(word_isdigit(h), v, h)

    def test_prev_word_lower(self):
        prev_word_lower = PrevWord(word_lower)

        sent0 = 'El gato come pescado .'.split()
        sent1 = 'La gata come salmón .'.split()
        feature_values = [
            (History(sent0, ('<s>', '<s>'), 0), 'BOS'),  # beginning of sentence
            (History(sent0, ('<s>', 'D'), 1), 'el'),
            (History(sent0, ('D', 'N'), 2), 'gato'),
            (History(sent0, ('N', 'V'), 3), 'come'),
            (History(sent0, ('V', 'N'), 4), 'pescado'),
            (History(sent1, ('<s>', '<s>'), 0), 'BOS'),  # beginning of sentence
            (History(sent1, ('<s>', 'D'), 1), 'la'),
            (History(sent1, ('D', 'N'), 2), 'gata'),
            (History(sent1, ('N', 'V'), 3), 'come'),
            (History(sent1, ('V', 'N'), 4), 'salmón'),
        ]
        for h, v in feature_values:
            self.assertEqual(prev_word_lower(h), v)

    def test_prev_word_istitle(self):
        prev_word_istitle = PrevWord(word_istitle)

        sent0 = 'EL gato come pescado .'.split()
        sent1 = 'La gata come SALMÓN .'.split()
        feature_values = [
            (History(sent0, ('<s>', '<s>'), 0), 'BOS'),
            (History(sent0, ('<s>', 'D'), 1), 'False'),
            (History(sent0, ('D', 'N'), 2), 'False'),
            (History(sent0, ('N', 'V'), 3), 'False'),
            (History(sent0, ('V', 'N'), 4), 'False'),
            (History(sent1, ('<s>', '<s>'), 0), 'BOS'),
            (History(sent1, ('<s>', 'D'), 1), 'True'),
            (History(sent1, ('D', 'N'), 2), 'False'),
            (History(sent1, ('N', 'V'), 3), 'False'),
            (History(sent1, ('V', 'N'), 4), 'False'),
        ]
        for h, v in feature_values:
            self.assertEqual(prev_word_istitle(h), v)

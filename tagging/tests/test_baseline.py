# https://docs.python.org/3/library/unittest.html
from unittest import TestCase

from tagging.baseline import BaselineTagger


class TestBaselineTagger(TestCase):

    def setUp(self):
        self.tagged_sents = [
            list(zip('el gato come pescado .'.split(),
                 'D N V N P'.split())),
            list(zip('la gata come salmón .'.split(),
                 'D N V N P'.split())),
        ]

    def test_tag_word(self):
        baseline = BaselineTagger(self.tagged_sents, default_tag='N')

        for w, t in zip('el gato come pescado .'.split(), 'D N V N P'.split()):
            self.assertEqual(t, baseline.tag_word(w))

        for w, t in zip('el perro come salame .'.split(), 'D N V N P'.split()):
            self.assertEqual(t, baseline.tag_word(w))

    def test_tag(self):
        baseline = BaselineTagger(self.tagged_sents, default_tag='N')

        y = baseline.tag('el gato come pescado .'.split())
        self.assertEqual(y, 'D N V N P'.split())

        y = baseline.tag('el perro come salame .'.split())
        self.assertEqual(y, 'D N V N P'.split())

    def test_unknown(self):
        baseline = BaselineTagger(self.tagged_sents, default_tag='N')

        known = {'el', 'gato', 'come', 'pescado', '.', 'la', 'gata', 'salmón'}
        for w in known:
            self.assertFalse(baseline.unknown(w))

        unknown = {'perro', 'salame'}
        for w in unknown:
            self.assertTrue(baseline.unknown(w))

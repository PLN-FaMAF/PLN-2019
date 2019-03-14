# https://docs.python.org/3/library/unittest.html
from unittest import TestCase

from languagemodeling.ngram import AddOneNGram


class TestAddOneNGram(TestCase):

    def setUp(self):
        self.sents = [
            'el gato come pescado .'.split(),
            'la gata come salmón .'.split(),
        ]

    def test_count_1gram(self):
        model = AddOneNGram(1, self.sents)

        counts = {
            (): 12,
            ('el',): 1,
            ('gato',): 1,
            ('come',): 2,
            ('pescado',): 1,
            ('.',): 2,
            ('</s>',): 2,
            ('la',): 1,
            ('gata',): 1,
            ('salmón',): 1,
        }
        for gram, c in counts.items():
            self.assertEqual(model.count(gram), c, gram)

        # size of the vocabulary
        self.assertEqual(model.V(), 9)

    def test_count_2gram(self):
        model = AddOneNGram(2, self.sents)

        counts = {
            ('<s>',): 2,
            ('el',): 1,
            ('gato',): 1,
            ('come',): 2,
            ('pescado',): 1,
            ('.',): 2,
            ('la',): 1,
            ('gata',): 1,
            ('salmón',): 1,
            ('<s>', 'el'): 1,
            ('el', 'gato'): 1,
            ('gato', 'come'): 1,
            ('come', 'pescado'): 1,
            ('pescado', '.'): 1,
            ('.', '</s>'): 2,
            ('<s>', 'la'): 1,
            ('la', 'gata'): 1,
            ('gata', 'come'): 1,
            ('come', 'salmón'): 1,
            ('salmón', '.'): 1,
        }
        for gram, c in counts.items():
            self.assertEqual(model.count(gram), c, gram)

        # size of the vocabulary
        self.assertEqual(model.V(), 9)

    def test_cond_prob_1gram(self):
        model = AddOneNGram(1, self.sents)

        probs = {
            'pescado': (1.0 + 1.0) / (12.0 + 9.0),
            'come': (2.0 + 1.0) / (12.0 + 9.0),
            'salame': 1.0 / (12.0 + 9.0),
        }
        for token, p in probs.items():
            self.assertAlmostEqual(model.cond_prob(token), p)

    def test_cond_prob_2gram(self):
        model = AddOneNGram(2, self.sents)

        probs = {
            ('pescado', 'come'): (1.0 + 1.0) / (2.0 + 9.0),
            ('salmón', 'come'): (1.0 + 1.0) / (2.0 + 9.0),
            ('salame', 'come'): 1.0 / (2.0 + 9.0),
        }
        for (token, prev), p in probs.items():
            self.assertAlmostEqual(model.cond_prob(token, (prev,)), p)

    def test_norm_1gram(self):
        model = AddOneNGram(1, self.sents)

        tokens = {'el', 'gato', 'come', 'pescado', '.', 'la', 'gata', 'salmón', '</s>'}

        prob_sum = sum(model.cond_prob(token) for token in tokens)
        # prob_sum < 1.0 or almost equal to 1.0:
        self.assertTrue(prob_sum < 1.0 or abs(prob_sum - 1.0) < 1e-10)

    def test_norm_2gram(self):
        model = AddOneNGram(2, self.sents)

        tokens = {'el', 'gato', 'come', 'pescado', '.', 'la', 'gata', 'salmón', '</s>'}

        for prev in list(tokens) + ['<s>']:
            prob_sum = sum(model.cond_prob(token, (prev,)) for token in tokens)
            # prob_sum < 1.0 or almost equal to 1.0:
            self.assertTrue(prob_sum < 1.0 or abs(prob_sum - 1.0) < 1e-10)

    def test_norm_3gram(self):
        model = AddOneNGram(3, self.sents)

        tokens = {'el', 'gato', 'come', 'pescado', '.', 'la', 'gata', 'salmón', '</s>'}
        prevs = [('<s>', '<s>')] + \
            [('<s>', t) for t in tokens] + \
            [(t1, t2) for t1 in tokens for t2 in tokens]

        for prev in prevs:
            prob_sum = sum(model.cond_prob(token, prev) for token in tokens)
            # prob_sum < 1.0 or almost equal to 1.0:
            self.assertTrue(prob_sum < 1.0 or abs(prob_sum - 1.0) < 1e-10)

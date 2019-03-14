# https://docs.python.org/3/library/unittest.html
from unittest import TestCase

from languagemodeling.ngram import InterpolatedNGram


class TestInterpolatedNGram(TestCase):

    def setUp(self):
        self.sents = [
            'el gato come pescado .'.split(),
            'la gata come salmón .'.split(),
        ]

    def test_count_1gram(self):
        model = InterpolatedNGram(1, self.sents, gamma=1.0)

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

    def test_count_2gram(self):
        ngram = InterpolatedNGram(2, self.sents, gamma=1.0)

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
            self.assertEqual(ngram.count(gram), c, gram)

    def test_cond_prob_1gram_no_addone(self):
        model = InterpolatedNGram(1, self.sents, gamma=1.0, addone=False)

        # behaves just like unsmoothed n-gram
        probs = {
            'pescado': 1 / 12.0,
            'come': 2 / 12.0,
            'salame': 0.0,
        }
        for token, p in probs.items():
            self.assertAlmostEqual(model.cond_prob(token), p, msg=token)

    def test_cond_prob_2gram_no_addone(self):
        gamma = 1.0
        model = InterpolatedNGram(2, self.sents, gamma, addone=False)

        c1 = 2.0  # count for 'come' (and '.')
        l1 = c1 / (c1 + gamma)

        probs = {
            ('pescado', 'come'): l1 * 0.5 + (1.0 - l1) * 1 / 12.0,
            ('salmón', 'come'): l1 * 0.5 + (1.0 - l1) * 1 / 12.0,
            ('salame', 'come'): 0.0,
            ('</s>', '.'): l1 * 1.0 + (1.0 - l1) * 2 / 12.0,
        }
        for (token, prev), p in probs.items():
            self.assertAlmostEqual(model.cond_prob(token, (prev,)), p, msg=token)

    def test_norm_1gram(self):
        models = [
            InterpolatedNGram(1, self.sents, gamma=1.0, addone=False),
            InterpolatedNGram(1, self.sents, gamma=5.0, addone=False),
            InterpolatedNGram(1, self.sents, gamma=10.0, addone=False),
            InterpolatedNGram(1, self.sents, gamma=50.0, addone=False),
            InterpolatedNGram(1, self.sents, gamma=100.0, addone=False),
            InterpolatedNGram(1, self.sents, gamma=1.0, addone=True),
            InterpolatedNGram(1, self.sents, gamma=5.0, addone=True),
            InterpolatedNGram(1, self.sents, gamma=10.0, addone=True),
            InterpolatedNGram(1, self.sents, gamma=50.0, addone=True),
            InterpolatedNGram(1, self.sents, gamma=100.0, addone=True),
        ]

        tokens = {'el', 'gato', 'come', 'pescado', '.', 'la', 'gata', 'salmón', '</s>'}

        for model in models:
            prob_sum = sum(model.cond_prob(token) for token in tokens)
            # prob_sum < 1.0 or almost equal to 1.0:
            self.assertAlmostLessEqual(prob_sum, 1.0)

    def test_norm_2gram(self):
        models = [
            InterpolatedNGram(2, self.sents, gamma=1.0, addone=False),
            InterpolatedNGram(2, self.sents, gamma=5.0, addone=False),
            InterpolatedNGram(2, self.sents, gamma=10.0, addone=False),
            InterpolatedNGram(2, self.sents, gamma=50.0, addone=False),
            InterpolatedNGram(2, self.sents, gamma=100.0, addone=False),
            InterpolatedNGram(2, self.sents, gamma=1.0, addone=True),
            InterpolatedNGram(2, self.sents, gamma=5.0, addone=True),
            InterpolatedNGram(2, self.sents, gamma=10.0, addone=True),
            InterpolatedNGram(2, self.sents, gamma=50.0, addone=True),
            InterpolatedNGram(2, self.sents, gamma=100.0, addone=True),
        ]

        tokens = {'el', 'gato', 'come', 'pescado', '.', 'la', 'gata', 'salmón', '</s>'}
        prevs = {'el', 'gato', 'come', 'pescado', '.', 'la', 'gata', 'salmón', '<s>'}

        for model in models:
            for prev in prevs:
                prob_sum = sum(model.cond_prob(token, (prev,)) for token in tokens)
                # prob_sum < 1.0 or almost equal to 1.0:
                self.assertAlmostLessEqual(prob_sum, 1.0, msg=prev)

    def test_norm_3gram(self):
        models = [
            InterpolatedNGram(3, self.sents, gamma=1.0, addone=False),
            InterpolatedNGram(3, self.sents, gamma=5.0, addone=False),
            InterpolatedNGram(3, self.sents, gamma=10.0, addone=False),
            InterpolatedNGram(3, self.sents, gamma=50.0, addone=False),
            InterpolatedNGram(3, self.sents, gamma=100.0, addone=False),
            InterpolatedNGram(3, self.sents, gamma=1.0, addone=True),
            InterpolatedNGram(3, self.sents, gamma=5.0, addone=True),
            InterpolatedNGram(3, self.sents, gamma=10.0, addone=True),
            InterpolatedNGram(3, self.sents, gamma=50.0, addone=True),
            InterpolatedNGram(3, self.sents, gamma=100.0, addone=True),
        ]

        tokens = {'el', 'gato', 'come', 'pescado', '.', 'la', 'gata', 'salmón', '</s>'}
        prev_tokens = {'el', 'gato', 'come', 'pescado', '.', 'la', 'gata', 'salmón', '<s>'}
        prevs = [('<s>', '<s>')] + \
            [('<s>', t) for t in prev_tokens] + \
            [(t1, t2) for t1 in prev_tokens for t2 in prev_tokens]

        for model in models:
            for prev in prevs:
                prob_sum = sum(model.cond_prob(token, prev) for token in tokens)
                # prob_sum < 1.0 or almost equal to 1.0:
                self.assertAlmostLessEqual(prob_sum, 1.0, msg=prev)

    def test_held_out(self):
        model = InterpolatedNGram(1, self.sents)

        # only first sentence (second sentence is held-out data)
        counts = {
            (): 6,
            ('el',): 1,
            ('gato',): 1,
            ('come',): 1,
            ('pescado',): 1,
            ('.',): 1,
            ('</s>',): 1,
        }
        for gram, c in counts.items():
            self.assertEqual(model.count(gram), c, gram)

    def assertAlmostLessEqual(self, a, b, places=7, msg=None):
        self.assertTrue(a < b or round(abs(a - b), places) == 0, msg=msg)

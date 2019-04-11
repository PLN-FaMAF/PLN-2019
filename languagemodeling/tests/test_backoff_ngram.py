# https://docs.python.org/3/library/unittest.html
from unittest import TestCase

from languagemodeling.ngram import BackOffNGram


class TestBackoffNGram(TestCase):

    def setUp(self):
        self.sents = [
            'el gato come pescado .'.split(),
            'la gata come salmón .'.split(),
        ]

    def test_init_2gram(self):
        model = BackOffNGram(2, self.sents, beta=0.5)

        A = {
            ('<s>',): {'el', 'la'},
            ('el',): {'gato'},
            ('gato',): {'come'},
            ('come',): {'pescado', 'salmón'},
            ('pescado',): {'.'},
            ('.',): {'</s>'},
            ('la',): {'gata'},
            ('gata',): {'come'},
            ('salmón',): {'.'},
        }
        for tokens, Aset in A.items():
            self.assertEqual(model.A(tokens), Aset, tokens)

        # missing probability mass
        alpha = {
            ('<s>',): 2 * 0.5 / 2,
            ('el',): 1 * 0.5 / 1,
            ('gato',): 1 * 0.5 / 1,
            ('come',): 2 * 0.5 / 2,
            ('pescado',): 1 * 0.5 / 1,
            ('.',): 1 * 0.5 / 2,
            ('la',): 1 * 0.5 / 1,
            ('gata',): 1 * 0.5 / 1,
            ('salmón',): 1 * 0.5 / 1,
        }
        for tokens, a in alpha.items():
            self.assertAlmostEqual(model.alpha(tokens), a, msg=tokens)

        # normalization factor
        denom = {
            ('<s>',): 1.0 - model.cond_prob('el') - model.cond_prob('la'),
            ('el',): 1.0 - model.cond_prob('gato'),
            ('gato',): 1.0 - model.cond_prob('come'),
            ('come',): 1.0 - model.cond_prob('pescado') - model.cond_prob('salmón'),
            ('pescado',): 1.0 - model.cond_prob('.'),
            ('.',): 1.0 - model.cond_prob('</s>'),
            ('la',): 1.0 - model.cond_prob('gata'),
            ('gata',): 1.0 - model.cond_prob('come'),
            ('salmón',): 1.0 - model.cond_prob('.'),
        }
        for tokens, d in denom.items():
            self.assertAlmostEqual(model.denom(tokens), d, msg=tokens)

    def test_count_1gram(self):
        models = [
            # same test for different values of beta and addone:
            BackOffNGram(1, self.sents, beta=0.5),
            BackOffNGram(1, self.sents, beta=0.5, addone=False),
            BackOffNGram(1, self.sents, beta=0.0),
            BackOffNGram(1, self.sents, beta=0.0, addone=False),
        ]

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
        for model in models:
            for gram, c in counts.items():
                self.assertEqual(model.count(gram), c)

    def test_count_2gram(self):
        models = [
            # same test for different values of beta and addone:
            BackOffNGram(2, self.sents, beta=0.5),
            BackOffNGram(2, self.sents, beta=0.5, addone=False),
            BackOffNGram(2, self.sents, beta=0.0),
            BackOffNGram(2, self.sents, beta=0.0, addone=False),
        ]

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
        for model in models:
            for gram, c in counts.items():
                self.assertEqual(model.count(gram), c)

    def test_cond_prob_1gram_no_addone(self):
        model = BackOffNGram(1, self.sents, beta=0.5, addone=False)

        # behaves just like unsmoothed n-gram
        probs = {
            'pescado': 1 / 12.0,
            'come': 2 / 12.0,
            'salame': 0.0,
        }
        for token, p in probs.items():
            self.assertAlmostEqual(model.cond_prob(token), p, msg=token)

    def test_cond_prob_2gram_no_addone(self):
        model = BackOffNGram(2, self.sents, beta=0.5, addone=False)

        probs = {
            ('pescado', 'come'): (1 - 0.5) / 2.0,
            ('salmón', 'come'): (1 - 0.5) / 2.0,
            ('salame', 'come'): 0.0,  # back-off to the unigram that is 0.0
        }
        for (token, prev), p in probs.items():
            self.assertAlmostEqual(model.cond_prob(token, (prev,)), p, msg=token)

    def test_cond_prob_normalization_2gram_no_addone(self):
        model = BackOffNGram(2, self.sents, beta=0.5, addone=False)

        alpha = 1.0 - (1.0 - 0.5) / 1.0
        denom = model.denom(('el',))
        self.assertAlmostEqual(model.alpha(('el',)), alpha)

        probs = {
            ('gato', 'el'): (1.0 - 0.5) / 1.0,
            # back-off to the unigrams:
            ('el', 'el'): alpha * 1.0 / (12.0 * denom),
            ('come', 'el'): alpha * 2.0 / (12.0 * denom),
            ('pescado', 'el'): alpha * 1.0 / (12.0 * denom),
            ('.', 'el'): alpha * 2.0 / (12.0 * denom),
            ('</s>', 'el'): alpha * 2.0 / (12.0 * denom),
            ('la', 'el'): alpha * 1.0 / (12.0 * denom),
            ('gata', 'el'): alpha * 1.0 / (12.0 * denom),
            ('salmón', 'el'): alpha * 1.0 / (12.0 * denom),
        }
        for (token, prev), p in probs.items():
            self.assertAlmostEqual(model.cond_prob(token, (prev,)), p, msg=(token, prev))

        # the sum is one:
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0)

    def test_cond_prob_2gram_no_discount_no_addone(self):
        model = BackOffNGram(2, self.sents, beta=0.0, addone=False)

        probs = {
            ('pescado', 'come'): 1.0 / 2.0,
            ('salmón', 'come'): 1.0 / 2.0,
            ('salame', 'come'): 0.0,  # back-off to the unigram that is 0.0
        }
        for (token, prev), p in probs.items():
            self.assertAlmostEqual(model.cond_prob(token, (prev,)), p, msg=(token))

    def test_norm_1gram(self):
        models = [
            BackOffNGram(1, self.sents, beta=0.0, addone=False),
            BackOffNGram(1, self.sents, beta=0.5, addone=False),
            BackOffNGram(1, self.sents, beta=0.0, addone=True),
            BackOffNGram(1, self.sents, beta=0.5, addone=True),
        ]

        tokens = ['el', 'gato', 'come', 'pescado', '.', 'la', 'gata', 'salmón', '</s>']

        for model in models:
            prob_sum = sum(model.cond_prob(token) for token in tokens)
            # prob_sum < 1.0 or almost equal to 1.0:
            self.assertAlmostLessEqual(prob_sum, 1.0)

    def test_norm_2gram(self):
        models = [
            BackOffNGram(2, self.sents, beta=0.0, addone=False),
            BackOffNGram(2, self.sents, beta=0.5, addone=False),
            BackOffNGram(2, self.sents, beta=0.0, addone=True),
            BackOffNGram(2, self.sents, beta=0.5, addone=True),
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
            BackOffNGram(3, self.sents, beta=0.0, addone=False),
            BackOffNGram(3, self.sents, beta=0.5, addone=False),
            BackOffNGram(3, self.sents, beta=0.0, addone=True),
            BackOffNGram(3, self.sents, beta=0.5, addone=True),
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
        model = BackOffNGram(1, self.sents)

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

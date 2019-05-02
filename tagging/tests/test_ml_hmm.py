# https://docs.python.org/3/library/unittest.html
from unittest import TestCase
from math import log2

from tagging.hmm import MLHMM
from tagging.hmm import ViterbiTagger


class TestMLHMM(TestCase):

    def setUp(self):
        self.tagged_sents = [
            list(zip('el gato come pescado .'.split(),
                 'D N V N P'.split())),
            list(zip('la gata come salmón .'.split(),
                 'D N V N P'.split())),
        ]

    def test_tcount_1gram(self):
        hmm = MLHMM(1, self.tagged_sents)

        tcount = {
            (): 12,
            ('D',): 2,
            ('N',): 4,
            ('V',): 2,
            ('P',): 2,
            ('</s>',): 2,
        }
        for gram, c in tcount.items():
            self.assertEqual(hmm.tcount(gram), c, gram)

    def test_tag_prob_1gram(self):
        hmm = MLHMM(1, self.tagged_sents, addone=False)

        y = 'D N V N P'.split()
        p = hmm.tag_prob(y)
        # D V P and </s> have prob 2.0 / 12.0, N has prob 4.0 / 12.0.
        tag_prob = (2.0 / 12.0)**4 *  \
                   (4.0 / 12.0)**2
        self.assertAlmostEqual(p, tag_prob)

        lp = hmm.tag_log_prob(y)
        self.assertAlmostEqual(lp, log2(tag_prob))

    def test_prob_1gram(self):
        hmm = MLHMM(1, self.tagged_sents, addone=False)

        x = 'el gato come pescado .'.split()
        y = 'D N V N P'.split()
        p = hmm.prob(x, y)
        # D V P and </s> have prob 2.0 / 12.0, N has prob 4.0 / 12.0.
        tag_prob = (2.0 / 12.0)**4 *  \
                   (4.0 / 12.0)**2
        # probs for el/D gato/N come/V pescado/N ./P
        out_prob = 0.5 * 0.25 * 1.0 * 0.25 * 1.0
        self.assertAlmostEqual(p, tag_prob * out_prob)

        lp = hmm.log_prob(x, y)
        self.assertAlmostEqual(lp, log2(tag_prob) + log2(out_prob))

    def test_tcount_2gram(self):
        hmm = MLHMM(2, self.tagged_sents)

        tcount = {
            ('D',): 2,
            ('N',): 4,
            ('V',): 2,
            ('P',): 2,
            ('D', 'N'): 2,
            ('N', 'V'): 2,
            ('V', 'N'): 2,
            ('N', 'P'): 2,
            ('P', '</s>'): 2,
        }
        for gram, c in tcount.items():
            self.assertEqual(hmm.tcount(gram), c, gram)

    def test_trans_prob_2gram(self):
        hmm = MLHMM(2, self.tagged_sents, addone=False)

        probs = {
            ('D', ('<s>',)): 1.0,
            ('N', ('D',)): 1.0,
            ('V', ('N',)): 0.5,
            ('N', ('V',)): 1.0,
            ('P', ('N',)): 0.5,
            ('</s>', ('P',)): 1.0,
        }
        for params, p in probs.items():
            self.assertAlmostEqual(hmm.trans_prob(*params), p, msg=params)

    def test_tag_prob_2gram(self):
        hmm = MLHMM(2, self.tagged_sents, addone=False)

        y = 'D N V N P'.split()
        p = hmm.tag_prob(y)
        tag_prob = 0.5 * 0.5
        self.assertAlmostEqual(p, tag_prob)

        lp = hmm.tag_log_prob(y)
        self.assertAlmostEqual(lp, log2(tag_prob))

    def test_prob_2gram(self):
        hmm = MLHMM(2, self.tagged_sents, addone=False)

        x = 'el gato come pescado .'.split()
        y = 'D N V N P'.split()
        p = hmm.prob(x, y)
        # V after N and P after N have prob 0.5. the rest is 1.0.
        tag_prob = 0.5 * 0.5
        # probs for el/D gato/N come/V pescado/N ./P
        out_prob = 0.5 * 0.25 * 1.0 * 0.25 * 1.0
        self.assertAlmostEqual(p, tag_prob * out_prob)

        lp = hmm.log_prob(x, y)
        self.assertAlmostEqual(lp, log2(tag_prob) + log2(out_prob))

    def test_unknown(self):
        hmm = MLHMM(2, self.tagged_sents)

        known = {'el', 'gato', 'come', 'pescado', '.', 'la', 'gata', 'salmón'}
        for w in known:
            self.assertFalse(hmm.unknown(w))

        unknown = {'perro', 'salame'}
        for w in unknown:
            self.assertTrue(hmm.unknown(w))

    def test_viterbi_tagger(self):
        hmm = MLHMM(2, self.tagged_sents, addone=False)
        # XXX: or directly test hmm.tag?
        tagger = ViterbiTagger(hmm)

        y = tagger.tag('el gato come pescado .'.split())

        pi = {
            0: {
                ('<s>',): (0.0, []),
            },
            1: {
                # 0.5 for el/D
                ('D',): (log2(0.5), ['D']),
            },
            2: {
                # 0.25 for gato/N
                ('N',): (log2(0.5 * 0.25), ['D', 'N']),
            },
            3: {
                # 0.5 for V after N
                ('V',): (log2(0.5 * 0.25 * 0.5), ['D', 'N', 'V']),
            },
            4: {
                # 0.25 for pescado/N
                ('N',): (log2(0.5 * 0.25 * 0.5 * 0.25), ['D', 'N', 'V', 'N']),
            },
            5: {
                # 0.5 for P after N
                ('P',): (log2(0.5 * 0.25 * 0.5 * 0.25 * 0.5), ['D', 'N', 'V', 'N', 'P']),
            }

        }
        self.assertEqualPi(tagger._pi, pi)

        self.assertEqual(y, 'D N V N P'.split())

    def assertEqualPi(self, pi1, pi2):
        self.assertEqual(set(pi1.keys()), set(pi2.keys()))

        for k in pi1.keys():
            self.assertEqual(pi1[k], pi2[k], k)

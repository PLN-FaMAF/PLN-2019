# https://docs.python.org/3/library/unittest.html
from unittest import TestCase
from math import log2

from tagging.hmm import HMM, ViterbiTagger


class TestViterbiTagger(TestCase):

    def test_tag(self):
        tagset = {'D', 'N', 'V'}
        trans = {
            ('<s>', '<s>'): {'D': 1.0},
            ('<s>', 'D'): {'N': 1.0},
            ('D', 'N'): {'V': 1.0},
            ('N', 'V'): {'</s>': 1.0},
        }
        out = {
            'D': {'the': 1.0},
            'N': {'dog': 0.4, 'barks': 0.6},
            'V': {'dog': 0.1, 'barks': 0.9},
        }
        hmm = HMM(3, tagset, trans, out)
        tagger = ViterbiTagger(hmm)

        x = 'the dog barks'.split()
        y = tagger.tag(x)

        pi = {
            0: {
                ('<s>', '<s>'): (log2(1.0), []),
            },
            1: {
                ('<s>', 'D'): (log2(1.0), ['D']),
            },
            2: {
                ('D', 'N'): (log2(0.4), ['D', 'N']),
            },
            3: {
                ('N', 'V'): (log2(0.4 * 0.9), ['D', 'N', 'V']),
            }
        }
        self.assertEqualPi(tagger._pi, pi)

        self.assertEqual(y, 'D N V'.split())

    def test_tag2(self):
        tagset = {'D', 'N', 'V'}
        trans = {
            ('<s>', '<s>'): {'D': 1.0},
            ('<s>', 'D'): {'N': 1.0},
            ('D', 'N'): {'V': 0.8, 'N': 0.2},
            ('N', 'N'): {'V': 1.0},
            ('N', 'V'): {'</s>': 1.0},
        }
        out = {
            'D': {'the': 1.0},
            'N': {'dog': 0.4, 'barks': 0.6},
            'V': {'dog': 0.1, 'barks': 0.9},
        }
        hmm = HMM(3, tagset, trans, out)
        tagger = ViterbiTagger(hmm)

        x = 'the dog barks'.split()
        y = tagger.tag(x)

        pi = {
            0: {
                ('<s>', '<s>'): (log2(1.0), []),
            },
            1: {
                ('<s>', 'D'): (log2(1.0), ['D']),
            },
            2: {
                ('D', 'N'): (log2(0.4), ['D', 'N']),
            },
            3: {
                ('N', 'V'): (log2(0.8 * 0.4 * 0.9), ['D', 'N', 'V']),
                ('N', 'N'): (log2(0.2 * 0.4 * 0.6), ['D', 'N', 'N']),
            }
        }
        self.assertEqualPi(tagger._pi, pi)

        self.assertEqual(y, 'D N V'.split())

    def test_tag3(self):
        tagset = {'D', 'N', 'V'}
        trans = {
            ('<s>', '<s>'): {'D': 1.0},
            ('<s>', 'D'): {'N': 0.8, 'V': 0.2},
            ('D', 'N'): {'V': 0.8, 'N': 0.2},
            ('D', 'V'): {'V': 0.8, 'N': 0.2},
            ('N', 'N'): {'V': 1.0},
            ('N', 'V'): {'</s>': 1.0},
            # XXX: not sure if needed
            ('V', 'N'): {'</s>': 1.0},
            ('V', 'V'): {'</s>': 1.0},
        }
        out = {
            'D': {'the': 1.0},
            'N': {'dog': 0.4, 'barks': 0.6},
            'V': {'dog': 0.1, 'barks': 0.9},
        }
        hmm = HMM(3, tagset, trans, out)
        tagger = ViterbiTagger(hmm)

        x = 'the dog barks'.split()
        y = tagger.tag(x)

        pi = {
            0: {
                ('<s>', '<s>'): (log2(1.0), []),
            },
            1: {
                ('<s>', 'D'): (log2(1.0), ['D']),
            },
            2: {
                ('D', 'N'): (log2(0.8 * 0.4), ['D', 'N']),
                ('D', 'V'): (log2(0.2 * 0.1), ['D', 'V']),
            },
            3: {
                ('N', 'V'): (log2(0.8 * 0.4 * 0.8 * 0.9), ['D', 'N', 'V']),
                ('N', 'N'): (log2(0.8 * 0.4 * 0.2 * 0.6), ['D', 'N', 'N']),
                ('V', 'V'): (log2(0.2 * 0.1 * 0.8 * 0.9), ['D', 'V', 'V']),
                ('V', 'N'): (log2(0.2 * 0.1 * 0.2 * 0.6), ['D', 'V', 'N']),
            }
        }
        self.assertEqualPi(tagger._pi, pi)

        self.assertEqual(y, 'D N V'.split())

    def assertEqualPi(self, pi1, pi2):
        self.assertEqual(set(pi1.keys()), set(pi2.keys()))

        for k in pi1.keys():
            d1, d2 = pi1[k], pi2[k]
            self.assertEqual(d1.keys(), d2.keys(), k)
            for k2 in d1.keys():
                prob1, tags1 = d1[k2]
                prob2, tags2 = d2[k2]
                self.assertAlmostEqual(prob1, prob2)
                self.assertEqual(tags1, tags2)

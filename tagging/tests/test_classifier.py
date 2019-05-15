# https://docs.python.org/3/library/unittest.html
from unittest import TestCase

from tagging.classifier import feature_dict


class TestFeatureDict(TestCase):

    def test_feature_dict(self):
        sent = 'El gato come pescado .'.split()

        fdict = {
            'w': 'el',    # lower
            'wu': False,  # isupper
            'wt': True,   # istitle
            'wd': False,  # isdigit
            'pw': '<s>',
            'nw': 'gato',
            'nwu': False,
            'nwt': False,
            'nwd': False,
        }

        self.assertEqual(feature_dict(sent, 0), fdict)

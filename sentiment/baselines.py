from itertools import repeat
from collections import Counter

import numpy as np
from sklearn.dummy import DummyClassifier


class MostFrequent(object):

	def __init__(self):
		self._clf = DummyClassifier('most_frequent')

	def fit(self, X, y):
		# X is ignored
		dummy = np.empty(1)
		self._clf.fit([dummy for _ in X], y)

	def predict(self, X):
		# X is ignored
		dummy = np.empty(1)
		return self._clf.predict([dummy for _ in X])

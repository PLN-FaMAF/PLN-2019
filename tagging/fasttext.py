import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from fastText import load_model


class FasttextDictVectorizer(BaseEstimator, TransformerMixin):
    """Vectorizer for fastText embeddings in binary format.
    """

    def __init__(self, filename, keys):
        """
        filename -- Binary model file with embeddings.
        keys -- list of keys that contain the words in the feature dict.s
        """
        self._filename = filename
        self._keys = keys
        self._model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self._model is None:
            self._model = load_model(self._filename)
        m = self._model
        result = []
        for x in X:
            result.append(np.concatenate([m.get_word_vector(x[k]) for k in self._keys]))
        return result

    def __getstate__(self):
        """Return internal state for pickling, omitting unneeded objects.
        """
        state = self.__dict__
        state['_model'] = None
        return state

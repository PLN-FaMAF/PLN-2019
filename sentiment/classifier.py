from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


classifiers = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}


class SentimentClassifier(object):

    def __init__(self, clf='svm'):
        """
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        """
        self._clf = clf
        self._pipeline = pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', classifiers[clf]()),
        ])

    def fit(self, X, y):
        self._pipeline.fit(X, y)

    def predict(self, X):
        return self._pipeline.predict(X)

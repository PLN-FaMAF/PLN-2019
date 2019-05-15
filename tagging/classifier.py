from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


classifiers = {
    'lr': LogisticRegression,
    'svm': LinearSVC,
}


def feature_dict(sent, i):
    """Feature dictionary for a given sentence and position.

    sent -- the sentence.
    i -- the position.
    """
    # WORK HERE!!
    return {}


class ClassifierTagger:
    """Simple and fast classifier based tagger.
    """

    def __init__(self, tagged_sents, clf='lr'):
        """
        clf -- classifying model, one of 'svm', 'lr' (default: 'lr').
        """
        # WORK HERE!!

    def fit(self, tagged_sents):
        """
        Train.

        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        # WORK HERE!!

    def tag_sents(self, sents):
        """Tag sentences.

        sent -- the sentences.
        """
        # WORK HERE!!

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        # WORK HERE!!

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        # WORK HERE!!

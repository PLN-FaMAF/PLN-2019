"""Train a Sentiment Analysis model.

Usage:
  train.py [options] -i <corpus> -o <file>
  train.py -h | --help

Options:
  -i <corpus>   Training corpus.
  -m <model>    Model to use [default: basemf]:
                  basemf: Most frequent sentiment
                  clf: Machine Learning Classifier
  -c <clf>      Classifier to use if the model is a MEMM [default: svm]:
                  maxent: Maximum Entropy (i.e. Logistic Regression)
                  svm: Support Vector Machine
                  mnb: Multinomial Bayes
  -o <file>    Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from sentiment.tass import InterTASSReader
from sentiment.baselines import MostFrequent
from sentiment.classifier import SentimentClassifier


models = {
    'basemf': MostFrequent,
    'clf': SentimentClassifier,
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load corpora
    corpus = opts['-i']
    reader = InterTASSReader(corpus)
    X, y = list(reader.X()), list(reader.y())

    # train model
    model_type = opts['-m']
    if model_type == 'clf':
        model = models[model_type](clf=opts['-c'])
    else:
        model = models[model_type]()  # baseline

    model.fit(X, y)

    # save model
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()

from collections import namedtuple

from sklearn.metrics import confusion_matrix


def f1(prec, rec):
    if prec + rec > 0.0:
        result = 2 * prec * rec / (prec + rec)
    else:
        result = 0.0
    return result


class Evaluator(object):

    def evaluate(self, y_true, y_pred):
        self._y_true = y_true
        self._y_pred = y_pred
        self._labels = labels = 'P N NEU NONE'.split()
        self._CM = CM = confusion_matrix(y_true, y_pred, labels=labels)

        # accuracy
        self._hits = hits = CM.diagonal().sum()  # also m.trace()
        self._total = total = CM.sum()
        self._acc = acc = float(hits) / total * 100.0

        # per-label precision, recall and F1
        Metrics = namedtuple('Metrics', 'hits pred true prec rec')
        self._label_metrics = metrics = {}
        precs, recs = [], []
        for i, label in enumerate(labels):
            hits = CM[i, i]
            pred = CM[:, i].sum()  # i-th column
            true = CM[i, :].sum()  # i-th row
            if pred > 0.0:
                prec = float(hits) / pred * 100.0
            else:
                prec = 100.0
            if true > 0.0:
                rec = float(hits) / true * 100.0
            else:
                rec = 100.0
            metrics[label] = Metrics(hits, pred, true, prec, rec)
            precs.append(prec)
            recs.append(rec)

        # macro-averaged precision, recall and F1
        self._macro_prec = sum(precs) / len(precs)
        self._macro_rec = sum(recs) / len(recs)

    def accuracy(self):
        return self._acc

    def macro_f1(self):
        return f1(self._macro_prec, self._macro_rec)

    def print_results(self):
        labels = self._labels
        CM = self._CM
        metrics = self._label_metrics

        # print per-label precision, recall and F1
        for label in labels:
            print('Sentiment {}:'.format(label))
            m = metrics[label]
            print('  Precision: {:2.2f}% ({}/{})'.format(m.prec, m.hits, m.pred))
            print('  Recall: {:2.2f}% ({}/{})'.format(m.rec, m.hits, m.true))
            print('  F1: {:2.2f}%'.format(f1(m.prec, m.rec)))

        hits = self._hits
        total = self._total
        acc = self._acc
        macro_prec = self._macro_prec
        macro_rec = self._macro_rec
        print('Accuracy: {:2.2f}% ({}/{})'.format(acc, hits, total))
        print('Macro-Precision: {:2.2f}%'.format(macro_prec))
        print('Macro-Recall: {:2.2f}%'.format(macro_rec))
        print('Macro-F1: {:2.2f}%'.format(f1(macro_prec, macro_rec)))

    def print_confusion_matrix(self):
        labels = self._labels
        CM = self._CM

        # confusion matrix
        for label in labels:
            print('\t{}'.format(label), end='')
        print('')

        # print table rows
        for i, label1 in enumerate(labels):
            print('{}\t'.format(label1), end='')
            for j, label2 in enumerate(labels):
                print('{}\t'.format(CM[i, j]), end='')
            print('')

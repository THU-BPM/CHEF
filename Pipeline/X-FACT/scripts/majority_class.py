import os, sys

from sklearn.metrics import  f1_score

MAJORITY_CLASS='false'


def load_labels(filename):
    labels = []

    with open(filename, 'r') as fp:
        for line in fp:
            label = line.strip().split('\t')[-1].lower()
            labels.append(label)

    return labels



if __name__=="__main__":

    labels = load_labels(sys.argv[1])

    preds = [MAJORITY_CLASS]*len(labels)

    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')

    print('F1 score for majority label {} : {}'.format(MAJORITY_CLASS, f1))

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM
)
import torch
import os
import re
import sys
import copy
import math
import random
import json, csv
import itertools
import time, datetime
from time import sleep
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, 
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn import (
    svm, 
    linear_model,
    preprocessing
)
from sklearn.model_selection import KFold
import numpy as np
from Semantic_Ranker import cosSimilarity

# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = torch.device("cuda")

def main():
    # labelEvidence()
    # # count tfidf
    # sents_list = []
    # data_list = json.load(open('EvidenceExtract/isgoldLabel.json', 'r', encoding='utf-8'))
    # for row in data_list:
    #     sents = [ev['text'] for ev in row['evidence']]
    #     sents_list.append(sents)
    # countTfIDF(sents_list)
    # count similarity
    data_list = json.load(open('EvidenceExtract/isgoldLabel.json', 'r', encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")
    model = model.to(device)
    similarityValues = []
    for row in tqdm(data_list):
        claim = row['claim']
        sims = []
        for ev in row['evidence']:
            sims.append(cosSimilarity(claim, ev['text'], model, tokenizer))
        similarityValues.append(sims)
        with open('EvidenceExtract/similarityValuesA.json', 'a+', encoding='utf-8') as f:
            tmp = json.dumps(sims, ensure_ascii=False)
            print(tmp, file=f)
    with open('EvidenceExtract/similarityValues.json', 'w', encoding='utf-8') as f:
        json.dump(similarityValues, f, indent=2, ensure_ascii=False)

    similaritys_list = json.load(open('EvidenceExtract/similarityValues.json', 'r', encoding='utf-8'))
    isgold_list = json.load(open('EvidenceExtract/isgoldLabel.json', 'r', encoding='utf-8'))
    tfidfvalues_list = json.load(open('EvidenceExtract/tfidfValues.json', 'r', encoding='utf-8'))
    X_trans = np.array([])
    y_trans = np.array([])
    for i in tqdm(range(len(isgold_list))):
        similaritys = similaritys_list[i]
        obj = isgold_list[i]
        tfidfvalues = tfidfvalues_list[i]
        y = [ev['isgold'] for ev in obj['evidence']]
        if np.all(np.array(y) == 0) or np.all(np.array(y) == 1):
            continue
        X = []
        for j in range(len(similaritys)):
            X.append([similaritys[j], tfidfvalues[j]])
        X, y = transform_pairwise(np.array(X), np.array(y))
        try:
            if len(X_trans) == 0:
                X_trans = X
                y_trans = y
            else:
                X_trans = np.concatenate((X_trans, X), axis=0)
                y_trans = np.concatenate((y_trans, y), axis=0)
        except Exception as e:
            print(e)
    print(f'X length is {len(X)}')
    rank_svm = RankSVM().fit_notrans(X_trans, y_trans)

    low = 0
    top5_sentences = []
    for i in tqdm(range(len(isgold_list))):
        similaritys = similaritys_list[i]
        obj = isgold_list[i]
        tfidfvalues = tfidfvalues_list[i]
        X = []
        for j in range(len(similaritys)):
            X.append([similaritys[j], tfidfvalues[j]])
        X = np.array(X)
        try:
            rank = rank_svm.predict(X).tolist()
            low += len(obj['evidence'])
            top5 = []
            for j in rank[-5:][::-1]:
                top5.append(obj['evidence'][j]['text'])
            top5_sentences.append(top5)
        except Exception as e:
            top5_sentences.append(['' for _ in range(5)])
    with open('hybrid_result.json', 'w', encoding='utf-8') as f:
        json.dump(top5_sentences, f, indent=2, ensure_ascii=False)



def labelEvidence():
    data_list = json.load(open('test.json', 'r', encoding='utf-8')) \
        + json.load(open('train.json', 'r', encoding='utf-8'))
    label_evidences = []
    for row in tqdm(data_list):
        ev_sents = []
        # ev_sents += re.split(r'[？：。！.“”…\t\n]', row['content'])
        for ev in row['evidence'].values():
            ev_sents += re.split(r'[？：。！.“”…\t\n]', ev['text'])
        ev_sents = [sent for sent in ev_sents if len(sent) > 5]
        obj = {
            'claim': row['claim'],
            'evidence': []
        }
        ev_isgold = {}
        for ev_sent in ev_sents:
            ev_isgold[ev_sent] = 0
            for ev in row['gold evidence'].values():
                gold_evidence = ev['text']
                if len(gold_evidence) == 0:
                    continue
                c1 = Counter(ev_sent)
                c2 = Counter(gold_evidence)
                miss_num = 0
                for character, count in c2.items():
                    miss_num += abs(count - c1[character])
                if miss_num / len(gold_evidence) < 0.3:
                    ev_isgold[ev_sent] = 1
        for sent, val in ev_isgold.items():
            obj['evidence'].append({
                'text': sent,
                'isgold': val
            })
        label_evidences.append(obj)
    with open('EvidenceExtract/isgoldLabel.json', 'w', encoding='utf-8') as f:
        json.dump(label_evidences, f, indent=2, ensure_ascii=False)

def countTfIDF(sents_list):
    """
    sents_list = [
        [document1_sent1, document1_sent2, ...],
        [document2_sent1, document2_sent2, ...],
        ...
    ]
    """
    document_count = len(sents_list)
    # count idf
    idf = Counter()
    for sents in sents_list:
        nsents = list(set(sents))
        for sent in nsents:
            idf[sent] += 1
    for key in idf.keys():
        idf[key] = math.log(document_count/idf[key]+1)
    result = []
    # count tf
    for sents in sents_list:
        tf = Counter()
        for sent in sents:
            tf[sent] += 1
        for key in tf.keys():
            tf[key] = tf[key] / len(tf.keys())
        tf_idf = Counter()
        for sent in sents:
            tf_idf[sent] = tf[sent] * idf[sent]
        result.append(list(tf_idf.values()))
    with open('EvidenceExtract/tfidfValues.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)


def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        Returns
        -------
        self
        """
        X_trans, y_trans = transform_pairwise(X, y)
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def fit_notrans(self, X, y):
        super(RankSVM, self).fit(X, y)
        return self

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            return np.argsort(np.dot(X, self.coef_.T).ravel())
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)


if __name__ == '__main__':
    main()
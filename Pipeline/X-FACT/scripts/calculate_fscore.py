import sys
import pprint
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def load_predictions(filename):

    pred = []
    gold = []

    with open(filename, 'r') as fp:
        for i, line in enumerate(fp):
            if i == 0:
                continue
            arr = line.strip().split('\t')
            pred.append(arr[0])
            gold.append(arr[1])

    #pred = ['false']*len(gold)
    return pred, gold

def most_common(lst):
    return max(set(lst), key=lst.count)


def calculate_score(pred, gold):

    score_mat = {

    'true':
            {
                'true':1.0,
                'mostly true':0.75,
                'partly true/misleading': 0.5,
                'complicated/hard to categorise': 0.0,
                'other': 0.0,
                'mostly false': 0.0,
                'false': 0.0
            },


    'mostly true':
            {
                'true':0.75,
                'mostly true':1.0,
                'partly true/misleading': 0.75,
                'complicated/hard to categorise': 0.0,
                'other': 0.0,
                'mostly false': 0.0,
                'false': 0.0
            },

    'partly true/misleading':
            {
                'true':0.25,
                'mostly true':0.50,
                'partly true/misleading': 1.0,
                'complicated/hard to categorise': 0.5,
                'other': 0.5,
                'mostly false': 0.50,
                'false': 0.25
            },



    'complicated/hard to categorise':
            {
                'true':0.0,
                'mostly true':0.0,
                'partly true/misleading': 0.0,
                'complicated/hard to categorise': 1.0,
                'other': 0.0,
                'mostly false': 0.0,
                'false': 0.0
            },


    'other':
            {
                'true':0.0,
                'mostly true':0.0,
                'partly true/misleading': 0,
                'complicated/hard to categorise': 0,
                'other': 1.0,
                'mostly false': 0.0,
                'false': 0.0
            },


    'mostly false':
            {
                'true':0.25,
                'mostly true':0.5,
                'partly true/misleading': 0.5,
                'complicated/hard to categorise': 0.5,
                'other': 0.5,
                'mostly false': 1.00,
                'false': 0.75
            },


    'false':
            {
                'true':0.0,
                'mostly true':0.25,
                'partly true/misleading': 0.5,
                'complicated/hard to categorise': 0.5,
                'other': 0.5,
                'mostly false': 0.75,
                'false': 1.0
            }


    }

    assert len(preds) == len(gold)

    total = 0.0
    for i in range(len(preds)):
        g = gold[i]
        total += score_mat[g][preds[i]]


    score = total/len(preds)
    print('Final Score is ', score)

def load_examples(filename):

    examples = []

    with open(filename, 'r') as fp:
        for i, line in enumerate(fp):
            if i == 0:
                continue
            arr = line.strip().split('\t')
            lang = arr[0]
            site = arr[1]
            examples.append(arr)

    return examples


def f_score(ex, typ='macro'):

    preds, gold = zip(*ex)


    return f1_score(gold, preds, average=typ)





def calculate_fscore(preds, gold, examples):


    metrics = {}

    for i, ex in enumerate(examples):
        t = (ex[0], ex[1])
        if t not in metrics:
            metrics[t] = []
        metrics[t].append((preds[i], gold[i]))


    scores = {}
    scores_micro = {}

    l = []

    for key, val in metrics.items():
        scores[key] = f_score(val)
        scores_micro[key] = f_score(val, typ='macro')
        l.append(scores_micro[key])

        #print('--'*40)
        #print('--'*40)
        #print(key)
        #p, g = zip(*val)
        #print(classification_report(g, p))
        #print('--'*40)
        #print('--'*40)

    #pprint.pprint(scores)
    #pprint.pprint(scores_micro)

    print('Average F1 Macro : ')
    print(sum(l)/len(l))


if __name__=="__main__":


    predict_file = sys.argv[1]

    preds, gold = load_predictions(predict_file)

    examples = load_examples(sys.argv[2])


    calculate_fscore(preds, gold, examples)

    #calculate_score(preds, gold)
    # calculate majority score
    #predictions = {}
    #golds = {}

    #for i, ex in enumerate(examples):
    #    t = (ex[0], ex[1])
    #    if t not in metrics:
    #        golds[t] = []
    #    golds[t].append(gold[i])

    #majority = {}
    #for key, val in golds.items():
    #    majority[key] = [most_common(val)]*len(val))

    #print(majority)

    #for key, val in majority.items():
        #calculate_fscore(majority[key], val, examples)
    common = most_common(gold)
    print(common)
    majority_preds = ['false']*len(gold)
    #calculate_fscore(majority_preds, gold, examples)


    print(f_score((zip(majority_preds, gold)), typ='macro'))

from collections import Counter
import sys, json, re
import math

def main():
    test_data = json.load(open('test.json', 'r', encoding='utf-8'))
    train_data = json.load(open('train.json', 'r', encoding='utf-8'))
    print(len(test_data))
    print(len(train_data))
    rowData = test_data + train_data
    # count idf
    idf = {}
    idf = Counter()
    for row in rowData:
        sentList = []
        for ev in row['evidence'].values():
            sentList += re.split(r'[？：。！（）.“”…\t\n]', ev['text'])
        # remove duplicates
        sentList = list(set(sentList))
        for sent in sentList:
            idf[sent] += 1
    document_count = len(rowData)
    for key in idf.keys():
        idf[key] = math.log(document_count/idf[key]+1)
    
    # count tf and select
    get_evidence_num = 5
    tdidf_ev = []
    for row in rowData:
        sentList = []
        for ev in row['evidence'].values():
            sentList += re.split(r'[？：。！（）.“”…\t\n]', ev['text'])
        tf = Counter()
        for sent in sentList:
            tf[sent] += 1
        for key in tf.keys():
            tf[key] = tf[key] / len(tf.keys())
        tf_idf = Counter()
        for sent in sentList:
            tf_idf[sent] = tf[sent] * idf[sent]
        tmp = list(tf_idf.items())
        tmp.sort(key=lambda s: s[1], reverse=True)
        tmp = [ele[0] for ele in tmp if len(ele[0]) > 5]
        tdidf_ev.append(tmp[:get_evidence_num])
    
    # with open('testTfidfSents.json', 'w', encoding='utf-8') as f:
    #     json.dump(tdidf_ev[:len(test_data)], f, indent=2, ensure_ascii=False)
    # with open('trainTfidfSents.json', 'w', encoding='utf-8') as f:
    #     json.dump(tdidf_ev[len(test_data):], f, indent=2, ensure_ascii=False)
    with open('tfidfSents.json', 'w', encoding='utf-8') as f:
        json.dump(tdidf_ev, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()


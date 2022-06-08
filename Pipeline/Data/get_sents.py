import os, sys
import json


def main():
    chef_test_train()
    # claim_ranksvm()
    # claim_cossim()
    # claim_tfidf()
    # claim_gold()

def chef_test_train():
    data = json.load(open('test.json', 'r', encoding='utf-8')) \
        + json.load(open('train.json', 'r', encoding='utf-8'))
    tfidf_sents_list = json.load(open('tfidfSents.json', 'r', encoding='utf-8'))
    cossim_sents_lines = open('semantic_result.jsonl', 'r', encoding='utf-8')\
        .readlines()[-len(data):]
    svm_sents_list = json.load(open('hybrid_result.json', 'r', encoding='utf-8'))
    for index in range(len(data)):
        row = data[index]
        tfidf_sents = tfidf_sents_list[index]
        cossim_sents = json.loads(cossim_sents_lines[index].strip())
        cossim_sents = [t[0] for t in cossim_sents]
        svm_sents = svm_sents_list[index]
        row['tfidf'] = tfidf_sents
        row['cossim'] = cossim_sents
        row['ranksvm'] = svm_sents
    with open('chef/CHEF_train.json', 'w', encoding='utf-8') as f:
        json.dump(data[999:], f, indent=2, ensure_ascii=False)
    with open('chef/CHEF_test.json', 'w', encoding='utf-8') as f:
        json.dump(data[:999], f, indent=2, ensure_ascii=False)
    

def claim_ranksvm():
    data = json.load(open('test.json', 'r', encoding='utf-8')) \
        + json.load(open('train.json', 'r', encoding='utf-8'))
    svm_sents_list = json.load(open('hybrid_result.json', 'r', encoding='utf-8'))
    sent_list = []
    for index in range(len(data)):
        row = data[index]
        sents = svm_sents_list[index]
        sent = claim_evidences2bert_type(row['claim'], sents)
        sent_list.append(sent)
    with open('claim_ranksvm.json', 'w', encoding='utf-8') as f:
        json.dump(sent_list, f, indent=2, ensure_ascii=False)
    

def claim_cossim():
    data = json.load(open('test.json', 'r', encoding='utf-8')) \
        + json.load(open('train.json', 'r', encoding='utf-8'))
    cossim_sents_lines = open('semantic_result.jsonl', 'r', encoding='utf-8')\
        .readlines()[-len(data):]
    sent_list = []
    for index in range(len(data)):
        row = data[index]
        sents = json.loads(cossim_sents_lines[index].strip())
        sents = [t[0] for t in sents]
        sent = claim_evidences2bert_type(row['claim'], sents)
        sent_list.append(sent)
    with open('claim_cossim.json', 'w', encoding='utf-8') as f:
        json.dump(sent_list, f, indent=2, ensure_ascii=False)

def claim_gold():
    data = json.load(open('dev.json', 'r', encoding='utf-8')) \
        + json.load(open('train.json', 'r', encoding='utf-8'))
    sent_list = []
    for row in data:
        evidences = [row['gold evidence'][str(i)]['text'] for i in range(5) if row['gold evidence'][str(i)]['text'] != '']
        sent = claim_evidences2bert_type(row['claim'], evidences)
        sent_list.append(sent)
    with open('claim_gold.json', 'w', encoding='utf-8') as f:
        json.dump(sent_list, f, indent=2, ensure_ascii=False)
    
    
def claim_tfidf():
    data = json.load(open('test.json', 'r', encoding='utf-8')) \
        + json.load(open('train.json', 'r', encoding='utf-8'))
    tfidf_sents_list = json.load(open('tfidfSents.json', 'r', encoding='utf-8'))
    sent_list = []
    for index in range(len(data)):
        row = data[index]
        sents = tfidf_sents_list[index]
        sent = claim_evidences2bert_type(row['claim'], sents)
        sent_list.append(sent)
    with open('claim_tfidf.json', 'w', encoding='utf-8') as f:
        json.dump(sent_list, f, indent=2, ensure_ascii=False)
        

def claim_evidences2bert_type(claim: str, evidences: list):
    evlist = [claim] + evidences
    return f"[CLS] {' [SEP] '.join(evlist)} [SEP]"

if __name__ == '__main__':
    main()
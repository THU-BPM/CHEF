import json, random

# dataList = json.load(open('Piyao+Taiwan_3609_newId.json', 'r', encoding='utf-8'))
# tfidf_data = json.load(open('top5Tfidf.json', 'r', encoding='utf-8'))
# f = open('top5Similar.json', 'r', encoding='utf-8')
# lines = f.readlines()
# for i, row in enumerate(dataList):
#     row['tfidf'] = tfidf_data[i]
#     evs = json.loads(lines[i].strip())
#     row['sim'] = [ev[0] for ev in evs]
# total_num = len(dataList)
# random.shuffle(dataList)
# offset = int(total_num*0.2)
# dev_set = dataList[:offset]
# train_set = dataList[offset:]
# with open('CHEF_train.json', 'w', encoding='utf-8') as f:
#     json.dump(train_set, f, indent=2, ensure_ascii=False)
# with open('CHEF_test.json', 'w', encoding='utf-8') as f:
#     json.dump(dev_set, f, indent=2, ensure_ascii=False)
dataList = json.load(open('CHEF_test.json', 'r', encoding='utf-8'))
# rsvm = json.load(open('rankSVMEvidence.json', 'r', encoding='utf-8'))
snippets_list = json.load(open('conv_snippets.json', 'r', encoding='utf-8'))
for row in dataList:
    row['snippets'] = snippets_list[row['claimId']][:5]
with open('CHEF_test.json', 'w', encoding='utf-8') as f:
    json.dump(dataList, f, indent=2, ensure_ascii=False)
import os, sys
import json

def main():
    # Append(['final.json', 'final_label2.json'])
    Check(['test.json', 'train.json', 'dev.json'])
    # j2row()
    
def Append(filenames: list):
    data = []
    for filename in filenames:
        data = data + json.load(open(filename, 'r', encoding='utf-8'))
    with open('CHEF.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def Check(filenames: list):
    for filename in filenames:
        data = json.load(open(filename, 'r', encoding='utf-8'))
        new_data = []
        for row in data:
            row['domain'] = row['domain'][:2]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def j2row():
    data = json.load(open('第二阶段727条数据.json', 'r', encoding='utf-8'))
    new_data = []
    for row in data:
        if row.get('data') is None:
            new_data.append(row)
        else:
            new_data.append(row['data'])
    with open('第二阶段727条数据1.json', 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False) 

if __name__ == '__main__':
    main()

# Bert Classification for CHEF

Use `BertForSequenceClassification` in transformers to do the classification

## Preprocess
You can run the code in `data` folder to get 'TF-IDF', 'Semantic Similarity' or 'RankSVM' result.

## How to run
First, you need to transform data to sentences. Each sentence in the form of 
```
[CLS] claim [SEP] evidence1 [SEP] evidence2 [SEP] ...(if there are more evidences)
```
Save the sentences as a list and save the list to a json file. Each evidence type(gold, TF-IDF, etc) in a file. After that, edit the file name in `train.py` line 171-200, then the program will load the sentences to train.

To trian:
```
pip install -r requirements.txt
python train.py
```
# Kernel Graph Attention Network (KGAT) for CHEF

This code is modified from [Kernel Graph Attention Network](https://github.com/thunlp/KernelGAT).

## How to run
First, follow `pre_README.md` to set up environment.

For training model, while the evidence_type can be 'gold', 'tfidf', 'cossim', 'ranksvm', 'snippet':
```
TRAIN_DATA_DIR=<path_to_data>
DEV_DATA_DIR=<path_to_data>

python -u train.py --outdir ../checkpoint/kgat \
--train_path TRAIN_DATA_DIR \
--valid_path DEV_DATA_DIR \
--evidence_type gold \
--bert_pretrain bert-base-chinese \
--num_train_epochs 16 \
--max_len 128
```
# python train.py --outdir ../checkpoint/kgat \
# --train_path ../data/bert_train.json \
# --valid_path ../data/bert_dev.json \
# --postpretrain ../pretrain/save_model/model.best.pt \
# --bert_pretrain ../bert_base

python -u train.py --outdir checkpoint/kgat \
--train_path ../data/chef/CHEF_train.json \
--valid_path ../data/chef/CHEF_test.json \
--evidence_type gold \
--bert_pretrain bert-base-chinese \
--num_train_epochs 16 \
--max_len 128
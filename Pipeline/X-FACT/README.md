# X-Fact for CHEF

This code is modified from [x-fact](https://github.com/utahnlp/x-fact).

## How to run
Follow `pre_README.md` to set up the environment.
```
cd transformers
```
And then, for training the claim-only model:
```
python -u examples/text-classification/run_xfact.py \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --data_dir ../../data/chef/ \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size 8 \
    --sources all \
    --learning_rate 2e-5 \
    --num_train_epochs 8.0 \
    --save_steps 40000  \
    --per_gpu_train_batch_size 8 \
    --overwrite_output_dir \
    --save_every_epoch \
    --evaluate_during_training \
    --output_dir models/chef/seed_1/ \
    --seed 1 \
    --overwrite_cache
```
For training Attention-based Evidence Aggregator, the evidence type can be 'gold', 'tfidf', 'cossim', 'ranksvm', 'snippet':
```
python -u examples/text-classification/run_xfact_evidence_attention.py \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --data_dir ../../data/chef/ \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 8 \
    --sources all \
    --learning_rate 2e-5 \
    --num_train_epochs 10.0 \
    --save_steps 40000  \
    --per_gpu_train_batch_size 8 \
    --overwrite_output_dir \
    --save_every_epoch \
    --evaluate_during_training \
    --output_dir models/evidences_attn_metadata/mBERT_seed_1/ \
    --seed 1 \
    --overwrite_cache \
    --evidence_type tfidf 
```
# x-fact
Official Code and Data repository of our ACL 2021 paper [X-FACT: A New Benchmark Dataset for Multilingual Fact Checking](https://arxiv.org/abs/2106.09248).

**Note**: This repository is built on a slightly older fork of the `transformers` repository. We will release the code compatible with the newest `transformers` repo in a few days.

## 0. Prerequisites: Installation (python 3.6+). 

#### Create a virtual environment and install the provided `transformers` code along with requirements.
```
git clone https://github.com/utahnlp/x-fact/
cd x-fact/
python3 -m venv env_transformers
source env_transformers/bin/activate
cd transformers
pip install -r examples/requirements.txt
pip install --editable ./
```
 **Tip**: If you are using cuda>=11, you can upgrade the pytorch to a specific version by specifying the mirror, as in the following:
 ```
 pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
 ```

## 1. Dataset Details

#### You can find the whole dataset in the `data/` directory.
```
data
│──── train.all.tsv 					# Training Data
│──── dev.all.tsv 					# Development Data
│──── test.all.tsv 					# In-Domain Test Data
│──── ood.tsv 					        # Out-of-Domain Test Data
│──── zeroshot.tsv 					# Zero-Shot Evaluation Test Data
│──── label_maps/				        # Contains Manually Created label mappings for each website
      ├── master_mapping.tsv			        # Tab Separated file with label mapping from all possible translated labels 
      ├── factual.ro.txt			        # Label mappings for factual.ro (from original language to translated in English)
```

## 2. Training the models
#### For training the **claim-only** model with **metadata**, please run the following command:
```
DATA_DIR=<path_to_data>

python -u examples/text-classification/run_xfact.py \ 
	--model_name_or_path bert-base-multilingual-cased \ 
	--do_train \ 
	--do_eval \ 
	--data_dir $DATA_DIR \ 
	--max_seq_length 512 \ 
	--per_gpu_eval_batch_size 8 \ 
	--sources all \ 
	--learning_rate 2e-5 \ 
	--num_train_epochs 10.0 \ 
	--save_steps 40000  \ 
	--per_gpu_train_batch_size 8 \ 
	--overwrite_output_dir \ 
	--save_every_epoch \ 
	--evaluate_during_training \
	--use_metadata \ 
	--output_dir models/claim_only_metadata/mBERT_seed_1/ \ 
	--seed 1

Most of the arguments are as defined in the Huggingface Transformers, but new ones are described below:

-- use_metadata: If to use the metadata
-- save_every_epoch: Not in HF transformers, saves after every epoch, and at the end of training saves the best validation model in the model_dir 
-- sources: selects the fact-check source to do training on, default: 'all'. Selects 'train.<source>.tsv' for training.
-- seed: set a particular seed

```

#### For training the **Attn-EA** model with **metadata** (Attention-based Evidence Aggregator), please run the following command:
```
DATA_DIR=<path_to_data>

python -u examples/text-classification/run_xfact_evidence_attention.py \ 
	--model_name_or_path bert-base-multilingual-cased \ 
	--do_train \ 
	--do_eval \ 
	--data_dir $DATA_DIR \ 
	--max_seq_length 360 \ 
	--per_gpu_eval_batch_size 8 \ 
	--sources all \ 
	--learning_rate 2e-5 \ 
	--num_train_epochs 10.0 \ 
	--save_steps 40000  \ 
	--per_gpu_train_batch_size 12 \ 
	--overwrite_output_dir \ 
	--save_every_epoch \ 
	--evaluate_during_training \
	--use_metadata \ 
	--output_dir models/evidences_attn_metadata/mBERT_seed_1/ \ 
	--seed 1
```

## 3. Evaluating the models
```
DATA_DIR=<path_to_data>

python -u examples/text-classification/run_xfact_evidence_attention.py \ 
	--do_eval \ 
	--evaluate_file test.all.tsv \
	--data_dir $DATA_DIR \ 
	--max_seq_length 360 \ 
	--per_gpu_eval_batch_size 8 \ 
	--sources all \ 
	--overwrite_output_dir \ 
	--use_metadata \ 
	--output_dir models/evidences_attn_metadata/mBERT_seed_1/ \ 
	--model_name_or_path models/evidences_attn_metadata/mBERT_seed_1/ \ 

-- evaluate_file: The name of the file we want to evaluate (not the absolute path), file should be present in $DATA_DIR
```
The above command runs evaluation for In-domain test set. For running the evaluation for OOD test, and Zeroshot test, change the `--evaluate_file` argument to `ood.tsv` and `zeroshot.tsv` respectively.

For running evaluation for claim-only model, use the same arguments and only change the filename to 
`examples/text-classification/run_xfact.py`.

**Note**: The results reported in the paper are averaged over four random seeds: `[1, 2, 3, 4]`.

## Citation
If you use the dataset and, or code from our work, please cite
```
@inproceedings{gupta2021xfact,
      title={{X-FACT: A New Benchmark Dataset for Multilingual Fact Checking}}, 
      author={Gupta, Ashim and Srikumar, Vivek},
      booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics",      
      month = jul,
      year = "2021",
      address = "Online",
      publisher = "Association for Computational Linguistics",
}
```

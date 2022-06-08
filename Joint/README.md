# Joint System for CHEF

This code is modified from [interpretable_predictions](https://github.com/bastings/interpretable_predictions).

## Installation

You need to have Python 3.6 or higher installed.
It is recommended that you use a virtual environment:
```
sudo pip3 install -U virtualenv
virtualenv --system-site-packages -p python3 ./my_venv
source ./my_venv/bin/activate
```

Install all required Python packages using:
```
pip install -r requirements.txt
```

## How to run

Put `test.json` and `train.json` into `data` folder.

To train the latent rationale model to select 30% of the evidences:
```
python -m latent_rationale.sst.CHEFtrain --selection 0.3 --save_path results/sst/chef --lr 0.00001 --logdir chef
```
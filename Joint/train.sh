CUDA_VISIBLE_DEVICES=2 python -m latent_rationale.sst.CHEFtrain --selection 0.3 --save_path results/sst/chef --lr 0.000005 --logdir lr000005

CUDA_VISIBLE_DEVICES=2 python -m latent_rationale.sst.CHEFtrain --selection 0.3 --save_path results/sst/chef --lr 0.000005 --logdir test

CUDA_VISIBLE_DEVICES=0 python -m latent_rationale.sst.CHEFtrain --selection 0.3 --save_path results/sst/chef --lr 0.000001 --logdir lr000001

CUDA_VISIBLE_DEVICES=1 python -m latent_rationale.sst.CHEFtrain --selection 0.3 --save_path results/sst/chef --lr 0.00001 --logdir lr00001

CUDA_VISIBLE_DEVICES=4 python -m latent_rationale.sst.CHEFtrain --selection 0.3 --save_path results/sst/chef --lr 0.000005 --nlayer 4 --logdir lr000005layer4
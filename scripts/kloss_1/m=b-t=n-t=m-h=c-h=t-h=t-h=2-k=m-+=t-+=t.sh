#!/bin/bash

#SBATCH --job-name=kloss_1
#SBATCH --output=joblogs/kloss_1_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gpus=p100:1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL

module load CUDA
module load cuDNN
module load miniconda
source activate salts

python tune.py model=bert \
	tuning=newverb_transitive_ext \
	tuning.which_args=model \
	hyperparameters.unfreezing=complete \
	hyperparameters.use_kl_baseline_loss=true \
	hyperparameters.mask_args=true \
	hyperparameters.max_epochs=250 \
	kl_loss_params.dataset=datamaker/datasets/miniboki-2022-04-01_22-58-30/miniboki \
	+debug=true \
	+use_gpu=true
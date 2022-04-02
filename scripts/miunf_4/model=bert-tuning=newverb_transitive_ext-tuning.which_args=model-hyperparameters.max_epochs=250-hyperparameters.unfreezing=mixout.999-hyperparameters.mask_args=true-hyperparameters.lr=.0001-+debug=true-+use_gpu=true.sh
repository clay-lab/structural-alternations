#!/bin/bash

#SBATCH --job-name=miunf_4
#SBATCH --output=joblogs/miunf_4_%j.txt
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
	hyperparameters.max_epochs=250 \
	hyperparameters.unfreezing=mixout.999 \
	hyperparameters.mask_args=true \
	hyperparameters.lr=.0001 \
	+debug=true \
	+use_gpu=true
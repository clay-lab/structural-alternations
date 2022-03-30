#!/bin/bash

#SBATCH --job-name=salts
#SBATCH --output=joblogs/salts_newverb_drink_miunf.9_%j.txt
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --gpus=p100:1
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --mail-type=ALL

module load CUDA
module load cuDNN
module load miniconda

source activate salts

python tune.py \
	model=bert \
	tuning=newverb_transitive_ext \
	tuning.which_args=drink \
	hyperparameters.max_epochs=100 \
	hyperparameters.unfreezing=mixout.9 \
	hyperparameters.lr=.001 \
	hyperparameters.mask_args=true \
	+debug=true \
	+use_gpu=true

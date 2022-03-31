#!/bin/bash

#SBATCH --job-name=salts
#SBATCH --output=joblogs/salts_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gpus=p100:1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL

module load CUDA
module load cuDNN
module load miniconda
source activate salts

python tune.py model=bert \
	tuning=newverb_transitive_ext \
	tuning.which_args=drink \
	hyperparameters.unfreezing=mixout.9 \
	hyperparameters.max_epochs=100 \
	+debug=true \
	+use_gpu=true
#!/bin/bash

#SBATCH --job-name=miunf_fix
#SBATCH --output=joblogs/miunf_fix_%j.txt
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

python model=bert \
	hyperparameters.lr=.0001 \
	hyperparameters.max_epochs=100 \
	tuning=newverb_transitive_ext \
	tuning.which_args=model \
	hyperparameters.unfreezing=mixout.9
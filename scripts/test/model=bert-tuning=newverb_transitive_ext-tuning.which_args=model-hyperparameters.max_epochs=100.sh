#!/bin/bash

#SBATCH --job-name=salts
#SBATCH --output=joblogs/salts_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=p100:1
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --time=01:30:00
#SBATCH --mail-type=ALL

module load CUDA
module load cuDNN
module load miniconda
source activate salts

python model=bert \
	tuning=newverb_transitive_ext \
	tuning.which_args=model \
	hyperparameters.max_epochs=100

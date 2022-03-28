#!/bin/bash

#SBATCH --job-name=testsalts
#SBATCH --output=logs/test_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus=p100:3
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

module load CUDA
module load cuDNN
module load miniconda

source activate salts
# cd ..

python tune.py -m \
	model=bert,distilbert,roberta \
	tuning=dative_DO_give_active,dative_DO_send_active,dative_PD_give_active,dative_PD_send_active \
	dev=best_matches \
	dev_exclude=mail \
	hyperparameters.max_epochs=10 \
	hyperparameters.strip_punct=false,true \
	hyperparameters.masked_tuning_style=always,bert,none,roberta \
	+use_gpu=true

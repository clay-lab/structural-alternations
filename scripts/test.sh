#!/bin/bash

#SBATCH --job-name=testsalts
#SBATCH --output=logs/test_log.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus=p100:3
#SBATCH --partition=gpu
#SBATCH --time=01:00:00

module load CUDA
module load cuDNN

source activate salts
cd ..

python tune.py -m \
	hydra/launcher=joblib \
	hydra.launcher.n_jobs=2 \
	model=bert,distilbert,roberta \
	'tuning=glob(dative*,exclude=[*large,*all,*mail*,*passive)' \
	dev=best_matches \
	dev_exclude=mail \
	hyperparameters.max_epochs=10 \
	hyperparameters.strip_punct=false,true \
	hyperparameters.masked_tuning_style=always,bert,none,roberta \
	+use_gpu=true
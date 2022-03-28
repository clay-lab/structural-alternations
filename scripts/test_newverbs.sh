#!/bin/bash

#SBATCH --job-name=testsalts
#SBATCH --output=logs/test_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus=p100:1
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL

module load CUDA
module load cuDNN
module load miniconda

source activate salts
# cd ..

python tune.py -m \
	model=bert,distilbert,roberta \
	tuning=newverb_transitive_ext \
	tuning.which_args=model,red_yellow,fruit_vegetable,animate_inanimate,inanimate_animate \
	hyperparameters.max_epochs=100 \
	hyperparameters.unfreezing=none,gradual1,gradual5,complete \
	hyperparameters.lr=.001,.0001 \
	hyperparameters.mask_args=false,true \
	+use_gpu=true

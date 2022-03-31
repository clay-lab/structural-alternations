#!/bin/bash

#SBATCH --job-name=salts
#SBATCH --output=joblogs/salts_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gpus=p100:1
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --mail-type=ALL

module load CUDA
module load cuDNN
module load miniconda
source activate salts

python checkpoint_dir=outputs/newverb_transitive_ext/bbert-amask-wpunc-nounf-lr0.001/bert_args-margs/2022-03-31_00-28-43 \
	data=syn_blork_ext
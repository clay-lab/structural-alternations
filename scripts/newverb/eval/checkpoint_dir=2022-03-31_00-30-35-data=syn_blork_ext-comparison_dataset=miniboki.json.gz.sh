#!/bin/bash

#SBATCH --job-name=salts
#SBATCH --output=joblogs/salts_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gpus=p100:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

module load CUDA
module load cuDNN
module load miniconda
source activate salts

python eval.py checkpoint_dir=outputs/newverb_transitive_ext/bbert-amask-wpunc-miunf0.9-lr0.0001/bert_args-margs/2022-03-31_00-30-35 \
	data=syn_blork_ext \
	comparison_dataset=datamaker/datasets/miniboki-2022-03-31_10-42-40/miniboki.json.gz

#!/bin/bash

#SBATCH --job-name=salts
#SBATCH --output=joblogs/salts_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=p100:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

# Run from structural-alternations dir using bash scripts/{name}.sh.
# It assumes it is being run from the higher directory. 
# Files will not end up in the right places if you run this from the scripts directory

module load CUDA
module load cuDNN
module load miniconda

source activate salts

# fill in what you want after 'python'
# note that when running on grace, use of hydra's glob resolver does not work, nor does using joblib for multithreading
py eval.py \
	data=syn_blork_ext \
	comparison_dataset=datamaker/datasets/miniboki-2022-03-28_22-06-50/miniboki.json.gz \
	checkpoint_dir=outputs/newverb_transitive_ext/bbert-amask-wpunc-miunf0.9-lr0.0001/drink_args-margs/2022-03-30_15-35-02 \
	+debug=true

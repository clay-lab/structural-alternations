#!/bin/bash

#SBATCH --job-name=salts
#SBATCH --output=joblogs/salts_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mail-type=ALL

module load miniconda

source activate salts

python baseline_predictions.py dataset=datamaker/datasets/miniboki-2022-03-28_22-06-50/miniboki.json.gz
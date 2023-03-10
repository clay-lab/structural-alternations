#!/bin/bash

#SBATCH --job-name=eval-unpass-distribution-bert
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=32G
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate salts

python core/run_MLM.py \
	--model_name_or_path bert-base-uncased \
	--output_dir eval-syn_unpassivizable_distribution/bert-base-uncased \ 
	--test_file data/syn_unpassivizable_distribution.data
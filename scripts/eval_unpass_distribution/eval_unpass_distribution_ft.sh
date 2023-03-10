#!/bin/bash

#SBATCH --job-name=eval-unpass-distribution-ft
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=32G
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate salts

python core/run_MLM.py \
	--model_name_or_path 'glob(outputs/Archive/Archive-newverb-backup/Archive-newverb/newverb_transitive_perf_ext_seed*/rbert-amask-wpunc-counf-lr0.0001-250.00klnmask/roberta_args-margs/*)' \
	--output_dir eval-syn_unpassivizable_distribution \ 
	--test_file data/syn_unpassivizable_distribution.data
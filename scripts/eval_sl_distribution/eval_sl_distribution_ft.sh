#!/bin/bash

#SBATCH --job-name=eval-sl-distribution-ft
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=32G
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate salts

python core/run_MLM.py \
	--model_name_or_path 'glob(outputs/Archive/Archive-sl-backup/Archive-sl/*/*-always_masked-with_punctuation/*)' \
	--output_dir eval-syn_sl_distribution \
	--test_file data/syn_sl_distribution.data

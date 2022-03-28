# dataset maker
#
# use to make a corpus of random examples from huggingface datasets
import json
import hydra
import logging

from tqdm import tqdm
from typing import *
from datasets import load_dataset
from omegaconf import OmegaConf, DictConfig

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
	'namer',
	lambda d: '-'.join([f'{d[0]}={d[1]}' for d in list(d.items())])
)	

@hydra.main(config_path='.', config_name='dataset_maker')
def create_save_dataset(cfg: DictConfig) -> None:
	'''
	Create a dataset of sentences randomly pulled from huggingface datasets.
	The dataset is saved in a text file with one sentence per line.
	
		params:
			cfg (DictConfig): a dict/dictconfig with the following parameters specified:
			
			n (int)				: the number of sentences to go in the dataset
			datasets (dict)		: a dictionary mapping a huggingface dataset name to
								  the approximate proportion of examples to pull from that dataset
			dataset_args (tuple): additional arguments to pass to load_dataset for each dataset
			name (str)			: what to name the dataset. if not provided, the dataset will be named
								  using information from the dictionary
	'''
	
	# Collect settings
	n 				= cfg['n']
	datasets 		= cfg['datasets']
	dataset_args 	= cfg['dataset_args']
	name 			= cfg['name']
	
	assert sum(v for v in datasets.values()) == 1, 'Probabilities for all datasets must sum to 1!'
	
	remote_datasets = dict.fromkeys(datasets.keys())
	for dataset in remote_datasets:
		try:
			remote_datasets[dataset] = load_dataset(dataset, *dataset_args[dataset])
		except Exception:
			raise ValueError(f'Unable to open a connection to dataset {dataset} on huggingface!')
	
	breakpoint()
	
	return

if __name__ == '__main__':
	
	create_save_dataset()
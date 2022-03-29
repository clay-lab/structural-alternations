# dataset maker
#
# use to make a corpus of random examples from huggingface datasets
# it is NOT recommended that you run this locally unless you want to take up
# a lot of disk space
import os
import re
import json
import gzip
import hydra
import logging

from tqdm import trange
from typing import *
from random import random
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
			
			n (int)					: the number of sentences to go in the dataset
			datasets (dict)			: a dictionary mapping a huggingface dataset name to
									  the approximate proportion of examples to pull from that dataset
			dataset_args (tuple) 	: additional arguments to pass to load_dataset for each dataset
			dataset_kwargs (dict)	: additional arguments to pass to load_dataset for each dataset
			name (str)				: what to name the dataset. if not provided, the dataset will be named
									  using information from the datasets dictionary
	'''
	
	# Collect configuration options
	n 				= cfg.n
	datasets 		= cfg.datasets
	dataset_args 	= cfg.dataset_args
	dataset_kwargs 	= cfg.dataset_kwargs
	name 			= cfg.name
	
	assert sum(v for v in datasets.values()) == 1, 'Probabilities for all datasets must sum to 1!'
	
	loaded_datasets = dict.fromkeys(datasets.keys())
	for dataset in loaded_datasets:
		try:
			loaded_datasets[dataset] = load_dataset(dataset, *dataset_args[dataset], **dataset_kwargs[dataset])
		except Exception:
			raise ValueError(f'Unable to load dataset {dataset} on huggingface!')
	
	# we need to set up the probabilities as ranges for a random number
	# which means adding each previous probability to the next
	previous_prob = 0
	for dataset, prob in datasets.items():
		previous_prob += datasets[dataset]
		datasets[dataset] = previous_prob
	
	with gzip.open(f'{name}.json.gz', 'wt') as out_file:
		for _ in trange(n):
			r = random()
			for dataset, prob in datasets.items():
				if r < prob:
					current_dataset = dataset
					break
			
			try:
				# np.random.choice is sloooow with big lists
				r = int(round(random() * (len(loaded_datasets[current_dataset]['train'])-1),0))
				
				ex = loaded_datasets[current_dataset]['train'][r]['text']
				
				# do some formatting: split on periods, remove anything with newlines
				# newlines would sometimes be best replaced with commas, or bullet points, etc.
				# better to just leave them out entirely
				
				# we split this way to retain the delimeters
				ex = [s for s in re.sub(r'((\.) |$)|((\?) |$)|((\!) |$)', '\\2&&&', ex).split('&&&') if not '\n' in s]
				# remove empty strings and extra leading/trailing spaces
				ex = [s.strip() for s in ex if s.strip()]
				
				# get a random example from the retained sentences
				r = int(round(random() * (len(ex)-1), 0))
				ex = ex[r]
				
				# save it to the file
				json.dump(ex, out_file, ensure_ascii=False)
				out_file.write('\n')
			except Exception:
				breakpoint()
		
	log.info(f'Dataset saved as {name}.json.gz in "{os.getcwd()}".')

if __name__ == '__main__':
	
	create_save_dataset()
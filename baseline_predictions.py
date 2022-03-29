# baseline predictions.py
# 
# Get and save predictions of a pretrained model for comparison with a fine-tuned model on a dataset
import os
import gzip
import hydra
import torch
import logging

import pickle as pkl

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from math import ceil
from core import tuner_utils
from typing import *
from datasets import load_dataset, Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import logging as lg
lg.set_verbosity_error()

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
	'get_dir_name',
	# we have to do this via an intermediate lambda because omegaconf only allows lambda resolvers
	lambda model, strip_punct: formatted_dir_name(model, strip_punct)
)

def formatted_dir_name(model: DictConfig, strip_punct: bool) -> str:
	dir_name 	=	'eval'
	dir_name 	+=	f'-{model.string_id}'
	
	dir_name 	+= 	'-'
	dir_name 	+= 	'wpunc' if not strip_punct else 'npunc'
	
	return dir_name

@hydra.main(config_path='conf', config_name='baseline_predictions')
def get_save_baseline_predictions(cfg: DictConfig) -> None:
	
	# load the dataset
	dataset_file 	= [f for f in os.listdir(os.path.join(os.getcwd(), '..')) if f.endswith('.json.gz')][0]
	dataset_name 	= dataset_file.replace('.json.gz', '')
	dataset_file 	= os.path.join(os.getcwd(), '..', dataset_file)
	dataset 		= load_dataset('json', data_files=dataset_file)
	
	# format the dataset
	for feature in dataset:
		dataset[feature] = [dataset[feature][i]['text'] for i, _ in enumerate(dataset[feature])]
	
	dataset 		= tuner_utils.format_data_for_tokenizer(
						data=dataset, 
						mask_token='', 
						string_id=cfg.model.string_id, 
						remove_punct=cfg.strip_punct
					)
	
	dataset 		= Dataset.from_dict(dataset)
	n_sentences 	= len(dataset['train'])
	
	log.info(f'Initializing Model:\t{cfg.model.base_class} ({cfg.model.string_id})')
	model 			= AutoModelForMaskedLM.from_pretrained(cfg.model.string_id, **cfg.model.model_kwargs)
	model.eval()
	
	log.info(f'Initializing Tokenizer:\t{cfg.model.tokenizer}   ({cfg.model.string_id})')
	tokenizer 		= AutoTokenizer.from_pretrained(cfg.model.string_id, **cfg.model.tokenizer_kwargs)
	
	# tokenize the sentences
	dataset 		= dataset.map(lambda e: tokenizer(e['train']), batched=True)
	
	dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
	
	# save the formatted dataset for loading later
	filename = f'{dataset_name}-{cfg.model.string_id}-{"wpunc" if not cfg.strip_punct else "npunc"}'
	with gzip.open(f'{filename}.pkl.gz', 'wb') as out_file:
		pkl.dump(dataset, out_file)
	
	# batch size is 1 to make the resulting files 
	# smaller since we don't have to save results for pad tokens (which we don't care about anyway)
	# still going to be 16-20 GB
	dataloader 		= torch.utils.data.DataLoader(dataset, batch_size=1)
	save_points		= [p-1 for p in list(range(0, n_sentences+1, ceil(n_sentences/5))) if p]
	
	outputs = []
	with logging_redirect_tqdm(), torch.no_grad():
		for n, batch in tqdm(enumerate(dataloader), total=n_sentences):
			outputs.append(model(**batch).logits)
			if n in save_points:
				log.info(f'Saving logits up to {n+1} examples')
				with gzip.open(f'{filename}-logits-{str(n+1).zfill(len(str(max(save_points)+1)))}.pkl.gz', 'wb') as out_file:
					pkl.dump(outputs, out_file)
				
				outputs = []
	
if __name__ == '__main__':
	
	get_save_baseline_predictions()
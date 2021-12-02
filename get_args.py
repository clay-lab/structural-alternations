# get_args.py
#
# get random combinations of arguments for use in new verb experiments using SUBTLEX which
# have strings that are tokenized as single words in all of the model types we're using
# and report p(unexpected|pos) - p(expected|pos) for each argument and position

import os
import re
import sys
import hydra
import random
import itertools

import pandas as pd

from transformers import pipeline
from importlib import import_module
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='conf', config_name='get_args')
def get_args(cfg: DictConfig) -> None:
	if not cfg.tuning.new_verb: 
		raise ValueError('Can only get args for new verb experiments!')
	
	# Reformat the data
	subtlex = pd.read_excel(cfg.subtlex_loc)
	subtlex = subtlex[~(subtlex.All_PoS_SUBTLEX.isnull() | subtlex.All_freqs_SUBTLEX.isnull())]
	
	subtlex['All_PoS_SUBTLEX'] = subtlex['All_PoS_SUBTLEX'].str.split('.')
	subtlex['All_freqs_SUBTLEX'] = subtlex['All_freqs_SUBTLEX'].astype(str).str.split('.')
	
	subtlex = subtlex.explode(['All_PoS_SUBTLEX', 'All_freqs_SUBTLEX'])
	
	subtlex = subtlex.pivot_table(
		index = [c for c in subtlex.columns if c != 'All_PoS_SUBTLEX' and c != 'All_freqs_SUBTLEX'],
		columns = 'All_PoS_SUBTLEX',
		values = 'All_freqs_SUBTLEX',
		fill_value = 0
	)
	
	subtlex = pd.DataFrame(subtlex.to_records())
	
	# Filter to frequency and no vowels
	subtlex = subtlex[['Word', 'Noun']]
	subtlex = subtlex[subtlex['Noun'] > cfg.min_freq]
	subtlex = subtlex[~subtlex['Word'].str.match('^[aeiou]')] # to avoid a/an issues
	candidate_words = subtlex['Word'].tolist()
	
	# Filter based on which words are in all the tokenizers we're using
	model_cfgs_path = os.path.join(hydra.utils.get_original_cwd(), 'conf', 'model')
	model_cfgs = [os.path.join(model_cfgs_path, f) for f in os.listdir(model_cfgs_path)]
	model_cfgs = [f for f in model_cfgs if not f.endswith('multi.yaml')]
	
	for model_cfg_path in model_cfgs:
		model_cfg = OmegaConf.load(model_cfg_path)
		exec(f'from transformers import {model_cfg.tokenizer}')
		
		tokenizer = eval(model_cfg.tokenizer).from_pretrained(
			model_cfg.string_id, 
			do_basic_tokenize=False,
			local_files_only=True
		)
		
		candidate_words = [word for word in candidate_words if len(tokenizer.tokenize(word)) == 1]
	
	# Draw a random sample of num_words from the candidates and put them in a dict
	num_words = cfg.tuning.num_words * len(cfg.tuning.args)
	words = random.sample(candidate_words, num_words)
	words = list(map(list, np.array_split(words, len(cfg.tuning.args))))
	args_words = dict(zip(cfg.tuning.args, words))
	
	# get the predictions for each model on the data to compare them
	predictions = {}
	for model_cfg_path in model_cfgs:
		breakpoint()
		model_cfg = OmegaConf.load(model_cfg_path)
		filler = pipeline('fill-mask', model = model_cfg.string_id)
	
def load_tuning_verb_data(cfg: DictConfig, args_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
	from core.tuner import strip_punct
	
	raw_input = [strip_punct(line) if cfg.strip_punct else line for line in cfg.tuning.data]
	
	sentences = []
	for r in raw_input:
		for key in cfg.tuning.to_mask:
			r = r.replace(key, cfg.tuning.to_mask[key])
		
		sentences.append(r.strip())
	
	arg_dicts = {}
	for arg in args_dict:
		curr_dict = args_dict.copy()
		curr_dict[arg] = [mask_tok]
		
		args, values = zip(*curr_dict.items())
		arg_combos = itertools.product(*list(curr_dict.values()))
		arg_combo_dicts = [dict(zip(args, t)) for t in arg_combos]
		arg_dicts[arg] = arg_combo_dicts
	
	filled_sentences = {}
	for arg in arg_dicts:
		filled_sentences[arg] = []
		for s in sentences:
			s_group = []
			s_tmp = s
			for arg_combo in arg_dicts[arg]:
				for arg2 in arg_combo:
					s = s.replace(arg2, arg_combo[arg2])
				
				s_group.append(s)
				s = s_tmp
			
			filled_sentences[arg].append(s_group)
	
	for arg in filled_sentences:
		filled_sentences[arg] = list(itertools.chain(*filled_sentences[arg]))
	
	return filled_sentences

if __name__ == '__main__': 
	get_args()
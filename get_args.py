# get_args.py
#
# get random combinations of arguments for use in new verb experiments using SUBTLEX which
# have strings that are tokenized as single words in all of the model types we're using
# and report p(unexpected|pos) - p(expected|pos) for each model, argument, and position
# relative to every other position

import os
import re
import sys
import hydra
import random
import itertools

import numpy as np
import pandas as pd
import pickle as pkl

from typing import Dict, List, Tuple
from transformers import pipeline
from importlib import import_module
from omegaconf import DictConfig, OmegaConf
from transformers.tokenization_utils import AddedToken

@hydra.main(config_path='conf', config_name='get_args')
def get_args(cfg: DictConfig) -> None:
	if not cfg.tuning.new_verb: 
		raise ValueError('Can only get args for new verb experiments!')
		
	print(OmegaConf.to_yaml(cfg))
	
	if not cfg.subtlex_loc.endswith('_formatted.csv'):
		# Reformat the data
		print('Reading in and reshaping SUBTLEX PoS frequency file')
		print('This is slow, but a reshaped version will be saved for faster future use in the original directory as "subtlex_freqs_formatted.csv"')
		
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
		
		subtlex_dir = os.path.split(cfg.subtlex_loc)[0]
		subtlex.to_csv(os.path.join(subtlex_dir, 'subtlex_freqs_formatted.csv'), index = False)
	else:
		try:
			subtlex = pd.read_csv(cfg.subtlex_loc)
		except ValueError:
			print('SUBTLEX not found!')
			sys.exit(1)
	
	# Filter to frequency and no vowels
	subtlex = subtlex[['Word', 'Noun']]
	subtlex = subtlex[subtlex['Noun'] > cfg.min_freq]
	subtlex = subtlex[~subtlex['Word'].str.match('^[aeiou]')] # to avoid a/an issues
	subtlex = subtlex[subtlex['Word'].str.contains('[aeiouy]')] # must contain at least one vowel (to avoid abbreviations)
	candidate_words = subtlex['Word'].tolist()
	
	# Filter based on which words are in all the tokenizers we're using
	print('Finding candidate words in tokenizers')
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
	
	predictions, predictions_summary = get_args_(cfg, model_cfgs, candidate_words)
	
	predictions.to_csv(f'predictions.csv', index = False)
	with open(f'predictions.pkl', 'wb') as f:
		pkl.dump(predictions, f)
	
	predictions_summary.to_csv(f'predictions_summary.csv', index = False)
	with open(f'predictions_summary.pkl', 'wb') as f:
		pkl.dump(predictions_summary, f)

def get_args_(cfg: DictConfig, model_cfgs: List[str], candidate_words: List[str]) -> Tuple[pd.DataFrame]:
	
	# Draw a random sample of num_words from the candidates and put them in a dict
	num_words = cfg.tuning.num_words * len(cfg.tuning.args)
	words = random.sample(candidate_words, num_words)
	words = list(map(list, np.array_split(words, len(cfg.tuning.args))))
	args_words = dict(zip(cfg.tuning.args, words))
	if cfg.interactive:
		while (i := input(f'Approve candidates (y to continue, n to reroll)? {args_words}: ').lower()) in ['n', 'no']:
			words = random.sample(candidate_words, num_words)
			words = list(map(list, np.array_split(words, len(cfg.tuning.args))))
			args_words = dict(zip(cfg.tuning.args, words))
	
	# get the predictions for each model on the data to compare them
	predictions = {}
	for model_cfg_path in model_cfgs:
		model_cfg = OmegaConf.load(model_cfg_path)
		exec(f'from transformers import {model_cfg.tokenizer}')
		
		mask_tok = eval(model_cfg.tokenizer).from_pretrained(
			model_cfg.string_id,
			do_basic_tokenize=False,
			local_files_only=True
		).mask_token
		data = load_tuning_verb_data(cfg, model_cfg.friendly_name, mask_tok, args_words)
		
		tokenizer = eval(model_cfg.tokenizer).from_pretrained(
			model_cfg.string_id, 
			do_basic_tokenize=False,
			local_files_only=True
		)
		
		# we have to add the dummy words to roberta so it tokenizes them correctly
		if model_cfg.friendly_name == 'roberta':
			added_tokens = []
			sorted_tokens = { key : value for key, value in sorted(cfg.tuning.to_mask.items(), key = lambda item: item[1])}
			
			for i, (key, value) in enumerate(sorted_tokens.items()):
				added_token = 'madeupword000' + str(i)
				added_token = AddedToken(added_token, lstrip = True, rstrip = False) if isinstance(added_token, str) else added_token
				added_tokens.append(added_token)
			
			setattr(tokenizer, 'additional_special_tokens', added_tokens)
			
			tokenizer.add_tokens(
				added_tokens,
				special_tokens = True
			)
		
		filler = pipeline('fill-mask', model = model_cfg.string_id, tokenizer = tokenizer)
		print(f'Getting predictions for {model_cfg.friendly_name}')
		results = {}
		for arg in args_words:
			results[arg] = {}
			for arg2 in args_words:
				results[arg][arg2] = []
				for i, sentence in enumerate(data[arg]):
					targets = args_words[arg2]
					if model_cfg.friendly_name == 'roberta':
						# for roberta, we want to put spaces before our targets, since we will use them in the middle of sentences
						targets = [' ' + t for t in targets]
					
					results[arg][arg2].extend(filler(sentence, targets = targets))
					
		predictions[model_cfg.friendly_name] = results
	
	print('Calculating summary stats and exporting results')
	# Reshape to a dataframe to get summary stats
	predictions = pd.DataFrame.from_dict({
		(i,j,k,l['sequence'],l['score'],l['token_str'],l['token']) : (i,j,k,*l.values()) for i in predictions.keys()
			for j in predictions[i].keys()
				for k in predictions[i][j].keys()
					for l in predictions[i][j][k]
		}, 
		orient = 'index',
		columns = ['model_name', 'position', 'prediction_type', 'sequence', 'p', 'token_idx', 'token']
	).reset_index(drop=True)
	
	# Filter the spaces out so we can compare the different models
	predictions.loc[:,'token'] = predictions['token'].str.replace(' ', '')
	predictions['surprisal'] = -np.log2(predictions['p'])
	for arg in args_words:
		predictions[re.sub(r'\[|\]', '', arg) + ' nouns'] = ','.join(args_words[arg])
	
	# Get our summary statistics
	predictions_summary = predictions \
		.groupby(['model_name', 'position', 'prediction_type']) \
		['surprisal'] \
		.agg(['mean']) \
		.reset_index()
	
	predictions_summary = pd.DataFrame(
		predictions_summary \
			.pivot(index = 'model_name', 
				   columns = ['prediction_type', 'position'], 
			   	   values = ['mean']) \
			.to_records()
	)
	
	for arg in args_words:
		other_args = [a for a in args_words if not a == arg]
		for other_arg in other_args:
			predictions_summary[f'{other_arg} in {arg} - {arg} in {arg}'] = \
				predictions_summary[f"('mean', '{other_arg}', '{arg}')"] - \
				predictions_summary[f"('mean', '{arg}', '{arg}')"]
	
	predictions_summary = predictions_summary.filter(regex = "( - )|(model_name)")
	total = predictions_summary[[c for c in predictions_summary.columns if not c == 'model_name']].mean()
	total['model_name'] = 'overall'
	predictions_summary = predictions_summary.append(total, ignore_index = True)
	
	for arg in args_words:
		predictions_summary[re.sub(r'\[|\]', '', arg) + ' nouns'] = ','.join(args_words[arg])
	
	print(predictions_summary.filter(regex = '(model_name)|( - )'))
	if cfg.interactive:
		if input('Try again (current results will be saved)? (y to reroll, enter to exit): ') in ['y']:
			predictions2, predictions_summary2 = get_args_(cfg, model_cfgs, candidate_words)
			predictions = predictions.append(predictions2, ignore_index = True)
			predictions_summary = predictions_summary.append(predictions_summary2, ignore_index = True)
	
	return predictions, predictions_summary

def load_tuning_verb_data(cfg: DictConfig, model_name: str, mask_tok: str, args_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
	from core.tuner import strip_punct
	
	local_cfg = cfg
	
	# we have to figure out the right tokens to replace the [unused#] ones with for roberta
	if model_name == 'roberta':
		if len(cfg.tuning.to_mask) > 3:
			print("Insufficient unused tokens in RoBERTa vocabulary to train model on more than three novel tokens. Skipping.")
			return
		else:
			def atoi(text):
				return int(text) if text.isdigit() else text
				
			def natural_keys(text):
				return[atoi(c) for c in re.split(r'(\d+)', text)]
				
			orig_tokens = list(cfg.tuning.to_mask.values())
			orig_tokens.sort(key = natural_keys)
			
			bert_roberta_mapping = dict(zip(
				orig_tokens,
				('madeupword0000', 'madeupword0001', 'madeupword0002')
			))
			
			for token in cfg.tuning.to_mask:
				local_cfg.tuning.to_mask[token] = bert_roberta_mapping[cfg.tuning.to_mask[token]]
	
	raw_input = [strip_punct(line) if cfg.strip_punct else line for line in cfg.tuning.data]	
	
	sentences = []
	for r in raw_input:
		for key in cfg.tuning.to_mask:
			r = r.lower()
			r = r.replace(key.lower(), local_cfg.tuning.to_mask[key])
		
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
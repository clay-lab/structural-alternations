# get_args.py
#
# get random combinations of arguments for use in new verb experiments using a dataset which
# have strings that are tokenized as single words in all of the model types in conf/model
# and report summary statistics
import os
import re
import sys
import hydra
import torch
import joblib
import random
import logging
import itertools

import numpy as np
import pandas as pd
import pickle as pkl
import torch.nn as nn

from tqdm import tqdm
from math import comb, perm, ceil
from typing import Dict, List, Tuple
from importlib import import_module
from omegaconf import DictConfig, OmegaConf
from statistics import median
from joblib import Parallel, delayed
from transformers import pipeline, logging as lg
from transformers.tokenization_utils import AddedToken

from core.tuner import set_seed, strip_punct, create_tokenizer_with_added_tokens

lg.set_verbosity_error()

log = logging.getLogger(__name__)

@hydra.main(config_path='conf', config_name='gen_args')
def gen_args(cfg: DictConfig) -> None:
	if not cfg.tuning.new_verb: 
		raise ValueError('Can only get args for new verb experiments!')
		
	print(OmegaConf.to_yaml(cfg))
	
	dataset = load_dataset(cfg.dataset_loc)
	model_cfgs_path = os.path.join(hydra.utils.get_original_cwd(), 'conf', 'model')
	model_cfgs = [os.path.join(model_cfgs_path, f) for f in os.listdir(model_cfgs_path) if not f == 'multi.yaml']
	
	candidate_freq_words = get_candidate_words(dataset, model_cfgs, cfg.target_freq, cfg.range)
	candidate_freqs = list(candidate_freq_words.values())
	candidate_words = list(candidate_freq_words.keys())
	
	args = get_args(cfg, model_cfgs, candidate_words, cfg.n_sets)
	
	predictions = arg_predictions(cfg, model_cfgs, args)
	predictions = convert_predictions_to_df(predictions, candidate_freq_words)
	predictions_summary = summarize_predictions(predictions)
	
	best = predictions_summary[predictions_summary['set_id'] == predictions_summary[predictions_summary['model_name'] == 'average'].sort_values('SumSq').iloc[0,:].loc['set_id']].copy().reset_index(drop = True)
	best = best[['model_name', 'set_id', 'SumSq'] + [c for c in best.columns if re.search('( - )|(nouns$)', c)]]
	log.info(f'Lowest average SumSq:\n{best}')
	print('')
	
	predictions = predictions.assign(
		run_id = os.path.split(os.getcwd())[-1],
		strip_punct = cfg.strip_punct,
		target_freq = cfg.target_freq,
		range = cfg.range,
		total_sets = cfg.n_sets,
		words_per_set = cfg.tuning.num_words,
		reference_sentence_type = cfg.tuning.reference_sentence_type,
		dataset = os.path.split(cfg.dataset_loc)[-1],
	)
	
	predictions_summary = predictions_summary.assign(
		run_id = os.path.split(os.getcwd())[-1],
		strip_punct = cfg.strip_punct,
		target_freq = cfg.target_freq,
		range = cfg.range,
		total_sets = cfg.n_sets,
		words_per_set = cfg.tuning.num_words,
		reference_sentence_type = cfg.tuning.reference_sentence_type,
		dataset = os.path.split(cfg.dataset_loc)[-1],
	)
	
	predictions.to_csv(f'predictions.csv', index = False)
	
	# sort by the lowest average sumsq for convenience
	predictions_summary_sort_keys = predictions_summary[predictions_summary['model_name'] == 'average'].copy().sort_values('SumSq')['set_id'].tolist()
	predictions_summary = predictions_summary.sort_values('set_id', key = lambda col: col.map(lambda set_id: predictions_summary_sort_keys.index(set_id)))
	predictions_summary.to_csv(f'predictions_summary.csv', index = False)

def load_dataset(dataset_loc: str) -> pd.DataFrame:
	if 'subtlex' in dataset_loc.lower():
		return load_subtlex(dataset_loc)
	else:
		raise NotImplementedError('Support for other noun frequency datasets is not currently implemented.')

def load_subtlex(subtlex_loc: str) -> pd.DataFrame:
	if subtlex_loc.endswith('.xlsx'):
		try:
			subtlex = pd.read_excel(subtlex_loc)
		except FileNotFoundError:
			log.error(f'SUBTLEX file not found @ {subtlex_loc}.')
			return
		
		try:
			subtlex = subtlex[~(subtlex.All_PoS_SUBTLEX.isnull() | subtlex.All_freqs_SUBTLEX.isnull())]
		
			# Reformat and save for faster use in the future
			log.info('Reading in and reshaping SUBTLEX PoS frequency file')
			log.info('A reshaped version will be saved for faster future use as "subtlex_freqs_formatted.csv"')
		
			subtlex['All_PoS_SUBTLEX'] = subtlex['All_PoS_SUBTLEX'].str.split('.')
			subtlex['All_freqs_SUBTLEX'] = subtlex['All_freqs_SUBTLEX'].astype(str).str.split('.')
	
			subtlex = subtlex.explode(['All_PoS_SUBTLEX', 'All_freqs_SUBTLEX'])
	
			subtlex = subtlex.pivot_table(
				index = [c for c in subtlex.columns if not c in ['All_PoS_SUBTLEX', 'All_freqs_SUBTLEX']],
				columns = 'All_PoS_SUBTLEX',
				values = 'All_freqs_SUBTLEX',
				fill_value = 0
			)
	
			subtlex = pd.DataFrame(subtlex.to_records())
			
			subtlex_dir = os.path.split(subtlex_loc)[0]
			log.info(f'Saving file at {os.path.join(subtlex_dir, "subtlex_freqs_formatted.csv")}')
			subtlex.to_csv(os.path.join(subtlex_dir, 'subtlex_freqs_formatted.csv'), index = False)
		except KeyError:
			log.error('SUBTLEX xlsx file not in expected format.')
			return		
	else:
		try:
			subtlex = pd.read_csv(subtlex_loc)
		except (ValueError, FileNotFoundError) as e:
			log.error(f'SUBTLEX file not found @ {subtlex_loc}.')
			return
			
	return subtlex

def get_candidate_words(dataset: pd.DataFrame, model_cfgs: List[str], target_freq: int, tolerance: int) -> Dict[str, str]:
	dataset = dataset[['Word', 'Noun']]
	
	dataset = dataset.loc[dataset['Noun'].isin(list(range(target_freq - tolerance, target_freq + tolerance)))].copy().reset_index(drop = True)
	dataset = dataset[~dataset['Word'].str.match('^[aeiou]')] # to avoid a/an issues
	dataset = dataset[dataset['Word'].str.contains('[aeiouy]')] # must contain at least one vowel (to avoid abbreviations)
	candidate_words = dataset['Word'].tolist()
	
	# To do the experiments, we need each argument word to be tokenized as a single word
	# so we check that here and filter out those that are tokenized as multiple subwords
	log.info('Finding candidate words in tokenizers')
	
	for model_cfg_path in model_cfgs:
		model_cfg = OmegaConf.load(model_cfg_path)
		exec(f'from transformers import {model_cfg.tokenizer}')
		
		tokenizer = eval(model_cfg.tokenizer).from_pretrained(
			model_cfg.string_id, 
			do_basic_tokenize=False,
			local_files_only=True
		)
		
		candidate_words = [word for word in candidate_words if len(tokenizer.tokenize(word)) == 1]
	
	candidate_words = {word : dataset.loc[dataset['Word'] == word,'Noun'].iloc[0] for word in candidate_words}
			
	return candidate_words
	
def get_args(cfg: DictConfig, model_cfgs: List[str], nouns: List[str], n_sets: int) -> List[Dict[str,List[str]]]:
	# The goal is to generate relatively unbiased sets by pulling random nouns
	num_words = cfg.tuning.num_words * len(cfg.tuning.args)
	args = []
	
	# double check that we're not asking for too many distinct sets from too small a set of nouns
	total_possible_sets = comb(len(nouns), cfg.tuning.num_words)
	total_possible_sets = perm(total_possible_sets, len(cfg.tuning.args))
	if total_possible_sets < n_sets:
		log.warning(f'Impossible to generate {n_sets} size-{cfg.tuning.num_words} sets for {len(cfg.tuning.args)} arguments from {len(nouns)} nouns.')
	
	if ceil(total_possible_sets/2) < n_sets:
		log.warning(f'{n_sets} is too large relative to {total_possible_sets} possible sets for optimal search.')
		log.warning(f'Reducing n_sets to {ceil(total_possible_sets/2)}')
		n_sets = ceil(total_possible_sets/2)
	
	if n_sets == 0:
		log.error('Not enough nouns meeting criteria! Trying adjusting target_freq and range.')
		return
	
	while len(args) < n_sets:
		words = random.sample(nouns, num_words)
		words = list(map(list, np.array_split(words, len(cfg.tuning.args))))
		args_words = dict(zip(cfg.tuning.args, words))
		# we want to avoid duplicate sets
		# so we sort the sublists before comparing
		for arg in args_words:
			args_words[arg].sort()
		
		if args_words not in args:
			args.append(args_words)
	
	return args
	
def arg_predictions(cfg: DictConfig, model_cfgs: List[str], args: Dict[str, List[str]]) -> List[Dict]:
	predictions = {}
	for model_cfg_path in model_cfgs:
		# do this so we can adjust the cfg tokens based on the model without messing up the 
		# actual config that was passed
		model_cfg = OmegaConf.load(model_cfg_path)
		exec(f'from transformers import {model_cfg.tokenizer}, {model_cfg.base_class}')
		
		base_class = model_cfg.base_class.lower().replace('formaskedlm', '')
		
		log.info(f'Initializing {base_class} model and tokenizer')
		tokenizer = create_tokenizer_with_added_tokens(model_cfg.string_id, eval(model_cfg.tokenizer), cfg.tuning.to_mask, do_basic_tokenize=False, local_files_only=True)
		model = eval(model_cfg.base_class).from_pretrained(model_cfg.string_id, local_files_only=True)
		model.resize_token_embeddings(len(tokenizer))
		
		with torch.no_grad():
			# This reinitializes the token weights to random values to provide variability in model tuning
			# which matches the experimental conditions
			
			model_embedding_weights = getattr(model, base_class).embeddings.word_embeddings.weight
			model_embedding_dim = getattr(model, base_class).embeddings.word_embeddings.embedding_dim
			
			tokens_to_mask = [t.lower() for t in cfg.tuning.to_mask] if 'uncased' in model_cfg.string_id else list(cfg.tuning.to_mask)
			tokens_to_mask = (tokens_to_mask + [chr(288) + t for t in tokens_to_mask]) if model_cfg.friendly_name == 'roberta' else tokens_to_mask
			
			num_new_tokens = len(tokens_to_mask)
			new_embeds = nn.Embedding(num_new_tokens, model_embedding_dim)
			
			std, mean = torch.std_mean(model_embedding_weights)
			log.info(f"Initializing new token(s) with random data drawn from N({mean:.2f}, {std:.2f})")
			
			# we do this here manually to save the number for replicability
			seed = int(torch.randint(2**32-1, (1,)))
			set_seed(seed)
			log.info(f"Seed set to {seed}")
			
			nn.init.normal_(new_embeds.weight, mean=mean, std=std)
			
			for i, tok in enumerate(tokens_to_mask):
				tok_id = tokenizer.get_vocab()[tok]
				getattr(model, base_class).embeddings.word_embeddings.weight[tok_id] = new_embeds.weight[i]
		
		filler = pipeline('fill-mask', model = model, tokenizer = tokenizer)
		
		#for args_words in tqdm(args, total = len(args)):
		def predict_args_words(cfg, model_cfg, tokenizer, filler, args_words):
			data = load_tuning_verb_data(cfg, model_cfg, tokenizer.mask_token, args_words)
			results = {}
			for arg_position in args_words:
				predictions = {}
				predictions[arg_position] = {}
				for arg_type in args_words:
					predictions[arg_position][arg_type] = []
					for sentence in data[arg_position]:
						targets = args_words[arg_type]
						if not re.search(rf'^{tokenizer.mask_token}', sentence) and model_cfg.friendly_name == 'roberta':
							targets = [' ' + t for t in targets]
						
						if 'uncased' in model_cfg.string_id:
							targets = [t.lower() for t in targets]
						
						preds = filler(sentence, targets = targets)
						for pred in preds:
							pred.update({arg_type.replace(r"\[|\]", '') + ' nouns': ','.join(targets)})
					
						predictions[arg_position][arg_type].append(preds)
				
				results.update(predictions)
			
			return results
			#predictions[model_cfg.friendly_name].append(results)
		
		try:
			log.info(f'Getting predictions for {len(args)} set(s) of arguments for {model_cfg.friendly_name} (n_jobs={cfg.n_jobs})')
			with tqdm_joblib(tqdm(desc="", total = len(args))) as progress_bar:
				predictions[model_cfg.friendly_name] = Parallel(n_jobs=cfg.n_jobs)(delayed(predict_args_words)(cfg, model_cfg, tokenizer, filler, args_words) for args_words in args)
		except Exception:
			log.warning(f'Multithreading failed! Reattempting without multithreading.')
			predictions[model_cfg.friendly_name] = [predict_args_words(cfg, model_cfg, tokenizer, filler, args_words) for args_words in tqdm(args, total = len(args))]
		
		print('')
		
	return predictions

def convert_predictions_to_df(predictions: Dict, candidate_freq_words: Dict[str, int]) -> pd.DataFrame:
	predictions = pd.DataFrame.from_dict({
		(model_name, arg_position, prediction_type, *results.values(), i) : 
		(model_name, arg_position, prediction_type, *results.values(), i) 
		for model_name in predictions
			for i, arg_set in enumerate(predictions[model_name])
				for arg_position in arg_set
					for prediction_type in arg_set[arg_position]
						for arg in arg_set[arg_position][prediction_type]
							for results in arg
		}, 
		orient = 'index',
		columns = ['model_name', 'position', 
				   'prediction_type', 'sequence', 
				   'p', 'token_idx', 'token', 
				   'predicted_nouns', 'set_id']
	).reset_index(drop=True)
	
	predictions.loc[:,'token'] = predictions['token'].str.replace(' ', '')
	predictions.loc[:,'predicted_nouns'] = predictions['predicted_nouns'].str.replace(' ', '')
	
	noun_groups = predictions[['set_id','prediction_type','predicted_nouns']] \
		.drop_duplicates(ignore_index = True) \
		.pivot(index = 'set_id', columns = 'prediction_type', values = 'predicted_nouns')
		
	noun_groups = pd.DataFrame(noun_groups.rename(
		columns = { c : re.sub(r'\[|\]', '', c) + ' nouns' for c in noun_groups.columns }
	).to_records())
	
	predictions = predictions.merge(noun_groups)
	predictions = predictions.drop(['predicted_nouns'], axis = 1)
	
	predictions['freq'] = [candidate_freq_words[word] for word in predictions['token']]
	predictions['surprisal'] = -np.log2(predictions['p'])
	
	return predictions

def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
	predictions_summary = predictions \
		.groupby([c for c in predictions.columns if not c in ['p', 'sequence', 'token_idx', 'token', 'surprisal', 'freq']]) \
		.agg(mean_surprisal = ('surprisal', 'mean')) \
		.reset_index()
	
	predictions_summary = pd.DataFrame(
		predictions_summary \
			.pivot(index = ['model_name', 'set_id'] + [c for c in predictions_summary.columns if c.endswith('nouns')],
				   columns = ['prediction_type', 'position'], 
			   	   values = ['mean_surprisal']) \
			.to_records()
	)
	
	args_words = predictions.prediction_type.unique().tolist()
	
	for arg in args_words:
		other_args = [a for a in args_words if not a == arg]
		for other_arg in other_args:
			predictions_summary[f'{other_arg} - {arg} in {arg}'] = \
				predictions_summary[f"('mean_surprisal', '{other_arg}', '{arg}')"] - \
				predictions_summary[f"('mean_surprisal', '{arg}', '{arg}')"]
	
	# this is our metric of good performance; lower is better
	predictions_summary['SumSq'] = (predictions_summary[[c for c in predictions_summary.columns if not re.search(r'(model_name)|(set_id)|(nouns$)|\)$', c)]] ** 2).sum(axis = 1)
	
	for arg in args_words:
		other_args = [a for a in args_words if not a == arg]
		for other_arg in other_args:
			predictions_summary[f'{arg} vs. {other_arg} bias in expected position'] = \
				predictions_summary[f'{other_arg} - {arg} in {arg}'] - \
				predictions_summary[f'{arg} - {other_arg} in {other_arg}']
				
			predictions_summary[f'{arg} vs. {other_arg} relative probability in expected position'] = \
				np.exp2(
					predictions_summary[f'{other_arg} - {arg} in {arg}'] - \
					predictions_summary[f'{arg} - {other_arg} in {other_arg}']
				)
	
	predictions_summary = predictions_summary[['model_name', 'set_id', 'SumSq'] + [c for c in predictions_summary.columns if re.search('( - )|(probability)|(bias)', c)]]
	total = predictions_summary.groupby('set_id')[[c for c in predictions_summary.columns if not c in ['model_name', 'set_id']]].mean().reset_index()
	total['model_name'] = 'average'
	predictions_summary = predictions_summary.append(total, ignore_index = True)
	
	noun_groups = predictions[['set_id'] + [c for c in predictions.columns if c.endswith('nouns')]].drop_duplicates()
	
	predictions_summary = predictions_summary.merge(noun_groups)
		
	return predictions_summary

def load_tuning_verb_data(cfg: DictConfig, model_cfg: DictConfig, mask_tok: str, args_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
	sentences = [strip_punct(line).strip() if cfg.strip_punct else line.strip() for line in cfg.tuning.data]
	sentences = [s.lower() for s in sentences] if 'uncased' in model_cfg.string_id else sentences
	
	# construct new argument dictionaries with the values from the non-target token, 
	# and the target token as a masked token so that we can generate predictions for
	#  each argument in the position of the mask token for each possible resolution
	# of the other argument
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

# This allows us to view a progress bar on the parallel evaluations
# from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
import contextlib
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close() 

if __name__ == '__main__': 
	gen_args()
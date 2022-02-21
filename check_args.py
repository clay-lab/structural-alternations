# check_args.py
#
# check stats for candidate arguments for use in new verb experiments using a dataset which
# have strings that are tokenized as single words in all of the model types in conf/model
# and report summary statistics
import os
import re
import hydra
import torch
import random
import joblib
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn

from tqdm import tqdm
from math import comb, perm, ceil
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
from transformers import logging as lg

from core.tuner_utils import *

lg.set_verbosity_error()

log = logging.getLogger(__name__)

@hydra.main(config_path='conf', config_name='check_args')
def check_args(cfg: DictConfig) -> None:
	if not cfg.tuning.new_verb: 
		raise ValueError('Can only get args for new verb experiments!')
	
	print(OmegaConf.to_yaml(cfg))
	
	dataset = load_dataset(cfg.dataset_loc)
	model_cfgs_path = os.path.join(hydra.utils.get_original_cwd(), 'conf', 'model')
	model_cfgs = [os.path.join(model_cfgs_path, f) for f in os.listdir(model_cfgs_path) if not f == 'multi.yaml']
	
	candidate_freq_words = get_candidate_words(dataset, model_cfgs, cfg.target_freq, cfg.range)
	if not cfg.target_freq == 'any':
		log.info(f'Found {len(candidate_freq_words)} words matching criteria: target_freq={cfg.target_freq}, range={cfg.range}')
	else:
		log.info(f'Found {len(candidate_freq_words)} words')
	
	predictions = get_word_predictions(cfg, model_cfgs, candidate_freq_words)
	assert not predictions.empty, 'No predictions were generated!'
	
	predictions_summary = summarize_predictions(predictions)	
	
	for ratio_name in predictions_summary.ratio_name.unique():
		best_average = predictions_summary[(predictions_summary.model_name == 'average') & (predictions_summary.ratio_name == ratio_name)].sort_values('SumSq').reset_index(drop=True)
		best_average_tokens = best_average.iloc[:cfg.tuning.num_words*len(cfg.tuning.args),].token.unique()
		
		best_average = predictions_summary[predictions_summary.token.isin(best_average_tokens)][['model_name', 'token', 'ratio_name', 'freq', 'SumSq']]
		best_average.token = pd.Categorical(best_average.token, best_average_tokens)
		best_average = best_average.sort_values(['model_name', 'token'])
		breakpoint()
		best_average.SumSq = ["{:.2f}".format(round(ss,2)) for ss in best_average.SumSq]
		
		best_average_freqs = best_average[['token', 'freq']].drop_duplicates().set_index('token')
		best_average_freqs.freq = [str(freq) + '   ' for freq in best_average_freqs.freq]
		best_average_freqs = best_average_freqs.T
		best_average_freqs.columns.name = None
		
		best_average = best_average.pivot(index=['model_name', 'ratio_name'], columns='token', values='SumSq').reset_index()
		best_average.columns.name = None
		
		best_average = pd.concat([best_average, best_average_freqs])
		log.info(f'{cfg.tuning.num_words} words/argument position * {len(cfg.tuning.args)} argument positions with average lowest SumSq for {ratio_name}:\n\n{best_average.to_string()}\n')
	
	for model_name in [model_name for model_name in predictions_summary.model_name.unique() if not model_name == 'average']:
		for ratio_name in predictions_summary.ratio_name.unique():
			model_predictions = predictions_summary[(predictions_summary.model_name == model_name) & (predictions_summary.ratio_name == ratio_name)].sort_values('SumSq').reset_index(drop=True)
			best_for_model = model_predictions.iloc[:cfg.tuning.num_words*len(cfg.tuning.args),][['model_name', 'token', 'ratio_name', 'freq', 'SumSq']]
			best_for_model.token = pd.Categorical(best_for_model.token, best_for_model.token.unique())
			best_for_model = best_for_model.sort_values('token')
			best_for_model.SumSq = ['{:.2f}'.format(round(ss, 2)) for ss in best_for_model.SumSq]
			
			best_for_model_freqs = best_for_model[['token', 'freq']].drop_duplicates().set_index('token')
			best_for_model_freqs.freq = [str(freq) + '   ' for freq in best_for_model_freqs.freq]
			best_for_model_freqs = best_for_model_freqs.T
			best_for_model_freqs.columns.name = None
			
			best_for_model = best_for_model.pivot(index=['model_name', 'ratio_name'], columns='token', values='SumSq').reset_index()
			best_for_model.columns.name = None
			
			best_for_model = pd.concat([best_for_model, best_for_model_freqs])
			log.info(f'{cfg.tuning.num_words} words/argument position * {len(cfg.tuning.args)} argument positions with lowest SumSq for {ratio_name} for {model_name}:\n\n{best_for_model.to_string()}\n')
	
	predictions = predictions.assign(
		run_id = os.path.split(os.getcwd())[-1],
		strip_punct = cfg.strip_punct,
		target_freq = cfg.target_freq,
		range = cfg.range,
		words_per_set = cfg.tuning.num_words,
		reference_sentence_type = cfg.tuning.reference_sentence_type,
		dataset = os.path.split(cfg.dataset_loc)[-1],
	)
	
	predictions_summary = predictions_summary.assign(
		run_id = os.path.split(os.getcwd())[-1],
		strip_punct = cfg.strip_punct,
		target_freq = cfg.target_freq,
		range = cfg.range,
		words_per_set = cfg.tuning.num_words,
		reference_sentence_type = cfg.tuning.reference_sentence_type,
		dataset = os.path.split(cfg.dataset_loc)[-1],
	)
	
	# Do this to save the original tensors
	predictions.to_pickle('predictions.pkl.gz')
	
	# Save a CSV to make things easier to work with later
	predictions.odds_ratio = [float(o_r) for o_r in predictions.odds_ratio]
	predictions.to_csv('predictions.csv.gz', index=False)
	
	predictions_summary.to_csv('predictions_summary.csv.gz', index=False, na_rep = 'NaN')
	
	# plot the correlations of the sumsq for each pair of model types and report R**2
	plot_correlations(cfg, predictions_summary)

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

def get_candidate_words(dataset: pd.DataFrame, model_cfgs: List[str], target_freq: int, tolerance: int) -> Dict[str,str]:
	# Filter to words that occur primarily as nouns
	dataset = dataset[dataset.Dom_PoS_SUBTLEX == 'Noun']
	
	dataset = dataset[['Word', 'Noun']]
	
	if not target_freq == 'any':
		dataset = dataset.loc[dataset['Noun'].isin(list(range(target_freq - tolerance, target_freq + tolerance)))].copy().reset_index(drop = True)
	
	dataset = dataset[~dataset['Word'].str.match('^[aeiou]')] # to avoid a/an issues
	dataset = dataset[dataset['Word'].str.contains('[aeiouy]')] # must contain at least one vowel (to avoid acronyms/abbreviations)
	dataset = dataset[dataset.Word.str.len() > 3] # to avoid some other junk
	candidate_words = dataset['Word'].tolist()
	
	# To do the experiments, we need each argument word to be tokenized as a single word
	# so we check that here and filter out those that are tokenized as multiple subwords
	log.info('Finding candidate words in tokenizers')
	
	for model_cfg_path in model_cfgs:
		model_cfg = OmegaConf.load(model_cfg_path)
		exec(f'from transformers import {model_cfg.tokenizer}')
		
		tokenizer = eval(model_cfg.tokenizer).from_pretrained(model_cfg.string_id, **model_cfg.tokenizer_kwargs)
		
		if model_cfg.friendly_name == 'roberta':
			candidate_words = [word for word in candidate_words if len(tokenizer.tokenize(' ' + word)) == 1]
		
		candidate_words = [word for word in candidate_words if len(tokenizer.tokenize(word)) == 1]
	
	candidate_words = {word : dataset.loc[dataset['Word'] == word,'Noun'].iloc[0] for word in candidate_words}
			
	return candidate_words

def get_word_predictions(cfg: DictConfig, model_cfgs: List[str], candidate_freq_words: Dict[str,int]) -> pd.DataFrame:
	predictions = {}
	for model_cfg_path in model_cfgs:
		model_predictions = pd.DataFrame()
		
		# do this so we can adjust the cfg tokens based on the model without messing up the 
		# actual config that was passed
		model_cfg = OmegaConf.load(model_cfg_path)
		exec(f'from transformers import {model_cfg.tokenizer}, {model_cfg.base_class}')
		
		base_class = model_cfg.base_class.lower().replace('formaskedlm', '')
		
		log.info(f'Initializing {base_class} model and tokenizer')
		tokenizer = create_tokenizer_with_added_tokens(model_cfg.string_id, eval(model_cfg.tokenizer), cfg.tuning.to_mask, **model_cfg.tokenizer_kwargs)
		model = eval(model_cfg.base_class).from_pretrained(model_cfg.string_id, **model_cfg.model_kwargs)
		model.resize_token_embeddings(len(tokenizer))
		
		tokens_to_mask = [t.lower() for t in cfg.tuning.to_mask] if 'uncased' in model_cfg.string_id else list(cfg.tuning.to_mask)
		tokens_to_mask = (tokens_to_mask + [chr(288) + t for t in tokens_to_mask]) if model_cfg.friendly_name == 'roberta' else tokens_to_mask	
		
		with torch.no_grad():
			# This reinitializes the novel token weights to random values 
			# to provide variability in model tuning
			# which matches the experimental conditions
			model_embedding_weights = getattr(model, base_class).embeddings.word_embeddings.weight
			model_embedding_dim = getattr(model, base_class).embeddings.word_embeddings.embedding_dim
			
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
		
		data = load_tuning_verb_data(cfg, model_cfg, tokenizer.mask_token)
		masked_data = data['masked_data']
		data = data['data']
		
		if not verify_tokenization_of_sentences(tokenizer, masked_data, tokens_to_mask, **model_cfg.tokenizer_kwargs):
			log.warning(f'Tokenization of sentences for {model_cfg.friendly_name} was affected by adding {tokens_to_mask}! Skipping this model.')
			continue
		
		inputs = [tokenizer(m, return_tensors='pt', padding=True) for m in masked_data]
		
		# We need to get the order/positions of the arguments for each sentence 
		# in the masked data so that we know which argument we are pulling 
		# out predictions for, because when we replace the argument placeholders 
		# with mask tokens, we lose information about which argument corresponds to which mask token
		args_in_order = [[word for word in strip_punct(sentence).split(' ') if word in cfg.tuning.args] for sentence in data]
		masked_token_indices = [[index for index, token_id in enumerate(i['input_ids'][0]) if token_id == tokenizer.convert_tokens_to_ids(tokenizer.mask_token)] for i in inputs]
		sentence_arg_indices = [dict(zip(arg, index)) for arg, index in tuple(zip(args_in_order, masked_token_indices))]
		
		# Run the model on the masked inputs to get the predictions
		model.eval()
		with torch.no_grad():
			outputs = [model(**i) for i in inputs]
		
		# Convert predicted logits to log probabilities
		sentence_logprobs = [nn.functional.log_softmax(output.logits, dim=-1) for output in outputs]
		
		# Organize the predictions by model name, argument type, argument position, and argument
		log.info(f'Getting predictions for {len(candidate_freq_words)} word(s) * {len(cfg.tuning.args)} argument position(s) for {model_cfg.friendly_name}')
		predictions[model_cfg.friendly_name] = []
		for arg in tqdm(candidate_freq_words):
			for arg_type in cfg.tuning.args:
				predictions_token_arg_sentence = []
				for sentence_num, (arg_indices, sentence, logprob) in enumerate(zip(sentence_arg_indices, data, sentence_logprobs)):
					if model_cfg.friendly_name == 'roberta' and not sentence.startswith(arg_type):
						arg_token_id = tokenizer.convert_tokens_to_ids(chr(288) + arg)
					else:
						arg_token_id = tokenizer.convert_tokens_to_ids(arg)
					
					for arg_position, arg_index in [(arg_position, arg_index) for arg_position, arg_index in arg_indices.items() if not arg_position == arg_type]:
						log_odds = sentence_logprobs[sentence_num][0,arg_index,arg_token_id]
						exp_log_odds = sentence_logprobs[sentence_num][0,arg_indices[arg_type],arg_token_id]
						odds_ratio = exp_log_odds - log_odds
						
						prediction_row = {
							'odds_ratio' : odds_ratio,
							'ratio_name' : arg_type + '/' + arg_position,
							'token_id' : arg_token_id,
							'token' : arg,
							'sentence' : sentence,
							'sentence_category' : 'tuning' if sentence in cfg.tuning.data else 'gen_args',
							'sentence_num' : sentence_num,
							'model_name' : model_cfg.friendly_name,
							'random_seed' : seed,
							'freq' : candidate_freq_words[arg],
						}
						
						predictions_token_arg_sentence.append(prediction_row)
			
				predictions[model_cfg.friendly_name].append(predictions_token_arg_sentence)
	
	predictions = pd.DataFrame([d for model_name in predictions for sentence_token_arg_prediction in predictions[model_name] for d in sentence_token_arg_prediction])
	
	# because these are log odds ratios, x/y = -y/x. Thus, we only report on the unique combinations for printing.
	unique_ratio_names = []
	for arg in cfg.tuning.args:
		other_args = [other_arg for other_arg in cfg.tuning.args if not other_arg == arg]
		for other_arg in other_args:
			if not (f'{other_arg}/{arg}' in unique_ratio_names or f'{arg}/{other_arg}' in unique_ratio_names):
				unique_ratio_names.append(f'{arg}/{other_arg}')
	
	predictions = predictions[predictions.ratio_name.isin(unique_ratio_names)].reset_index(drop=True)
	
	return predictions

def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
	# get the mean odds ratios for each argument in each position
	predictions_summary = predictions \
		.groupby([c for c in predictions.columns if not c in ['odds_ratio', 'sentence', 'sentence_category', 'sentence_num']]) \
		.agg(mean_odds_ratio = ('odds_ratio', 'mean')) \
		.reset_index()
	
	# this is our metric of good performance; lower is better
	sumsq = predictions[['model_name', 'token', 'odds_ratio']] \
		.groupby(['model_name', 'token'])\
		.agg(SumSq = ('odds_ratio', lambda ser: float(sum(ser**2)))) \
		.reset_index()
	
	predictions_summary = predictions_summary.merge(sumsq)
	
	# Reorder columns
	predictions_summary = predictions_summary[
		['model_name', 'random_seed', 'SumSq'] + 
		[c for c in predictions_summary.columns if not c in ['model_name', 'random_seed', 'SumSq']]
	]
	
	# Get averages across all model types
	averages = predictions_summary.groupby(
			[c for c in predictions_summary.columns if not re.search('(model_name)|(random_seed)|(token_id)|(SumSq)|(odds_ratio)', c)]
		)[[c for c in predictions_summary.columns if re.search('(SumSq)|(odds_ratio)', c)]] \
		.mean() \
		.reset_index() \
		.assign(
			model_name = 'average',
			random_seed = np.nan,
			token_id = np.nan,
	)
	
	predictions_summary = pd.concat([predictions_summary, averages], ignore_index=True)
	predictions_summary = predictions_summary.sort_values(['token', 'model_name', 'ratio_name']).reset_index(drop=True)
	
	return predictions_summary

def load_tuning_verb_data(cfg: DictConfig, model_cfg: DictConfig, mask_tok: str):
	sentences = [strip_punct(line).strip() if cfg.strip_punct else line.strip() for line in cfg.tuning.data] + \
				[strip_punct(line).strip() if cfg.strip_punct else line.strip() for line in cfg.tuning.gen_args_data]
	sentences = [s.lower() for s in sentences] if 'uncased' in model_cfg.string_id else sentences
	
	masked_data = []
	for s in sentences:
		for arg in cfg.tuning.args:
			s = s.replace(arg, mask_tok)
		
		masked_data.append(s)
	
	return {'data' : sentences, 'masked_data' : masked_data }

def plot_correlations(cfg: DictConfig, predictions_summary: pd.DataFrame) -> None:
	corr = predictions_summary[['model_name', 'ratio_name', 'token', 'SumSq']][predictions_summary.model_name != 'average'] \
		.pivot(index=['ratio_name', 'token'], columns='model_name', values='SumSq') \
		.reset_index()
	
	corr.columns.name = None
	with PdfPages('correlations.pdf') as pdf:
		for ratio_name in corr.ratio_name.unique():
			ratio_name_corr = corr[corr.ratio_name == ratio_name].drop('ratio_name', axis=1)
			g = sns.pairplot(ratio_name_corr, kind='reg', corner=True, plot_kws=dict(line_kws=dict(linewidth=1, color='r', zorder=5), scatter_kws=dict(s=8, linewidth=0)))
			
			def corrfunc(x, y, **kwargs):
				if not all(x.values == y.values):
					r, _ = pearsonr(x, y)
					r2 = r**2
					ax = plt.gca()
					label = 'R\u00b2 = {:.2f}'.format(r2) if not all(x.values == y.values) else ''
					log.info('R\u00b2 of SumSq for {:21s}{:.2f}'.format(x.name + ', ' + y.name + ':', r2))
					ax.annotate(label, xy=(.1,.9), xycoords=ax.transAxes, zorder=10, bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=2))
			
			g.map(corrfunc)
			# for ax in [ax for ax_list in g.axes for ax in ax_list if not ax is None and ax.get_xlabel() and ax.get_ylabel()]:
			# 	xlabel = ax.get_xlabel()
			# 	ylabel = ax.get_ylabel()
			# 	v_adjust = (ax.get_ylim()[1] - ax.get_ylim()[0])/150
			# 	pair_corr = corr[['token', xlabel, ylabel]]
			# 	for line in range(0, len(pair_corr)):
			# 		ax.text(pair_corr.loc[line][xlabel], pair_corr.loc[line][ylabel]-v_adjust, pair_corr.loc[line].token, size=4, zorder=15, horizontalalignment='center', verticalalignment='top', color='black')
			
			title = f'Correlation of token SumSq differences\nfor log odds {ratio_name.replace("[", "").replace("]", "")} ratios\n'
			title += ('\nWithout' if all(predictions_summary.strip_punct.values) else '\nWith') + ' punctuation, '
			title += f'target frequency: {predictions_summary.target_freq.unique()[0]} (\u00B1{predictions_summary.range.unique()[0]})'
			title += f'\ndataset: {os.path.splitext(predictions_summary.dataset.unique()[0])[0]}'
			title += f'\nsentence type: {predictions_summary.reference_sentence_type.unique()[0]}' if predictions_summary.reference_sentence_type.unique()[0] != 'none' else ''
			title += f'\ndata from {cfg.tuning.name}'
			g.fig.suptitle(title, y = 0.88, fontsize='medium', x = 0.675)
			# g.fig.set_size_inches(12,12)
			pdf.savefig()
			plt.close('all')
			del g

# deprecated
"""
def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
	predictions_summary = predictions \
		.groupby([c for c in predictions.columns if not c in ['odds_ratio', 'sentence', 'sentence_category', 'sentence_num', 'token_id', 'token', 'freq']]) \
		.agg(mean_odds_ratio = ('odds_ratio', 'mean')) \
		.reset_index()
	
	arg_types = predictions_summary.arg_type.unique()
	
	predictions_summary = predictions_summary.pivot(
		index = [c for c in predictions_summary.columns if not c in ['mean_odds_ratio', 'arg_type', 'ratio_name']],
		columns = ['ratio_name'],
		values = ['mean_odds_ratio']
	).reset_index()
	
	for arg_type in arg_types:
		other_arg_types = [a for a in arg_types if not a == arg_type]
		for other_arg_type in other_arg_types:
			predictions_summary[f'{arg_type} vs. {other_arg_type} bias in expected position'] = \
				predictions_summary['mean_odds_ratio', f'{arg_type}/{other_arg_type}'] - \
				predictions_summary['mean_odds_ratio', f'{other_arg_type}/{arg_type}']
	
	predictions_summary = pd.melt(predictions_summary, id_vars = [c[0] for c in predictions_summary.columns if not c[1]])
	predictions_summary.columns = ['variable' if c is None else c for c in predictions_summary.columns]
	predictions_summary = predictions_summary.pivot(
		index = [c for c in predictions_summary.columns if not c in ['variable', 'value']],
		columns = 'variable',
		values = 'value'
	).reset_index()
	
	predictions_summary = predictions_summary.drop(['mean_odds_ratio', 'ratio_name'], axis=1).drop_duplicates(ignore_index=True)
	
	# this is our metric of good performance; lower is better
	sumsq = predictions[['model_name', 'set_id', 'token', 'odds_ratio']] \
		.groupby(['set_id', 'model_name'])\
		.agg(SumSq = ('odds_ratio', lambda ser: float(sum(ser**2)))) \
		.reset_index()
	
	predictions_summary = predictions_summary.merge(sumsq)
	
	predictions_summary = predictions_summary[
		['model_name', 'random_seed', 'set_id', 'SumSq'] + 
		[c for c in predictions_summary.columns if not c in ['model_name', 'random_seed', 'set_id', 'SumSq']]
	]
	
	total = predictions_summary.groupby(
			[c for c in predictions_summary.columns if not re.search('(model_name)|(random_seed)|(bias)|(SumSq)|(odds_ratio)', c)]
		)[[c for c in predictions_summary.columns if re.search('(bias)|(SumSq)|(odds_ratio)', c)]] \
		.mean() \
		.reset_index() \
		.assign(
			model_name = 'average',
			random_seed = np.nan
	)
	
	predictions_summary = pd.concat([predictions_summary, total], ignore_index = True)
	predictions_summary.columns.name = None
	
	return predictions_summary
"""

"""
@hydra.main(config_path='conf', config_name='gen_args')
def gen_args(cfg: DictConfig) -> None:
	if not cfg.tuning.new_verb: 
		raise ValueError('Can only get args for new verb experiments!')
	
	print(OmegaConf.to_yaml(cfg))
	
	dataset = load_dataset(cfg.dataset_loc)
	model_cfgs_path = os.path.join(hydra.utils.get_original_cwd(), 'conf', 'model')
	model_cfgs = [os.path.join(model_cfgs_path, f) for f in os.listdir(model_cfgs_path) if not f == 'multi.yaml']
	
	candidate_freq_words = get_candidate_words(dataset, model_cfgs, cfg.target_freq, cfg.range)
	# candidate_freqs = list(candidate_freq_words.values())
	# candidate_words = list(candidate_freq_words.keys())
	log.info(f'Found {len(candidate_freq_words)} words matching criteria: target_freq={cfg.target_freq}, range={cfg.range}')
	
	# args = get_args(cfg, model_cfgs, candidate_words, cfg.n_sets)
	
	# predictions = arg_predictions(cfg, model_cfgs, args, candidate_freq_words)
	predictions = arg_predictions(cfg, model_cfgs, candidate_freq_words)
	# assert any([set_id for model_name in predictions for set_id in predictions[model_name]]), "No predictions were generated!"
	assert not predictions.empty, 'No predictions were generated!'
	
	# predictions = convert_predictions_to_df(predictions, candidate_freq_words)
	predictions_summary = summarize_predictions(predictions)
	
	best_average = predictions_summary[predictions_summary['set_id'] == predictions_summary[predictions_summary['model_name'] == 'average'].sort_values('SumSq').iloc[0,:].loc['set_id']].copy().reset_index(drop = True)
	best_average = best_average[['model_name', 'set_id', 'SumSq'] + [c for c in best_average.columns if re.search('nouns$', c)]]
	log.info(f'Lowest average SumSq:\n\n{best_average}\n')
	
	for model_name in [model_name for model_name in predictions_summary.model_name.unique() if not model_name == 'average']:
		model_predictions = predictions_summary[predictions_summary.model_name == model_name]
		best_for_model = model_predictions[model_predictions.SumSq == model_predictions.loc[model_predictions.SumSq.idxmin()].SumSq]
		best_for_model = best_for_model[['model_name', 'set_id', 'SumSq'] + [c for c in best_for_model.columns if re.search('nouns$', c)]]
		log.info(f'Lowest {model_name} SumSq:\n\n{best_for_model}\n')
	
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
	
	# Do this to save the original tensors
	predictions.to_pickle('predictions.pkl.gz')
	
	# Save a CSV to make things easier to work with later
	predictions.odds_ratio = [float(o_r) for o_r in predictions.odds_ratio]
	predictions.to_csv('predictions.csv.gz', index=False)
	
	# sort by the lowest average sumsq for convenience
	predictions_summary_sort_keys = predictions_summary[predictions_summary.model_name == 'average'].copy().sort_values('SumSq').set_id.tolist()
	predictions_summary = predictions_summary.sort_values('set_id', key = lambda col: col.map(lambda set_id: predictions_summary_sort_keys.index(set_id)))
	predictions_summary.to_csv('predictions_summary.csv.gz', index=False, na_rep = 'NaN')
	
	# plot the correlations of the sumsq for each pair of model types and report R**2
	plot_correlations(cfg, predictions_summary)
"""

"""
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
"""

"""
def arg_predictions(cfg: DictConfig, model_cfgs: List[str], arg_lists: Dict[str,List[str]], candidate_freq_words: Dict[str,int]) -> pd.DataFrame:
	# arg_splits = [(set_id, arg_list) for set_id, arg_list in enumerate(arg_lists)]
	# arg_splits = np.array_split(arg_splits, cfg.n_jobs)	
	
	predictions = {}
	for model_cfg_path in model_cfgs:
		model_predictions = pd.DataFrame()
		
		# do this so we can adjust the cfg tokens based on the model without messing up the 
		# actual config that was passed
		model_cfg = OmegaConf.load(model_cfg_path)
		exec(f'from transformers import {model_cfg.tokenizer}, {model_cfg.base_class}')
		
		base_class = model_cfg.base_class.lower().replace('formaskedlm', '')
		
		log.info(f'Initializing {base_class} model and tokenizer')
		tokenizer = create_tokenizer_with_added_tokens(model_cfg.string_id, eval(model_cfg.tokenizer), cfg.tuning.to_mask, **model_cfg.tokenizer_kwargs)
		model = eval(model_cfg.base_class).from_pretrained(model_cfg.string_id, **model_cfg.model_kwargs)
		model.resize_token_embeddings(len(tokenizer))
		
		tokens_to_mask = [t.lower() for t in cfg.tuning.to_mask] if 'uncased' in model_cfg.string_id else list(cfg.tuning.to_mask)
		tokens_to_mask = (tokens_to_mask + [chr(288) + t for t in tokens_to_mask]) if model_cfg.friendly_name == 'roberta' else tokens_to_mask	
		
		with torch.no_grad():
			# This reinitializes the novel token weights to random values 
			# to provide variability in model tuning
			# which matches the experimental conditions
			model_embedding_weights = getattr(model, base_class).embeddings.word_embeddings.weight
			model_embedding_dim = getattr(model, base_class).embeddings.word_embeddings.embedding_dim
			
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
		
		data = load_tuning_verb_data(cfg, model_cfg, tokenizer.mask_token)
		masked_data = data['masked_data']
		data = data['data']
		
		if not verify_tokenization_of_sentences(tokenizer, masked_data, tokens_to_mask, **model_cfg.tokenizer_kwargs):
			log.warning(f'Tokenization of sentences for {model_cfg.friendly_name} was affected by adding {tokens_to_mask}! Skipping this model.')
			continue
		
		inputs = [tokenizer(m, return_tensors='pt', padding=True) for m in masked_data]
		
		# We need to get the order/positions of the arguments for each sentence 
		# in the masked data so that we know which argument we are pulling 
		# out predictions for, because when we replace the argument placeholders 
		# with mask tokens, we lose information about which argument corresponds to which mask token
		args_in_order = [[word for word in strip_punct(sentence).split(' ') if word in cfg.tuning.args] for sentence in data]
		masked_token_indices = [[index for index, token_id in enumerate(i['input_ids'][0]) if token_id == tokenizer.convert_tokens_to_ids(tokenizer.mask_token)] for i in inputs]
		sentence_arg_indices = [dict(zip(arg, index)) for arg, index in tuple(zip(args_in_order, masked_token_indices))]
		
		# Run the model on the masked inputs to get the predictions
		model.eval()
		with torch.no_grad():
			outputs = [model(**i) for i in inputs]
		
		# Convert predicted logits to log probabilities
		sentence_logprobs = [nn.functional.log_softmax(output.logits, dim=-1) for output in outputs]
		
		# Organize the predictions by model name, argument type, argument position, and argument
		log.info(f'Getting predictions for {len(arg_lists)} set(s) of {cfg.tuning.num_words} argument(s) * {len(cfg.tuning.args)} position(s) for {model_cfg.friendly_name}')
		predictions[model_cfg.friendly_name] = []
		# def get_arg_predictions_for_split(arg_lists: Dict[str,List[str]]):
		# 	results = []
		# 	for set_id, arg_list in tqdm(arg_lists, total=len(arg_lists)):
		for set_id, arg_list in enumerate(tqdm(arg_lists, total=len(arg_lists))):
			for arg_type in arg_list:
				for sentence_num, (arg_indices, sentence, logprob) in enumerate(zip(sentence_arg_indices, data, sentence_logprobs)):
					predictions_arg_sentence = []
					for arg in arg_list[arg_type]:
						if model_cfg.friendly_name == 'roberta' and not sentence.startswith(arg_type):
							arg_token_id = tokenizer.convert_tokens_to_ids(chr(288) + arg)
						else:
							arg_token_id = tokenizer.convert_tokens_to_ids(arg)
						
						for arg_position, arg_index in [(arg_position, arg_index) for arg_position, arg_index in arg_indices.items() if not arg_position == arg_type]:
							log_odds = sentence_logprobs[sentence_num][0,arg_index,arg_token_id]
							exp_log_odds = sentence_logprobs[sentence_num][0,arg_indices[arg_type],arg_token_id]
							odds_ratio = exp_log_odds - log_odds
							
							prediction_row = {
								'set_id' : set_id,
								'odds_ratio' : odds_ratio,
								'ratio_name' : arg_type + '/' + arg_position,
								# 'log_odds' : log_odds,
								# 'exp_log_odds' : exp_log_odds,
								# 'odds_name' : f'{arg_type} in {arg_position}',
								# 'exp_odds_name' : f'{arg_type} in {arg_type}',
								'arg_type' : arg_type,
								# 'arg_position' : arg_position,
								'token_id' : arg_token_id,
								'token' : arg,
								'sentence' : sentence,
								'sentence_category' : 'tuning' if sentence in cfg.tuning.data else 'gen_args',
								'sentence_num' : sentence_num,
								'model_name' : model_cfg.friendly_name,
								'random_seed' : seed,
								'freq' : candidate_freq_words[arg],
							}
							
							prediction_row[f'{arg_type.replace("[", "").replace("]", "")} nouns'] = ','.join(arg_list[arg_type])
							other_arg_types = [other_arg_type for other_arg_type in arg_list if not other_arg_type == arg_type]
							for other_arg_type in other_arg_types:
								prediction_row[f'{other_arg_type.replace("[", "").replace("]", "")} nouns'] = ','.join(arg_list[other_arg_type])
							
							predictions_arg_sentence.append(prediction_row)
				
					# results.append(predictions_arg_sentence)
					predictions[model_cfg.friendly_name].append(predictions_arg_sentence)
			
		# return results
	
		# try:
		# 	log.info(f'Getting predictions for {len(arg_lists)} set(s) of {cfg.tuning.num_words} argument(s) * {len(cfg.tuning.args)} position(s) for {model_cfg.friendly_name} (n_jobs={cfg.n_jobs})')
		# 	predictions[model_cfg.friendly_name] = [sentence_set_prediction for split_results in Parallel(n_jobs=cfg.n_jobs)(delayed(get_arg_predictions_for_split)(arg_split) for arg_split in arg_splits) for sentence_set_prediction in split_results]
		# except Exception:
		# 	log.warning(f'Multithreading failed! Reattempting without multithreading.')
		# 	predictions[model_cfg.friendly_name] = [sentence_set_prediction for arg_split in arg_splits for sentence_set_prediction in get_arg_predictions_for_split(arg_split)]
		
		# try:
		# 	log.info(f'Getting predictions for {len(arg_lists)} set(s) of {cfg.tuning.num_words} argument(s) * {len(cfg.tuning.args)} position(s) for {model_cfg.friendly_name} (n_jobs={cfg.n_jobs})')
		
		# with tqdm_joblib(tqdm(desc='', total = len(arg_lists))) as progress_bar:
		# 	predictions[model_cfg.friendly_name] = Parallel(n_jobs=cfg.n_jobs)(delayed(get_arg_predictions_for_set)(set_id, arg_list) for set_id, arg_list in enumerate(arg_lists))
		# except Exception:
		#	log.warning(f'Multithreading failed! Reattempting without multithreading.')
		# log.info(f'Getting predictions for {len(arg_lists)} set(s) of {cfg.tuning.num_words} argument(s) * {len(cfg.tuning.args)} position(s) for {model_cfg.friendly_name}')
		# predictions[model_cfg.friendly_name] = [get_arg_predictions_for_set(set_id, arg_list) for set_id, arg_list in tqdm(enumerate(arg_lists), total = len(arg_lists))]
	
	# predictions = pd.DataFrame([d for model_name in predictions for set_id in predictions[model_name] for sentence_prediction in set_id for d in sentence_prediction])
	predictions = pd.DataFrame([d for model_name in predictions for sentence_set_prediction in predictions[model_name] for d in sentence_set_prediction])
	
	# predictions = predictions.query('exp_odds_name != odds_name').assign(
	# 	ratio_name = lambda df: df['arg_type'] + '/' + df['arg_position'],
	# 	odds_ratio = lambda df: df['exp_log_odds'] - df['log_odds']
	# ).drop(['log_odds', 'exp_log_odds', 'odds_name', 'exp_odds_name', 'arg_position'], axis=1)
	
	# predictions['freq'] = [candidate_freq_words[token] for token in predictions.token]
	
	return predictions
"""

"""
def arg_predictions(cfg: DictConfig, model_cfgs: List[str], args: Dict[str,List[str]]) -> List[Dict]:
	predictions = {}
	for model_cfg_path in model_cfgs:
	# do this so we can adjust the cfg tokens based on the model without messing up the 
	# actual config that was passed
	model_cfg = OmegaConf.load(model_cfg_path)
	exec(f'from transformers import {model_cfg.tokenizer}, {model_cfg.base_class}')

	base_class = model_cfg.base_class.lower().replace('formaskedlm', '')

	log.info(f'Initializing {base_class} model and tokenizer')
	tokenizer = create_tokenizer_with_added_tokens(model_cfg.string_id, eval(model_cfg.tokenizer), cfg.tuning.to_mask, **model_cfg.tokenizer_kwargs)
	model = eval(model_cfg.base_class).from_pretrained(model_cfg.string_id, **model_cfg.model_kwargs)
	model.resize_token_embeddings(len(tokenizer))

	tokens_to_mask = [t.lower() for t in cfg.tuning.to_mask] if 'uncased' in model_cfg.string_id else list(cfg.tuning.to_mask)
	tokens_to_mask = (tokens_to_mask + [chr(288) + t for t in tokens_to_mask]) if model_cfg.friendly_name == 'roberta' else tokens_to_mask	

	with torch.no_grad():
		# This reinitializes the novel token weights to random values 
		# to provide variability in model tuning
		# which matches the experimental conditions
		model_embedding_weights = getattr(model, base_class).embeddings.word_embeddings.weight
		model_embedding_dim = getattr(model, base_class).embeddings.word_embeddings.embedding_dim
		
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

	data = load_tuning_verb_data(cfg, model_cfg, tokenizer.mask_token)
	masked_data = data['masked_data']
	data = data['data']

	if not verify_tokenization_of_sentences(tokenizer, masked_data, tokens_to_mask, **model_cfg.tokenizer_kwargs):
		log.warning(f'Tokenization of sentences for {model_cfg.friendly_name} was affected by adding {tokens_to_mask}! Skipping this model.')
		continue

	inputs = [tokenizer(m, return_tensors='pt', padding=True) for m in masked_data]
			
	# We need to get the order/positions of the arguments for each sentence 
	# in the masked data so that we know which argument we are pulling 
	# out predictions for, because when we replace the argument placeholders 
	# with mask tokens, we lose information about which argument corresponds to which mask token
	arg_orders = [[w for w in strip_punct(s).split(' ') if w in cfg.tuning.args] for s in data]
	masked_token_indices = [[j for j, token_id in enumerate(i['input_ids'][0]) if token_id == tokenizer.convert_tokens_to_ids(tokenizer.mask_token)] for i in inputs]
	sentence_arg_indices = [dict(zip(arg, i)) for arg, i in tuple(zip(arg_orders, masked_token_indices))]

	# Run the model on the masked inputs to get the predictions
	model.eval()
	with torch.no_grad():
		outputs = [model(**i) for i in inputs]

	# Convert predicted logits to probabilities
	probs = [torch.nn.functional.softmax(output.logits, dim=-1) for output in outputs]

	# Organize the predictions by model name, argument type, argument position, and argument
	predictions[model_cfg.friendly_name] = []
	for arg_set in args:
		arg_set_predictions = {}
		for arg_type in arg_set:
			arg_set_predictions[arg_type] = {}
			for arg_position in arg_set:
				arg_set_predictions[arg_type][arg_position] = []
				for arg_indices, sentence, prob in zip(sentence_arg_indices, data, probs):
					other_args = [arg for arg in arg_indices if not arg == arg_position]
					for other_arg in other_args:
						sentence = sentence.replace(other_arg, tokenizer.mask_token)
					
					sentence_probs = []								
					for arg in arg_set[arg_type]:
						d = {'score' : float(prob[0,arg_indices[arg_position],tokenizer.convert_tokens_to_ids(arg)]),
							 'token' : tokenizer.convert_tokens_to_ids(arg), 
							 'token_str': arg,
							 'sequence' : sentence.replace(arg_position, arg),
							 f'{arg_type} nouns': ','.join(arg_set[arg_type]),
							 'random_seed': seed
						}
						
						sentence_probs.append(d)
					
					arg_set_predictions[arg_type][arg_position].append(sentence_probs)
		
		predictions[model_cfg.friendly_name].append(arg_set_predictions)

		# for args_words in tqdm(args, total = len(args)):
		# def predict_args_words(cfg: DictConfig, model_cfg: DictConfig, tokenizer: 'PreTrainedTokenizer', filler: 'FillMaskPipeline', args_words: Dict[str,List[str]]) -> Dict:
		# 	data = load_tuning_verb_data(cfg, model_cfg, tokenizer.mask_token, args_words)
		# 	results = {}
			
		# 	if not verify_tokenization_of_sentences(tokenizer, list(data.values()), tokens_to_mask, **model_cfg.tokenizer_kwargs):
		# 		log.warning(f'Tokenization of the set {args_words} was affected by adding {tokens_to_mask}! Skipping this set.')
		# 		return results
			
		# 	for arg_position in args_words:
		# 		predictions = {}
		# 		predictions[arg_position] = {}
		# 		for arg_type in args_words:
		# 			predictions[arg_position][arg_type] = []
		# 			for sentence in data[arg_position]:
		# 				targets = args_words[arg_type]
		# 				if 'uncased' in model_cfg.string_id:
		# 					targets = [t.lower() for t in targets]
						
		# 				if not re.search(rf'^{tokenizer.mask_token}', sentence) and model_cfg.friendly_name == 'roberta':
		# 					targets = [chr(288) + t for t in targets]
						
		# 				preds = filler(sentence, targets = targets)
		# 				for pred in preds:
		# 					pred.update({arg_type.replace(r"\[|\]", '') + ' nouns': ','.join(targets).replace(chr(288),'')})
					
		# 				predictions[arg_position][arg_type].append(preds)
				
		# 		results.update(predictions)
			
		# 	return results
		# 	#predictions[model_cfg.friendly_name].append(results)

		# model.eval()
		# filler = pipeline('fill-mask', model = model, tokenizer = tokenizer)

		# try:
		# 	log.info(f'Getting predictions for {len(args)} set(s) of {cfg.tuning.num_words} argument(s) * {len(cfg.tuning.args)} position(s) for {model_cfg.friendly_name} (n_jobs={cfg.n_jobs})')
		# 	with tqdm_joblib(tqdm(desc="", total = len(args))) as progress_bar:
		# 		predictions[model_cfg.friendly_name] = Parallel(n_jobs=cfg.n_jobs)(delayed(predict_args_words)(cfg, model_cfg, tokenizer, filler, args_words) for args_words in args)
		# except Exception:
		# 	log.warning(f'Multithreading failed! Reattempting without multithreading.')
		# 	predictions[model_cfg.friendly_name] = [predict_args_words(cfg, model_cfg, tokenizer, filler, args_words) for args_words in tqdm(args, total = len(args))]

		# print('')

		# import pickle as pkl
		# with open('predictions.pkl', 'wb') as f:
		# 	pkl.dump(predictions, f)

		return predictions
	"""
	
"""
def convert_predictions_to_df(predictions: Dict, candidate_freq_words: Dict[str,int]) -> pd.DataFrame:
	predictions = pd.DataFrame.from_dict({
		(model_name, arg_position, prediction_type, *results.values(), i) : 
		(model_name, arg_position, prediction_type, *results.values(), i) 
		for model_name in predictions
			for i, arg_set in enumerate(predictions[model_name])
				for prediction_type in arg_set
					for arg_position in arg_set[prediction_type]
						for arg in arg_set[prediction_type][arg_position]
							for results in arg
		}, 
		orient = 'index',
		columns = ['model_name', 'position', 
				   'prediction_type', 'p', 
				   'token_idx', 'token', 'sequence',
				   'predicted_nouns', 'random_seed', 'set_id']
	).reset_index(drop=True)

	# predictions.loc[:,'token'] = predictions['token'].str.replace(' ', '')
	# predictions.loc[:,'predicted_nouns'] = predictions['predicted_nouns'].str.replace(' ', '')

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

"""

"""
def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
	predictions_summary = predictions \
		.groupby([c for c in predictions.columns if not c in ['odds_ratio', 'sentence', 'sentence_num', 'token_id', 'token', 'freq']]) \
		.agg(mean_odds_ratio = ('odds_ratio', 'mean'),
			 se_odds_ratio = ('odds_ratio', 'sem')) \
		.reset_index()

	# predictions_summary = pd.DataFrame(
	# 	predictions_summary \
	# 		.pivot(index = ['model_name', 'random_seed', 'set_id'] + [c for c in predictions_summary.columns if c.endswith('nouns')],
	# 			   columns = ['ratio_name'], 
	# 		   	   values = ['mean_odds_ratio']) \
	# 		.to_records()
	# )

	args_words = predictions.prediction_type.unique().tolist()

	for arg in args_words:
		other_args = [a for a in args_words if not a == arg]
		for other_arg in other_args:
			predictions_summary[f'{other_arg} - {arg} in {arg}'] = \
				predictions_summary[f"('mean_surprisal', '{other_arg}', '{arg}')"] - \
				predictions_summary[f"('mean_surprisal', '{arg}', '{arg}')"]

	# this is our metric of good performance; lower is better
	predictions_summary['SumSq'] = (predictions_summary[[c for c in predictions_summary.columns if not re.search(r'(model_name)|(set_id)|(random_seed)|(nouns$)|\)$', c)]] ** 2).sum(axis = 1)

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

	predictions_summary = predictions_summary[['model_name', 'random_seed', 'set_id', 'SumSq'] + [c for c in predictions_summary.columns if re.search('( - )|(probability)|(bias)', c)]]
	total = predictions_summary.groupby('set_id')[[c for c in predictions_summary.columns if not c in ['model_name', 'set_id']]].mean().reset_index()
	total['model_name'] = 'average'
	predictions_summary = pd.concat([predictions_summary, total], ignore_index = True)

	noun_groups = predictions[['set_id'] + [c for c in predictions.columns if c.endswith('nouns')]].drop_duplicates()

	predictions_summary = predictions_summary.merge(noun_groups)

	return predictions_summary
"""

"""
# This allows us to view a progress bar on the parallel evaluations
# from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
import contextlib
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
	# Context manager to patch joblib to report into tqdm progress bar given as argument
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
"""

"""
def load_tuning_verb_data(cfg: DictConfig, model_cfg: DictConfig, mask_tok: str , args_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
	sentences = [strip_punct(line).strip() if cfg.strip_punct else line.strip() for line in cfg.tuning.data] + \
				[strip_punct(line).strip() if cfg.strip_punct else line.strip() for line in cfg.tuning.gen_args_data]
	sentences = [s.lower() for s in sentences] if 'uncased' in model_cfg.string_id else sentences

	# construct new argument dictionaries with the values from the non-target token, 
	# and the target token as a masked token so that we can generate predictions for
	# each argument in the position of the mask token for each possible resolution
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
"""

if __name__ == '__main__': 
	check_args()
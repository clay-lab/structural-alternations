# check_args.py
#
# check stats for candidate arguments for use in new verb experiments using a dataset which
# have strings that are tokenized as single words in all of the model types in conf/model
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
lg.set_verbosity_error()

from core import tuner_utils

log = logging.getLogger(__name__)

@hydra.main(config_path='conf', config_name='check_args')
def check_args(cfg: DictConfig) -> None:
	if not cfg.tuning.exp_type == 'newverb': 
		raise ValueError('Can only get args for new verb experiments!')
	
	print(OmegaConf.to_yaml(cfg, resolve=True))
	
	dataset 				= load_dataset(cfg.dataset_loc)
	model_cfgs_path 		= os.path.join(hydra.utils.get_original_cwd(), '..', 'conf', 'model')
	model_cfgs 				= [os.path.join(model_cfgs_path, f) for f in os.listdir(model_cfgs_path) if not f == 'multi.yaml']
	
	candidate_freq_words 	= get_candidate_words(dataset, model_cfgs, cfg.target_freq, cfg.range, cfg.min_length)
	
	if not cfg.target_freq == 'any':
		log.info(f'Found {len(candidate_freq_words)} words matching criteria: target_freq={cfg.target_freq}, range={cfg.range}')
	else:
		log.info(f'Found {len(candidate_freq_words)} words')
	
	predictions = get_word_predictions(cfg, model_cfgs, candidate_freq_words)
	assert not predictions.empty, 'No predictions were generated!'
	
	predictions_summary = summarize_predictions(predictions)	
	
	# Report the tokens treated most identically across models
	for ratio_name in predictions_summary.ratio_name.unique():
		averages = predictions_summary[(predictions_summary.model_name == 'average') & (predictions_summary.ratio_name == ratio_name)].reset_index(drop=True)[['model_name', 'token', 'ratio_name', 'SumSq']].sort_values('token')
		for model_name in [model_name for model_name in predictions_summary.model_name.unique() if not model_name == 'average']:
			model_predictions = predictions_summary[(predictions_summary.model_name == model_name) & (predictions_summary.ratio_name == ratio_name)].sort_values('token').reset_index(drop=True)
			
			if all(model_predictions.token.values == averages.token.values):
				averages[f'{model_name}_diff'] = averages.SumSq - model_predictions.SumSq
				if not 'SumSq_diff_average' in averages.columns:
					averages['SumSq_diff_average'] = [d**2 for d in averages[f'{model_name}_diff']]
				else:
					averages['SumSq_diff_average'] = [ss + d**2 for ss, d in zip(averages.SumSq_diff_average, averages[f'{model_name}_diff'])]
			else:
				raise Exception(f"Order of tokens doesn't match in {model_name} and averages!")
		
		most_similar = averages.sort_values('SumSq_diff_average').reset_index(drop=True)[['model_name', 'token', 'ratio_name', 'SumSq', 'SumSq_diff_average']]
		
		# best_average = predictions_summary[(predictions_summary.model_name == 'average') & (predictions_summary.ratio_name == ratio_name)].sort_values('SumSq').reset_index(drop=True)	
		most_similar_tokens = most_similar.iloc[:cfg.tuning.num_words*len(cfg.tuning.best_average_args),].token.unique()
		
		most_similar_sumsq_diffs = most_similar[most_similar.token.isin(most_similar_tokens)][['token', 'SumSq_diff_average']].drop_duplicates().set_index('token')
		most_similar_sumsq_diffs.SumSq_diff_average = ["{:.2f}".format(round(ss,2)) for ss in most_similar_sumsq_diffs.SumSq_diff_average]
		most_similar_sumsq_diffs = most_similar_sumsq_diffs.T
		most_similar_sumsq_diffs.columns.name = None
		
		most_similar = predictions_summary[predictions_summary.token.isin(most_similar_tokens)][['model_name', 'token', 'ratio_name', 'freq', 'SumSq']]
		most_similar.token = pd.Categorical(most_similar.token, most_similar_tokens)
		most_similar = most_similar.sort_values(['model_name', 'token'])
		most_similar.SumSq = ["{:.2f}".format(round(ss,2)) for ss in most_similar.SumSq]
		
		most_similar_freqs = most_similar[['token', 'freq']].drop_duplicates().set_index('token')
		most_similar_freqs.freq = [str(freq) + '   ' for freq in most_similar_freqs.freq]
		most_similar_freqs = most_similar_freqs.T
		most_similar_freqs.columns.name = None
		
		most_similar = most_similar.pivot(index=['model_name', 'ratio_name'], columns='token', values='SumSq').reset_index()
		most_similar.columns.name = None
		
		most_similar = pd.concat([most_similar, most_similar_freqs, most_similar_sumsq_diffs])
		log.info(f'{cfg.tuning.num_words} words/argument position * {len(cfg.tuning.best_average_args)} argument positions with most similar SumSq for ' + re.sub(r"\[|\]", "", ratio_name) + f' across models:\n\n{most_similar.to_string()}\n')
	
	# Report the tokens with the best average SumSq
	for ratio_name in predictions_summary.ratio_name.unique():
		best_average = predictions_summary[(predictions_summary.model_name == 'average') & (predictions_summary.ratio_name == ratio_name)].sort_values('SumSq').reset_index(drop=True)	
		best_average_tokens = best_average.iloc[:cfg.tuning.num_words*len(cfg.tuning.best_average_args),].token.unique()
		
		best_average = predictions_summary[predictions_summary.token.isin(best_average_tokens)][['model_name', 'token', 'ratio_name', 'freq', 'SumSq']]
		best_average.token = pd.Categorical(best_average.token, best_average_tokens)
		best_average = best_average.sort_values(['model_name', 'token'])
		best_average.SumSq = ["{:.2f}".format(round(ss,2)) for ss in best_average.SumSq]
		
		best_average_freqs = best_average[['token', 'freq']].drop_duplicates().set_index('token')
		best_average_freqs.freq = [str(freq) + '   ' for freq in best_average_freqs.freq]
		best_average_freqs = best_average_freqs.T
		best_average_freqs.columns.name = None
		
		best_average = best_average.pivot(index=['model_name', 'ratio_name'], columns='token', values='SumSq').reset_index()
		best_average.columns.name = None
		
		best_average = pd.concat([best_average, best_average_freqs])
		log.info(f'{cfg.tuning.num_words} words/argument position * {len(cfg.tuning.best_average_args)} argument positions with lowest average SumSq for ' + re.sub(r"\[|\]", "", ratio_name) + f':\n\n{best_average.to_string()}\n')
	
	# Report the tokens with the lowest SumSq for each model
	for model_name in [model_name for model_name in predictions_summary.model_name.unique() if not model_name == 'average']:
		for ratio_name in predictions_summary.ratio_name.unique():
			model_predictions = predictions_summary[(predictions_summary.model_name == model_name) & (predictions_summary.ratio_name == ratio_name)].sort_values('SumSq').reset_index(drop=True)
			best_for_model = model_predictions.iloc[:cfg.tuning.num_words*len(cfg.tuning.best_average_args),][['model_name', 'token', 'ratio_name', 'freq', 'SumSq']]
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
			log.info(f'{cfg.tuning.num_words} words/argument position * {len(cfg.tuning.best_average_args)} argument positions with lowest SumSq for ' + re.sub(r"\[|\]", "", ratio_name) + f' for {model_name}:\n\n{best_for_model.to_string()}\n')
	
	predictions = predictions.assign(
		run_id 					= os.path.split(os.getcwd())[-1],
		strip_punct 			= cfg.strip_punct,
		target_freq 			= cfg.target_freq,
		range 					= cfg.range,
		words_per_set 			= cfg.tuning.num_words,
		reference_sentence_type = cfg.tuning.reference_sentence_type,
		dataset 				= os.path.split(cfg.dataset_loc)[-1],
	)
	
	predictions_summary = predictions_summary.assign(
		run_id 					= os.path.split(os.getcwd())[-1],
		strip_punct 			= cfg.strip_punct,
		target_freq 			= cfg.target_freq,
		range 					= cfg.range,
		words_per_set 			= cfg.tuning.num_words,
		reference_sentence_type = cfg.tuning.reference_sentence_type,
		dataset 				= os.path.split(cfg.dataset_loc)[-1],
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
		
			subtlex.All_PoS_SUBTLEX		= subtlex.All_PoS_SUBTLEX.str.split('.')
			subtlex.All_freqs_SUBTLEX 	= subtlex.All_freqs_SUBTLEX.astype(str).str.split('.')
	
			subtlex = subtlex.explode(['All_PoS_SUBTLEX', 'All_freqs_SUBTLEX'])
	
			subtlex = subtlex.pivot_table(
				index 		= [c for c in subtlex.columns if not c in ['All_PoS_SUBTLEX', 'All_freqs_SUBTLEX']],
				columns 	= 'All_PoS_SUBTLEX',
				values 		= 'All_freqs_SUBTLEX',
				fill_value 	= 0
			)
	
			subtlex = pd.DataFrame(subtlex.to_records())
			
			subtlex_dir = os.path.split(subtlex_loc)[0]
			log.info(f'Saving file at {os.path.join(subtlex_dir, "subtlex_freqs_formatted.csv")}')
			subtlex.to_csv(os.path.join(subtlex_dir, 'subtlex_freqs_formatted.csv'), index=False)
		except KeyError:
			log.error('SUBTLEX xlsx file not in expected format.')
			return		
	else:
		try:
			subtlex = pd.read_csv(subtlex_loc)
		except (ValueError, FileNotFoundError):
			log.error(f'SUBTLEX file not found @ {subtlex_loc}.')
			return
			
	return subtlex

def get_candidate_words(dataset: pd.DataFrame, model_cfgs: List[str], target_freq: int, tolerance: int, min_length: int) -> Dict[str,str]:
	# Filter to words that occur primarily as nouns
	dataset = dataset[dataset.Dom_PoS_SUBTLEX == 'Noun']
	
	dataset = dataset[['Word', 'Noun']]
	
	if not target_freq == 'any':
		dataset = dataset.loc[dataset['Noun'].isin(list(range(target_freq - tolerance, target_freq + tolerance)))].copy().reset_index(drop = True)
	
	dataset = dataset[~dataset['Word'].str.match('^[aeiou]')] # to avoid a/an issues
	dataset = dataset[dataset['Word'].str.contains('[aeiouy]')] # must contain at least one vowel (to avoid acronyms/abbreviations)
	dataset = dataset[dataset.Word.str.len() >= min_length] # to avoid some other junk
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
	
	candidate_words = {word : dataset.loc[dataset['Word'] == word,'Noun'].to_numpy()[0] for word in candidate_words}
			
	return candidate_words

def get_word_predictions(cfg: DictConfig, model_cfgs: List[str], candidate_freq_words: Dict[str,int]) -> pd.DataFrame:
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
		args_in_order = [[word for word in strip_punct(sentence).split(' ') if word in cfg.tuning.best_average_args] for sentence in data]
		masked_token_indices = [[index for index, token_id in enumerate(i['input_ids'][0]) if token_id == tokenizer.convert_tokens_to_ids(tokenizer.mask_token)] for i in inputs]
		sentence_arg_indices = [dict(zip(arg, index)) for arg, index in tuple(zip(args_in_order, masked_token_indices))]
		
		# Run the model on the masked inputs to get the predictions
		model.eval()
		with torch.no_grad():
			outputs = [model(**i) for i in inputs]
		
		# Convert predicted logits to log probabilities
		sentence_logprobs = [nn.functional.log_softmax(output.logits, dim=-1) for output in outputs]
		
		# Organize the predictions by model name, argument type, argument position, and argument
		log.info(f'Getting predictions for {len(candidate_freq_words)} word(s) * {len(cfg.tuning.best_average_args)} argument position(s) for {model_cfg.friendly_name}')
		predictions[model_cfg.friendly_name] = []
		for arg in tqdm(candidate_freq_words):
			for arg_type in cfg.tuning.best_average_args:
				predictions_token_arg_sentence = []
				for sentence_num, (arg_indices, sentence, logprob) in enumerate(zip(sentence_arg_indices, data, sentence_logprobs)):
					if model_cfg.friendly_name == 'roberta' and not sentence.startswith(arg_type):
						arg_token_id = tokenizer.convert_tokens_to_ids(chr(288) + arg)
					else:
						arg_token_id = tokenizer.convert_tokens_to_ids(arg)
					
					for arg_position, arg_index in [(arg_position, arg_index) for arg_position, arg_index in arg_indices.items() if not arg_position == arg_type]:
						log_odds = logprob[0,arg_index,arg_token_id]
						exp_log_odds = logprob[0,arg_indices[arg_type],arg_token_id]
						odds_ratio = exp_log_odds - log_odds
						
						prediction_row = {
							'odds_ratio' : odds_ratio,
							'ratio_name' : arg_type + '/' + arg_position,
							'token_id' : arg_token_id,
							'token' : arg,
							'sentence' : sentence,
							'sentence_category' : 'tuning' if strip_punct(sentence.lower()) in [strip_punct(s.lower()) for s in cfg.tuning.data] else 'check_args',
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
	for arg in cfg.tuning.best_average_args:
		other_args = [other_arg for other_arg in cfg.tuning.best_average_args if not other_arg == arg]
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
				[strip_punct(line).strip() if cfg.strip_punct else line.strip() for line in cfg.tuning.check_args_data]
	sentences = [s.lower() for s in sentences] if 'uncased' in model_cfg.string_id else sentences
	
	masked_data = []
	for s in sentences:
		for arg in cfg.tuning.best_average_args:
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
			
			title = f'Correlation of token SumSq differences\nfor log odds {ratio_name.replace("[", "").replace("]", "")} ratios\n'
			title += ('\nWithout' if all(predictions_summary.strip_punct.values) else '\nWith') + ' punctuation, '
			title += f'target frequency: {predictions_summary.target_freq.unique()[0]}' + (f' (\u00B1{predictions_summary.range.unique()[0]})' if predictions_summary.target_freq.unique()[0] != 'any' else '')
			title += f'\ndataset: {os.path.splitext(predictions_summary.dataset.unique()[0])[0]}'
			title += f'\nsentence type: {predictions_summary.reference_sentence_type.unique()[0]}' if predictions_summary.reference_sentence_type.unique()[0] != 'none' else ''
			title += f'\ndata from {cfg.tuning.name}'
			g.fig.suptitle(title, y = 0.88, fontsize='medium', x = 0.675)
			pdf.savefig()
			plt.close('all')
			del g

if __name__ == '__main__':
	
	check_args()
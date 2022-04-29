# check_args.py
#
# check stats for candidate arguments for use in new verb experiments using a dataset which
# have strings that are tokenized as single words in all of the model types in conf/model
import os
import re
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn

from tqdm import tqdm
from typing import *
from omegaconf import DictConfig, OmegaConf, open_dict

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import pearsonr
from transformers import logging as lg
from transformers import AutoTokenizer, AutoModelForMaskedLM
lg.set_verbosity_error()

from core import tuner_utils

log = logging.getLogger(__name__)

# subtlex lists this as a noun, but it's clearly not. it keeps showing up,
# so this lets us get rid of it
MANUAL_EXCLUDE = ['doesn']

OmegaConf.register_new_resolver(
	'tuning_name',
	lambda tunings: ','.join(tunings)
)

@hydra.main(config_path='conf', config_name='check_args')
def check_args(cfg: DictConfig) -> None:
	'''
	Generates candidate nouns that models consider roughly equally likely in different argument positions. Saves results to disk
	
		params:
			cfg (dictconfig)	: a configuration specifying sentence data to use, as well as target frequency for candidate nouns
	'''
	load_tunings(cfg)
	if not all(cfg.tuning[tuning].exp_type == 'newverb' for tuning in cfg.tuning):
		raise ValueError('Can only get args for new verb experiments!')
	
	print(OmegaConf.to_yaml(cfg, resolve=True))
	
	dataset 				= load_dataset(os.path.join(hydra.utils.get_original_cwd(), cfg.dataset_loc.replace(hydra.utils.get_original_cwd(), '')))
	model_cfgs_path 		= os.path.join(hydra.utils.get_original_cwd(), 'conf', 'model')
	model_cfgs 				= sorted([os.path.join(model_cfgs_path, f) for f in os.listdir(model_cfgs_path)])
	
	candidate_freq_words 	= get_candidate_words(dataset, model_cfgs, cfg.target_freq, cfg.range, cfg.min_length)
	
	predictions 			= get_word_predictions(cfg, model_cfgs, candidate_freq_words)
	assert not predictions.empty, 'No predictions were generated!'
	
	predictions_summary 	= summarize_predictions(predictions)	
	
	log_predictions_summary(predictions_summary, cfg)
	
	predictions 			= add_hyperparameters_to_df(predictions, cfg)
	predictions_summary 	= add_hyperparameters_to_df(predictions_summary, cfg)
	
	# Do this to save the original tensors
	predictions.to_pickle('predictions.pkl.gz')
	
	# Save a CSV to make things easier to work with later
	predictions.odds_ratio = [float(o_r) for o_r in predictions.odds_ratio]
	predictions.to_csv('predictions.csv.gz', index=False, na_rep='NaN')
	
	predictions_summary.to_csv('predictions_summary.csv.gz', index=False, na_rep='NaN')
	
	# plot the correlations of the sumsq for each pair of model types and report R**2
	plot_correlations(predictions_summary, cfg)

def load_tunings(cfg: DictConfig) -> None:
	'''
	Loads tuning data for each dataset specified in cfg.tunings
	
		params:
			cfg (dictConfig): the config to load tuning data for
	'''
	d = HydraConfig.get().runtime.config_sources[1].path
	with open_dict(cfg):
		cfg['tuning'] = {}
		for tuning in cfg.tunings:
			cfg.tuning[tuning] = OmegaConf.load(os.path.join(d, 'tuning', f'{tuning}.yaml'))

def load_dataset(dataset_loc: str) -> pd.DataFrame:
	'''
	Placeholder for use when implementing support for other datasets.
	Loads a dataset containing noun frequency information.
	Currently only SUBTLEX is supported.
	
		params:
			dataset_loc (str): the path to the file containing the noun frequency dataset
	'''
	if 'subtlex' in dataset_loc.lower():
		return load_subtlex(dataset_loc)
	else:
		raise NotImplementedError('Support for other noun frequency datasets is not currently implemented.')

def load_subtlex(subtlex_loc: str) -> pd.DataFrame:
	'''
	Loads the SUBTLEX dataset containing noun frequency information.
	Due to the way SUBTLEX is formatted by default, doing this can be quite slow, so also save a reformatted version for quicker loading
	if it has not already been saved in this format.
	
		params:
			subtlex_loc (str): 	the path to the SUBTLEX dataset in XLSX or CSV format.
								note that the assumption is that XLSX format has not been reformatted for faster loading
	'''
	if subtlex_loc.endswith('.xlsx'):
		subtlex = pd.read_excel(subtlex_loc)
		subtlex = subtlex[~(subtlex.All_PoS_SUBTLEX.isnull() | subtlex.All_freqs_SUBTLEX.isnull())]
	
		# Reformat and save for faster use in the future
		log.info('Reading in and reshaping SUBTLEX PoS frequency file')
		log.info('A reshaped version will be saved for faster future use as "subtlex_freqs_formatted.csv.gz"')
		
		subtlex.All_PoS_SUBTLEX		= subtlex.All_PoS_SUBTLEX.str.split('.')
		subtlex.All_freqs_SUBTLEX 	= subtlex.All_freqs_SUBTLEX.astype(str).str.split('.')
		
		subtlex = subtlex.explode(['All_PoS_SUBTLEX', 'All_freqs_SUBTLEX'])
		
		subtlex = subtlex.pivot_table(
			index 		= [c for c in subtlex.columns if not c in ['All_PoS_SUBTLEX', 'All_freqs_SUBTLEX']],
			columns 	= 'All_PoS_SUBTLEX',
			values 		= 'All_freqs_SUBTLEX',
			fill_value 	= 0
		)
		
		subtlex 	= pd.DataFrame(subtlex.to_records())
		subtlex_dir = os.path.split(subtlex_loc)[0]
		
		log.info(f'Saving file at {os.path.join(subtlex_dir, "subtlex_freqs_formatted.csv.gz")}')
		subtlex.to_csv(os.path.join(subtlex_dir, 'subtlex_freqs_formatted.csv.gz'), index=False, na_rep='NaN')	
	else:
		subtlex = pd.read_csv(subtlex_loc)
	
	subtlex = subtlex[~subtlex.Word.isin(MANUAL_EXCLUDE)]
	
	return subtlex

def get_candidate_words(
	dataset: pd.DataFrame, 
	model_cfgs: List[str], 
	target_freq: int, 
	tolerance: int, 
	min_length: int
) -> Dict[str,int]:
	'''
	Generates candidate words tokenized as single tokens in each model that match frequency requirements
	
		params:
			dataset (pd.DataFrame)		: a dataset containing words and frequency information.
										  words should be in a column named 'Word'
										  frequency (count) should be in a column named 'Noun'
			model_cfgs (list)			: a list of paths to cfg files in yaml format specifying model parameters
			target_freq (int)			: the desired frequency of candidate words
			tolerance (int)				: candidates words must be within +/- tolerance of the target freq to be included
			min_length (int)			: the minimum acceptable length of a target word
		
		returns:
			candidate_words_freqs (dict): a dictionary mapping words meeting requirements to their frequency of occurence in the dataset
	'''
	dataset 		= dataset[dataset.Dom_PoS_SUBTLEX == 'Noun']
	dataset 		= dataset[['Word', 'Noun']]
	
	if not target_freq == 'any':
		dataset 	= dataset.loc[dataset.Noun.isin(list(range(target_freq - tolerance, target_freq + tolerance)))].copy().reset_index(drop=True)
	
	# filter out potentially problematic words
	dataset 		= dataset[~dataset.Word.str.match('^[aeiou]')] # to avoid a/an issues
	dataset 		= dataset[dataset.Word.str.contains('[aeiouy]')] # must contain at least one vowel (to avoid acronyms/abbreviations)
	dataset 		= dataset[dataset.Word.str.len() >= min_length] # to avoid some other abbrevations
	candidate_words = dataset.Word.tolist()
	
	# To do the experiments, we need each argument word to be tokenized as a single word
	# so we check that here and filter out those that are tokenized as multiple subwords
	log.info('Finding candidate words in tokenizers')
	
	for model_cfg_path in model_cfgs:
		model_cfg 			= OmegaConf.load(model_cfg_path)
		tokenizer 			= AutoTokenizer.from_pretrained(model_cfg.string_id, **model_cfg.tokenizer_kwargs)
		
		if model_cfg.friendly_name == 'roberta':
			# we want to make sure the preceding space versions are tokenized as one token as well for roberta
			candidate_words = [word for word in candidate_words if tuner_utils.verify_tokens_exist(tokenizer, chr(288) + word)]
		
		candidate_words 	= [word for word in candidate_words if tuner_utils.verify_tokens_exist(tokenizer, word)]
	
	candidate_words_freqs 	= {word : dataset.loc[dataset.Word == word].Noun.to_numpy()[0] for word in candidate_words}
	
	log.info(f'Found {len(candidate_words_freqs)} words matching criteria: target_freq={target_freq}, range={tolerance if target_freq != "any" else np.nan}')
			
	return candidate_words_freqs

def get_word_predictions(
	cfg: DictConfig, 
	model_cfgs: List[str], 
	candidate_freq_words: Dict[str,int]
) -> pd.DataFrame:
	all_gfs 			= list(dict.fromkeys([gf for tuning in cfg.tuning for gf in cfg.tuning[tuning].best_average.keys()]))
	all_data_sentences 	= [tuner_utils.strip_punct(s).lower() for tuning in cfg.tuning for s in cfg.tuning[tuning].data]
	all_sentences 		= {tuning: [tuner_utils.strip_punct(s).lower() for s in cfg.tuning[tuning].data + cfg.tuning[tuning].check_args_data] for tuning in cfg.tuning}
	
	predictions 		= []
	# there is a lot similar here to how tuner is set up. maybe we could integrate some of this?
	for model_cfg_path in model_cfgs:
		
		# do this so we can adjust the cfg tokens based on the model without messing up the 
		# actual config that was passed
		model_cfg 	= OmegaConf.load(model_cfg_path)
		to_mask 	= list(dict.fromkeys([t for tuning in cfg.tuning for t in cfg.tuning[tuning].to_mask]))
		
		log.info('')
		log.info(f'Initializing {model_cfg.friendly_name} model and tokenizer')
		to_add 		= tuner_utils.format_data_for_tokenizer(data=to_mask, mask_token='', string_id=model_cfg.string_id, remove_punct=cfg.strip_punct)
		if model_cfg.friendly_name == 'roberta':
			to_add 	= tuner_utils.format_roberta_tokens_for_tokenizer(to_add)
		
		tokenizer 	= tuner_utils.create_tokenizer_with_added_tokens(model_cfg.string_id, to_add, **model_cfg.tokenizer_kwargs)
		
		model 		= AutoModelForMaskedLM.from_pretrained(model_cfg.string_id, **model_cfg.model_kwargs)
		model.resize_token_embeddings(len(tokenizer))
		
		getattr(model, model.config.model_type).embeddings.word_embeddings.weight, seed = \
			tuner_utils.reinitialize_token_weights(
				word_embeddings=getattr(model, model.config.model_type).embeddings.word_embeddings.weight,
				tokens_to_initialize=to_add,
				tokenizer=tokenizer,
			)
		
		data, masked_data = load_tuning_verb_data(cfg, model_cfg, tokenizer.mask_token)
		
		assert tuner_utils.verify_tokenization_of_sentences(tokenizer, masked_data, to_add, **model_cfg.tokenizer_kwargs), \
			f'Tokenization of sentences for {model_cfg.friendly_name} was affected by adding {to_add}!'
		
		inputs 					= tokenizer(masked_data, return_tensors='pt', padding=True)
		
		# We need to get the order/positions of the arguments for each sentence 
		# in the masked data so that we know which argument we are pulling 
		# out predictions for, because when we replace the argument placeholders 
		# with mask tokens, we lose information about which argument corresponds to which mask token
		args_in_order 			= [[word for word in tuner_utils.strip_punct(sentence).split(' ') if word in all_gfs] for sentence in data]
		masked_token_indices 	= [[index for index, token_id in enumerate(i) if token_id == tokenizer.convert_tokens_to_ids(tokenizer.mask_token)] for i in inputs['input_ids']]
		sentence_arg_indices 	= [dict(zip(arg, index)) for arg, index in tuple(zip(args_in_order, masked_token_indices))]
		
		# Run the model on the masked inputs to get the predictions
		model.eval()
		with torch.no_grad():
			outputs = model(**inputs)
		
		# Convert predicted logits to log probabilities
		logprobs = nn.functional.log_softmax(outputs.logits, dim=-1)
		
		# Organize the predictions by model name, argument type, argument position, and argument
		# we can probably make this go with the collect results function currently in tuner.py
		# we'll have to load the verb data differently, though. we'd also have to deal with the eval groups (new words) not existing in the arg indices
		log.info(f'Getting predictions for {len(candidate_freq_words)} word(s) * {len(all_gfs)} argument position(s) for {model_cfg.friendly_name}')
		for arg in tqdm(candidate_freq_words):
			for arg_type in all_gfs:
				for sentence_num, (arg_indices, sentence, logprob) in enumerate(zip(sentence_arg_indices, data, logprobs)):
					
					arg_token_id 		= tokenizer.convert_tokens_to_ids(arg)
					
					if model_cfg.friendly_name == 'roberta' and not sentence.startswith(arg_type):
						arg_token_id 	= tokenizer.convert_tokens_to_ids(chr(288) + arg)
					
					for arg_position, arg_index in [(arg_position, arg_index) for arg_position, arg_index in arg_indices.items() if not arg_position == arg_type]:
						log_odds 		= logprob[arg_index,arg_token_id]
						exp_log_odds 	= logprob[arg_indices[arg_type],arg_token_id]
						odds_ratio 		= exp_log_odds - log_odds
						tuning 			= [tuning for tuning in cfg.tuning if tuner_utils.strip_punct(sentence.lower()) in all_sentences[tuning]][0]
						
						predictions.append({
							'odds_ratio' 		: odds_ratio,
							'ratio_name' 		: f'{arg_type}/{arg_position}',
							'token_id' 			: arg_token_id,
							'token' 			: arg,
							'sentence' 			: sentence,
							'sentence_category' : 'tuning' if tuner_utils.strip_punct(sentence).lower() in all_data_sentences else 'check_args',
							'tuning' 			: tuning,
							'sentence_num' 		: all_sentences[tuning].index(tuner_utils.strip_punct(sentence).lower()),
							'model_name' 		: model_cfg.friendly_name,
							'random_seed' 		: seed,
							'freq' 				: candidate_freq_words[arg],
						})
		
		del model
		del tokenizer
		
	predictions = pd.DataFrame(predictions)
	
	# because these are log odds ratios, log(x/y) = -log(y/x). Thus, we only report the unique combinations for printing.
	unique_ratio_names = []
	for arg in all_gfs:
		other_args = [other_arg for other_arg in all_gfs if not other_arg == arg]
		for other_arg in other_args:
			if not (f'{other_arg}/{arg}' in unique_ratio_names or f'{arg}/{other_arg}' in unique_ratio_names):
				unique_ratio_names.append(f'{arg}/{other_arg}')
	
	predictions = predictions[predictions.ratio_name.isin(unique_ratio_names)].reset_index(drop=True)
	
	return predictions

def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
	# get the mean odds ratios for each argument in each position
	predictions_summary = predictions \
		.groupby([c for c in predictions.columns if not c in ['tuning', 'odds_ratio', 'sentence', 'sentence_category', 'sentence_num']]) \
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
			[c for c in predictions_summary.columns if not re.search('(model_name)|(tuning)|(random_seed)|(token_id)|(SumSq)|(odds_ratio)', c)]
		)[[c for c in predictions_summary.columns if re.search('(SumSq)|(odds_ratio)', c)]] \
		.mean() \
		.reset_index() \
		.assign(
			model_name 	= 'average',
			random_seed = np.nan,
			token_id 	= np.nan,
	)
	
	predictions_summary = pd.concat([predictions_summary, averages], ignore_index=True)
	predictions_summary = predictions_summary.sort_values(['token', 'model_name', 'ratio_name']).reset_index(drop=True)
	predictions_summary['tuning'] = ','.join(predictions.tuning.unique())
	
	return predictions_summary

def load_tuning_verb_data(
	cfg: DictConfig,
	model_cfg: DictConfig, 
	mask_token: str
) -> Tuple[List[str]]:
	# we might be able to integrate this with how tuner loads these data, and then use that to integrate the arg predictions with collects results
	# though this may not be possible directly because we don't have specified eval groups
	sentences = [strip_punct(line).strip() if cfg.strip_punct else line.strip() for tuning in cfg.tuning for line in cfg.tuning[tuning].data] + \
				[strip_punct(line).strip() if cfg.strip_punct else line.strip() for tuning in cfg.tuning for line in cfg.tuning[tuning].check_args_data]
	sentences = [s.lower() for s in sentences] if 'uncased' in model_cfg.string_id else sentences
	
	all_gfs = list(dict.fromkeys([gf for tuning in cfg.tuning for gf in cfg.tuning[tuning].best_average.keys()]))
	
	masked_data = []
	for s in sentences:
		for arg in all_gfs:
			s = s.replace(arg, mask_token)
		
		masked_data.append(s)
	
	return sentences, masked_data

def log_predictions_summary(predictions_summary: pd.DataFrame, cfg: DictConfig) -> None:
	'''
	Report summaries of token with most similar, best average, and best per model sumsqs
	
		params:
			predictions_summary (pd.DataFrame)	: a dataframe containing odds ratios predictions for tokens
			cfg (dictconfig)					: a configuration with settings for the experiment
	'''
	all_gfs 	= list(dict.fromkeys([gf for tuning in cfg.tuning for gf in cfg.tuning[tuning].best_average.keys()]))
	num_words 	= max([cfg.tuning[tuning].num_words for tuning in cfg.tuning])
	
	df 			= predictions_summary.copy()
	model_dfs 	= [df for _, df in df[df.model_name != 'average'].groupby('model_name')]
	averages	= df[df.model_name == 'average'].reset_index(drop=True)[['model_name', 'token', 'ratio_name', 'SumSq', 'freq']].sort_values('token')
	
	for model_df in model_dfs:
		model_name 	= model_df.model_name.unique()[0]
		model_df 	= model_df.copy()
		model_df 	= model_df.sort_values('token').reset_index(drop=True)
		assert all(model_df.token.values == averages.token.values), f"Order of tokens doesn't match in {model_name} and averages!"
		
		averages[f'{model_name}_diff'] = averages.SumSq - model_df.SumSq
		if not 'SumSq_diff_average' in averages.columns:
			averages['SumSq_diff_average'] 	= [d**2 for d in averages[f'{model_name}_diff']]
		else:
			averages.SumSq_diff_average 	= [ss + d**2 for ss, d in zip(averages.SumSq_diff_average, averages[f'{model_name}_diff'])]
	
	best_average 	= df[df.model_name == 'average'].sort_values('SumSq').reset_index(drop=True)
	
	dfs = [averages, best_average]
	dfs.extend(model_dfs)
	
	metrics = ['SumSq_diff_average'] + ['SumSq' for df in dfs if not df.equals(averages)]
	labels 	= ['most similar SumSq', 'lowest average SumSq'] + [f'lowest SumSq for {df.model_name.unique()[0]}' for df in model_dfs]
	
	for dataset, metric, label in zip(dfs, metrics, labels):
		for ratio_name, df in dataset.groupby('ratio_name'):
			if metric == 'SumSq_diff_average':
				df.model_name 		= df.model_name.unique()[0] + ' \u0394SumSq'
			
			df						= df.sort_values(metric).reset_index(drop=True)
			df_tokens				= df.iloc[:num_words*len(all_gfs),].token.unique()
			
			df 						= df[df.token.isin(df_tokens)]
			df.SumSq 				= ['{:.2f}'.format(round(ss,2)) for ss in df.SumSq]
			
			if metric == 'SumSq_diff_average':
				df[metric]			= ['{:.2f}'.format(round(ss,2)) for ss in df[metric]]
			
			df_freqs 				= df[['token', 'freq']].drop_duplicates().set_index('token')
			df_freqs.freq 			= [str(freq) + '   ' for freq in df_freqs.freq]
			df_freqs 				= df_freqs.T
			df_freqs.columns.name 	= None
			
			# add info about individual models to the summary stats to see how they stack up
			if metric == 'SumSq_diff_average' or label == 'lowest average SumSq':
				to_concat 	= []
				datasets 	= model_dfs.copy()
				datasets 	= datasets + [best_average.copy()] if not label == 'lowest average SumSq' else datasets
				
				for dataset in datasets:
					dataset 				= dataset.copy()
					dataset 				= dataset
					dataset 				= dataset[dataset.token.isin(df.token.to_numpy())]
					dataset.token 			= pd.Categorical(dataset.token, df_tokens)
					dataset 				= dataset.sort_values('token')
					dataset.SumSq 			= [f'{ss:.2f}' for ss in dataset.SumSq]
					dataset 				= dataset
					dataset 				= dataset.pivot(index=['model_name', 'ratio_name'], columns='token', values='SumSq')
					dataset 				= dataset.reset_index()
					dataset.columns.name 	= None
					to_concat.append(dataset)
			else:
				to_concat = []
			
			# we have to do this categorical on the tokens to make the sorting work correctly when pivoting
			df.token				= pd.Categorical(df.token, categories=df.token.unique(), ordered=True)
			df 			 			= df.pivot(index=['model_name', 'ratio_name'], columns='token', values=metric).reset_index()
			df.columns.name 		= None
			
			df 						= pd.concat([df] + to_concat + [df_freqs])
			log.info(f'{num_words} words/argument position * {len(all_gfs)} argument positions with {label} for ' + re.sub(r"\[|\]", "", ratio_name) + f':\n\n{df.to_string()}\n')

def add_hyperparameters_to_df(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
	'''
	Adds config information to a dataframe
	
		params:
			df (pd.DataFrame)	: the dataframe to add config information to
			cfg (DictConfig)	: a config containing information to add to the dataframe
		
		returns:
			df (pd.DataFrame)	: the dataframe with information from config added
	'''
	num_words = max([cfg.tuning[tuning].num_words for tuning in cfg.tuning])
	
	df = df.assign(
		run_id 					= os.path.split(os.getcwd())[-1],
		strip_punct 			= cfg.strip_punct,
		target_freq 			= cfg.target_freq,
		range 					= cfg.range,
		words_per_set 			= num_words,
		dataset 				= os.path.split(cfg.dataset_loc)[-1],
	)
	
	if all(tuning in cfg.tuning for tuning in df.tuning.unique()):
		df = df.assign(reference_sentence_type = [cfg.tuning[tuning].reference_sentence_type for tuning in df.tuning])
	else:
		df = df.assign(reference_sentence_type = ','.join([cfg.tuning[tuning].reference_sentence_type for tuning in cfg.tuning]))
		
	return df

def plot_correlations(predictions_summary: pd.DataFrame, cfg: DictConfig) -> None:
	'''
	Plots correlations of sumsq odds ratios for all models
	
		params:
			cfg (dictconfig)					: a configuration containing experiment options
			predictions_summary (pd.DataFrame)	: a dataframe containing a summary of argument predictions
	'''
	def corrfunc(x, y, **kwargs):
		'''Calculates pearson r add adds it to the corr plot'''
		if not all(x.values == y.values):
			r, _ 	= pearsonr(x, y)
			r2 		= r**2
			ax 		= plt.gca()
			label 	= 'R\u00b2 = ' + f'{r2:.2f}' if not all(x.values == y.values) else ''
			log.info('R\u00b2 of SumSq for {:21s}{:.2f}'.format(x.name + ', ' + y.name + ':', r2))
			ax.annotate(label, xy=(.1,.9), xycoords=ax.transAxes, zorder=10, bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=2))			
	
	corr = predictions_summary[['model_name', 'ratio_name', 'token', 'SumSq']][predictions_summary.model_name != 'average'] \
		.pivot(index=['ratio_name', 'token'], columns='model_name', values='SumSq') \
		.reset_index()
	
	corr.columns.name = None
	for ratio_name, ratio_name_corr in corr.groupby('ratio_name'):
		ratio_name_corr = ratio_name_corr.drop('ratio_name', axis=1)
		g = sns.pairplot(
			ratio_name_corr, 
			kind='reg', 
			corner=True, 
			plot_kws=dict(
				line_kws=dict(
					linewidth=1, 
					color='r',
					zorder=5
				), 
				scatter_kws=dict(
					s=8, 
					linewidth=0
				)
			)
		)
		
		g.map(corrfunc)
		
		title = f'Correlation of token SumSq differences\nfor log odds {ratio_name.replace("[", "").replace("]", "")} ratios\n'
		title += ('\nWithout' if all(predictions_summary.strip_punct.values) else '\nWith') + ' punctuation, '
		title += f'target frequency: {predictions_summary.target_freq.unique()[0]}' + (f' (\u00B1{predictions_summary.range.unique()[0]})' if predictions_summary.target_freq.unique()[0] != 'any' else '')
		title += f'\ndataset: {os.path.splitext(predictions_summary.dataset.unique()[0])[0]}'
		title +=  '\ndata from ' + ',\n'.join(predictions_summary.tuning.unique()[0].split(','))
		g.fig.suptitle(title, y = 0.88, fontsize='medium', x = 0.675)
		
		plt.savefig('correlations.pdf')
		plt.close('all')
		del g

if __name__ == '__main__':
	
	check_args()
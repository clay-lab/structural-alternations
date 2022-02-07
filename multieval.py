# multieval.py
# 
# Application entry point for evaluating and summarizing multiple masked language models.

import os
import re
import sys
import gzip
import hydra
import logging

import numpy as np
import pandas as pd
import pickle as pkl

from glob import glob
from math import floor
from typing import List
from omegaconf import DictConfig, OmegaConf, open_dict
from distutils.dir_util import copy_tree, remove_tree
from distutils.file_util import copy_file

from core.tuner import Tuner

config_path='conf'

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
	'criterianame', 
	lambda criteria: ','.join([
		c.replace('always_masked',    'a_m'). \
		  replace('bert_masking',     'b_m'). \
		  replace('no_masking',       'n_m'). \
		  replace('roberta_masking',  'r_m'). \
		  replace('no_punctuation',   'n_p'). \
		  replace('with_punctuation', 'w_p'). \
		  replace('^bert',            'bert')
		for c in criteria.split(',')
	])
)

@hydra.main(config_path=config_path, config_name='multieval')
def multieval(cfg: DictConfig) -> None:
	
	print(OmegaConf.to_yaml(cfg))
	
	# Get directory information
	source_dir = hydra.utils.get_original_cwd()
	starting_dir = os.getcwd()
	
	# clean the current directory log if needed
	with open('multieval.log', 'r') as log_file:
		f = log_file.read()
	
	if f: 
		logging.shutdown()
		os.remove('multieval.log')
	
	# Get a regex for the score file name so we can just load it if it already exists
	# make this global so we can access it in the helper functions later
	global scores_name
	scores_name = 'surprisals' if cfg.data.new_verb else 'odds_ratios' if cfg.data.entail else 'scores'
	
	if cfg.epoch == 'None':
		cfg.epoch = None
		score_file_name = '(.hydra|eval.log|(' + cfg.data.friendly_name + f'-(([0-9]+)-+)+(accuracies.csv.gz|tsne.csv.gz|tsne-plots.pdf|{scores_name}-plots.pdf|{scores_name}.(csv|pkl).gz|cossims.csv.gz)))'
		log.warning('Epoch not specified. If no evaluation has been performed, evaluation will be performed on the final epoch. Otherwise, all epochs on which evaluation has been performed will be loaded for each model.')
	elif 'best' in cfg.epoch:
		score_file_name = '(.hydra|eval.log|(' + cfg.data.friendly_name + f'-(([0-9]+)-+)+{cfg.epoch}-(accuracies.csv.gz|tsne.csv.gz|tsne-plots.pdf|{scores_name}-plots.pdf|{scores_name}.(csv|pkl).gz|cossims.csv.gz)))'
	else:
		score_file_name = '(.hydra|eval.log|(' + cfg.data.friendly_name + '-' + cfg.epoch + f'-(accuracies.csv.gz|tsne.csv.gz|tsne-plots.pdf|{scores_name}-plots.pdf|{scores_name}.(csv|pkl).gz|cossims.csv.gz)))'
	
	# Get checkpoint dirs in outputs
	chkpt_dirs = os.path.join(hydra.utils.to_absolute_path(cfg.dir), '**')
	chkpt_dirs = [os.path.split(f)[0] for f in glob(chkpt_dirs, recursive = True) if f.endswith('weights.pkl.gz')]
	
	if not chkpt_dirs:
		print(f'No model information found in dir {cfg.dir}. Did you put in the right directory path?')
		sys.exit(1)
	
	# filter paths based on criteria
	criteria = cfg.criteria.split(',')
	criteria = [''] if criteria == ['all'] else criteria # do this to give us a reasonable dir name
	os_path_sep = r'\\\\' if os.name == 'nt' else '/' # hack because windows is terrible
	criteria = [re.sub(r'\^', os_path_sep, c) for c in criteria]
	chkpt_dirs = [d for d in chkpt_dirs if all([re.search(c, d) for c in criteria])]
	chkpt_cfg_paths = [os.path.join(chkpt_dir, '.hydra', 'config.yaml') for chkpt_dir in chkpt_dirs]
	
	# For each model, check if we've already evaluated it, and do so if we haven't already
	for chkpt_dir, chkpt_cfg_path in tuple(zip(chkpt_dirs, chkpt_cfg_paths)):
		eval_dir = os.path.join(chkpt_dir, f'eval-{cfg.data.friendly_name}')
		
		# If we haven't already evaluated the model in the directory, evaluate it
		if not (os.path.exists(eval_dir) and len([f for f in os.listdir(eval_dir) if re.search(score_file_name, f)]) == 9):
			
			chkpt_cfg = OmegaConf.load(chkpt_cfg_path)
				
			if not os.path.exists(eval_dir):
				os.mkdir(eval_dir)
			
			os.chdir(eval_dir)
				
			# Eval model
			tuner = Tuner(chkpt_cfg)
			
			if cfg.data.new_verb:
				args_cfg_path = os.path.join(chkpt_dir, 'args.yaml')
				args_cfg = OmegaConf.load(args_cfg_path)
				
				tuner.eval_new_verb(eval_cfg=cfg, args_cfg=args_cfg, checkpoint_dir=chkpt_dir)
			elif cfg.data.entail:
				tuner.eval_entailments(eval_cfg=cfg, checkpoint_dir=chkpt_dir)
			else:
				tuner.eval(eval_cfg=cfg, checkpoint_dir=chkpt_dir)
				
			# Switch back to the starting dir and copy the eval information to each individual directory
			if not eval_dir == starting_dir:
				logging.shutdown()
				os.chdir(os.path.join(starting_dir, '..'))
				copy_tree(os.path.join(starting_dir, '.hydra'), os.path.join(eval_dir, '.hydra'))
				copy_file(os.path.join(starting_dir, 'multieval.log'), os.path.join(eval_dir, 'multieval.log'))
				if os.path.exists(os.path.join(eval_dir, 'eval.log')):
					os.remove(os.path.join(eval_dir, 'eval.log'))
				
				os.rename(os.path.join(eval_dir, 'multieval.log'), os.path.join(eval_dir, 'eval.log'))
				os.remove(os.path.join(starting_dir, 'multieval.log'))
	
	if chkpt_dirs:
		log.info(f'Evaluation complete for {len(chkpt_dirs)} models')
	
	# If we are comparing the models, get the summary files and run the comparison
	if cfg.compare:
		eval_dirs = [os.path.join(chkpt_dir, f'eval-{cfg.data.friendly_name}') for chkpt_dir in chkpt_dirs]
		
		summary_files = [
			os.path.join(eval_dir, f) 
			for eval_dir in eval_dirs 
				for f in os.listdir(eval_dir)
					if re.match(score_file_name, f) and f.endswith(f'-{scores_name}.csv.gz')
		]
		
		cossims_files = [
			os.path.join(eval_dir, f)
			for eval_dir in eval_dirs
				for f in os.listdir(eval_dir)
					if re.match(score_file_name, f) and f.endswith('-cossims.csv.gz')
		]
		
		summary_of_summaries = load_summaries(summary_files)
		summary_of_cossims = load_summaries(cossims_files)
		
		multi_eval(cfg, source_dir, starting_dir, summary_of_summaries, summary_of_cossims)

def load_summaries(summary_files: List[str]) -> pd.DataFrame:
	summaries = pd.DataFrame()
	for summary_file in summary_files:
		if summary_file.endswith('.pkl.gz'):
			summary = pd.read_pickle(f)
			summaries = pd.concat([summaries, summary], ignore_index = True)
		else:
			summary = pd.read_csv(summary_file)
			summaries = pd.concat([summaries, summary], ignore_index = True)
	
	return summaries

def save_summary(save_dir: str, summary: pd.DataFrame, suffix: str = None, filetype: List[str] = ['pkl', 'csv']) -> None:
	original_dir = os.getcwd()
	
	filetype = [filetype] if isinstance(filetype, str) else filetype
	if any([f for f in filetype if not f in ['pkl', 'csv']]):
		log.warning('Invalid filetype provided. Acceptable filetypes are "csv", "pkl". Excluding invalid types.')
		filetype = [f for f in filetype if f in ['pkl', 'csv']]
			
		if not filetype:
			log.warning('No valid filetype provided. Using defaults ["pkl", "csv"].')
	
	# Get information for saved file names
	filename = summary.eval_data.unique()[0] + '-'
	epoch_label = summary.epoch_criteria.unique()[0] if len(summary.epoch_criteria.unique()) == 1 else ''
	if len(summary.model_id.unique()) == 1 and not 'multiple' in summary.total_epochs.unique():
		epoch_label = '-' + epoch_label
		try: 
			magnitude = floor(1 + np.log10(summary.total_epochs.unique()[0]))
		except:
			breakpoint()
		epoch_label = f'{str(summary.eval_epoch.unique()[0]).zfill(magnitude)}{epoch_label}'
	
	filename += epoch_label + f'-{suffix or scores_name}'
	
	os.chdir(save_dir)
	
	if 'pkl' in filetype:
		summary.to_pickle(f"{filename}.pkl.gz")
	
	if 'csv' in filetype:
		summary.to_csv(f"{filename}.csv.gz", index = False, na_rep = 'NaN')
	
	os.chdir(original_dir)

def adjust_cfg(cfg: DictConfig, source_dir: str, summary: pd.DataFrame) -> DictConfig:
	# Load the appropriate config file for model parameters,
	# or use a special file if parameters are being compared across different models
	with open_dict(cfg):
		model_name = summary['model_name'].unique()[0] if len(summary['model_name'].unique()) == 1 else 'multi'
		model_cfg_path = os.path.join(source_dir, config_path, 'model', f'{model_name}.yaml')
		cfg.model = OmegaConf.load(model_cfg_path)
		
		tuning_name = '_'.join(summary['tuning'].unique()[0].split()) if len(summary['tuning'].unique()) == 1 else 'multi'
		tuning_cfg_path = os.path.join(source_dir, config_path, 'tuning', f'{tuning_name}.yaml')
		cfg.tuning = OmegaConf.load(tuning_cfg_path)
	
	return cfg

def multi_eval(cfg: DictConfig, source_dir: str, save_dir: str, summary: pd.DataFrame, cossims: pd.DataFrame) -> None:
	log.info(f'Creating summary of cosine similarity data from {len(cossims.model_id.unique())} models')
	cossims = multi_eval_cossims(cfg, source_dir, save_dir, cossims)
	
	log.info(f'Creating summary of {re.sub("s$", "", scores_name.replace("_", " "))} data from {len(summary.model_id.unique())} models')
	if cfg.data.new_verb:
		multi_eval_new_verb(cfg, source_dir, save_dir, summary, cossims)
	elif cfg.data.entail:
		multi_eval_entailments(cfg, source_dir, save_dir, summary, cossims)
	else:
		multi_eval_(cfg, source_dir, save_dir, summary, cossims)
	
	log.info(f'Summarization of data from {len(summary.model_id.unique())} models complete')

def multi_eval_(cfg: DictConfig, source_dir: str, save_dir: str, summary: pd.DataFrame, cossims: pd.DataFrame) -> None:
	raise NotImplementedError('Comparison of non-entailment data not currently supported.')

def multi_eval_entailments(cfg: DictConfig, source_dir: str, save_dir: str, summaries: pd.DataFrame, cossims: pd.DataFrame) -> None:
	"""
	Combines entailment summaries over multiple models and plots them
	"""
	summary_of_summaries = summaries. \
		groupby(['model_id', 'eval_data', 
				 'sentence_type', 'ratio_name', 
				 'role_position', 'position_num',
				 'model_name', 'masked', 
				 'eval_epoch', 'total_epochs', 'epoch_criteria', 'min_epochs', 'max_epochs',
				 'patience', 'delta',
				 'masked_tuning_style', 'tuning', 'strip_punct']) \
		['odds_ratio']. \
		agg(['mean', 'sem']). \
		reset_index()
	
	# re-add the sentence types to the summary of summaries for plot labels
	summaries.sentence_num = pd.to_numeric(summaries.sentence_num)
	sentence_types_nums = summaries.loc[summaries.groupby(['model_id', 'sentence_type']).sentence_num.idxmin()].reset_index(drop=True)[['model_id', 'sentence_type','sentence']].rename({'sentence' : 'ex_sentence'}, axis = 1)
	summary_of_summaries = summary_of_summaries.merge(sentence_types_nums)
	
	save_summary(save_dir, summary_of_summaries)
	
	summary_of_summaries['sentence_num'] = 0
	summary_of_summaries = summary_of_summaries.rename({'ex_sentence' : 'sentence'}, axis=1)
	
	cfg = adjust_cfg(cfg, source_dir, summary_of_summaries)
	
	# Plot the overall results
	tuner = Tuner(cfg)
	log.info(f'Creating {scores_name.replace("_", " ")} plots with data from {len(summary_of_summaries.model_id.unique())} models')
	tuner.graph_entailed_results(summary_of_summaries, cfg)
	
	acc = tuner.get_entailed_accuracies(summary_of_summaries)
	save_summary(save_dir, acc, 'accuracies', 'csv')

def multi_eval_new_verb(cfg: DictConfig, source_dir: str, save_dir: str, summaries: pd.DataFrame, cossims: pd.DataFrame) -> None:
	return NotImplementedError('Comparison of new verb data not currently supported.')

def multi_eval_cossims(cfg: DictConfig, source_dir: str, save_dir: str, cossims: pd.DataFrame) -> pd.DataFrame:
	"""
	Combines and plots cosine similarity data from multiple models.
	"""
	if not (len(cossims.model_name.unique()) == 1 and cossims.model_name.unique()[0] == 'roberta'):
		roberta_cossims = cossims[(~cossims.target_group.str.endswith('most similar')) & (cossims.model_name == 'roberta')].copy()
		num_tokens_in_cossims = len(roberta_cossims.token.unique())
		
		roberta_cossims['token'] = [re.sub(chr(288), '', token) for token in roberta_cossims.token]
		num_tokens_after_change = len(roberta_cossims.token.unique())
		if num_tokens_in_cossims != num_tokens_after_change:
			# this isn't going to actually get rid of any info, but it's worth logging
			# we only check this for the target tokens, because the situation is common enough with found tokens that it's not really worth mentioning
			log.warning('RoBERTa cossim target tokens were used with and without preceding spaces. This may complicate comparing results to BERT models.')
		
		# first, replace the ones that don't start with spaces before with a preceding ^
		cossims.loc[(cossims['model_name'] == 'roberta') & ~(cossims.token.str.startswith(chr(288))), 'token'] = \
		cossims[(cossims['model_name'] == 'roberta') & ~(cossims.token.str.startswith(chr(288)))].token.str.replace(r'^(.)', r'^\1', regex=True)
		
		# then, replace the ones with the preceding special character (since we are mostly using them in the middle of sentences)
		cossims.loc[(cossims['model_name'] == 'roberta') & (cossims.token.str.startswith(chr(288))), 'token'] = \
		[re.sub(chr(288), '', token) for token in cossims[(cossims['model_name'] == 'roberta') & (cossims.token.str.startswith(chr(288)))].token]
		
		cossims = cossims.assign(
			predicted_arg = [predicted_arg.replace(chr(288), '') for predicted_arg in cossims.predicted_arg],
			target_group = [target_group.replace(chr(288), '') for target_group in cossims.target_group],
		)
	
	# this means that we have at least one cased and one uncased model, so we convert all novel words to uppercase for standardization
	if len(cossims.model_name.unique()) > 1 and 'roberta' in cossims.model_name.unique():
		cossims = cossims.assign(
			predicted_arg = [predicted_arg.upper() for predicted_arg in cossims.predicted_arg],
			target_group = [target_group.upper() if not target_group.endswith('most similar') else target_group for target_group in cossims.target_group],
		)
	
	# different models have different token ids for the same token, so we need to fix that when that happens
	for token in cossims.token.unique():
		# the second part of this conditional accounts for cases where roberta tokenizers and bert tokenizers behave differently
		# in terms of how many tokens they split a target token into. in this case, since it is a set target, 
		# we want to compare the group mean regardless, so we manually replace the token id even if it's only included in one kind of model
		if len(cossims[cossims.token == token].token_id.unique()) > 1 or any([tv for tv in ~cossims[cossims.token == token].target_group.str.endswith('most similar').unique() for target_group in cossims[cossims.token == token].target_group.unique()]):
			cossims.loc[cossims.token == token, 'token_id'] = 'multiple'
	
	# we summarize the most similar tokens and target tokens separately
	# for the most similar tokens, we want to know something the agreement 
	# in token choice across models, which means summarizing across tokens rather than models
	# for the target tokens, we want to know something about average similarity within each model, 
	# which means summarizing across models and not tokens
	multiplator = lambda x: x.unique()[0] if len(x.unique()) == 1 else 'multiple'
	
	most_similars = cossims[cossims.target_group.str.endswith('most similar')].copy()
	if not most_similars.empty:
		for token in most_similars.token.unique():
			most_similars.loc[most_similars.token == token, 'model_id'] = multiplator(most_similars.loc[most_similars.token == token, 'model_id'])
			most_similars.loc[most_similars.token == token, 'eval_epoch'] = multiplator(most_similars.loc[most_similars.token == token, 'eval_epoch'])
			most_similars.loc[most_similars.token == token, 'total_epochs'] = multiplator(most_similars.loc[most_similars.token == token, 'total_epochs'])
			most_similars.loc[most_similars.token == token, 'min_epochs'] = multiplator(most_similars.loc[most_similars.token == token, 'min_epochs'])
			most_similars.loc[most_similars.token == token, 'max_epochs'] = multiplator(most_similars.loc[most_similars.token == token, 'max_epochs'])
			most_similars.loc[most_similars.token == token, 'eval_data'] = multiplator(most_similars.loc[most_similars.token == token, 'eval_data'])
			most_similars.loc[most_similars.token == token, 'patience'] = multiplator(most_similars.loc[most_similars.token == token, 'patience'])
			most_similars.loc[most_similars.token == token, 'delta'] = multiplator(most_similars.loc[most_similars.token == token, 'delta'])
			most_similars.loc[most_similars.token == token, 'model_name'] = multiplator(most_similars.loc[most_similars.token == token, 'model_name'])
			most_similars.loc[most_similars.token == token, 'tuning'] = multiplator(most_similars.loc[most_similars.token == token, 'tuning'])
			most_similars.loc[most_similars.token == token, 'masked'] = multiplator(most_similars.loc[most_similars.token == token, 'masked'])
			most_similars.loc[most_similars.token == token, 'masked_tuning_style'] = multiplator(most_similars.loc[most_similars.token == token, 'masked_tuning_style'])
			most_similars.loc[most_similars.token == token, 'strip_punct'] = multiplator(most_similars.loc[most_similars.token == token, 'strip_punct'])
		
		most_similars = most_similars. \
			groupby([c for c in most_similars.columns if not c == 'cossim']) \
			['cossim']. \
			agg(['mean', 'sem', 'size']). \
			reset_index(). \
			sort_values(['predicted_arg', 'target_group']). \
			rename({'size' : 'num_points'}, axis=1)
	
	targets = cossims[~cossims.target_group.str.endswith('most similar')].copy()
	if not targets.empty:
		for target_group in targets.target_group.unique():
			targets.loc[targets.target_group == target_group, 'token'] = multiplator(targets.loc[targets.target_group == target_group, 'token'])
		
		targets = targets. \
			groupby([c for c in targets.columns if not c == 'cossim']) \
			['cossim']. \
			agg(['mean', 'sem', 'size']). \
			reset_index(). \
			sort_values(['predicted_arg', 'target_group']). \
			rename({'size' : 'num_points'}, axis=1)
	
	cossims = pd.concat([targets, most_similars], ignore_index=True)
	
	save_summary(save_dir, cossims, 'cossims', 'csv')
	
	cfg = adjust_cfg(cfg, source_dir, cossims)
	
	tuner = Tuner(cfg)
	log.info(f'Creating cosine similarity plots with data from {len(cossims[cossims.model_id != "multiple"].model_id.unique())} models')
	tuner.plot_cossims(cossims)


if __name__ == "__main__":
	multieval()
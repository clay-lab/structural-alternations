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

@hydra.main(config_path=config_path, config_name='multieval')
def multieval(cfg: DictConfig) -> None:
	
	print(OmegaConf.to_yaml(cfg))
	
	# Get directory information
	source_dir = hydra.utils.get_original_cwd()
	starting_dir = os.getcwd()
	
	# Get a regex for the score file name so we can just load it if it already exists
	# change the regexes to use an appropriate metric instead of 'scores'
	if cfg.epoch == 'None':
		cfg.epoch = None
		score_file_name = '(.hydra|eval.log|(' + cfg.data.friendly_name + '-(([0-9]+)-+)+(accuracies.csv.gz|tsne.csv.gz|tsne-plots.pdf|plots.pdf|scores.csv.gz|scores.pkl.gz|cossim.csv.gz)))'
		log.warning('Epoch not specified. If no evaluation has been performed, evaluation will be performed on the final epoch. Otherwise, all epochs on which evaluation has been performed will be loaded for each model.')
	elif 'best' in cfg.epoch:
		score_file_name = '(.hydra|eval.log|(' + cfg.data.friendly_name + f'-(([0-9]+)-+)+{cfg.epoch}-(accuracies.csv.gz|tsne.csv.gz|tsne-plots.pdf|plots.pdf|scores.csv.gz|scores.pkl.gz|cossim.csv.gz)))'
	else:
		score_file_name = '(.hydra|eval.log|(' + cfg.data.friendly_name + '-' + cfg.epoch + '-(accuracies.csv.gz|tsne.csv.gz|tsne-plots.pdf|plots.pdf|scores.csv.gz|scores.pkl.gz|cossim.csv.gz)))'
	
	# Get checkpoint dirs in outputs
	chkpt_dirs = os.path.join(hydra.utils.to_absolute_path(cfg.checkpoint_dir), '**')
	chkpt_dirs = [os.path.split(f)[0] for f in glob(chkpt_dirs, recursive = True) if f.endswith('weights.pkl.gz')]
	
	if not chkpt_dirs:
		print(f'No model information found in checkpoint_dir {cfg.checkpoint_dir}. Did you put in the right directory path?')
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
	
	# If we are comparing the models, get the summary files and run the comparison
	if cfg.compare:
		eval_dirs = [os.path.join(chkpt_dir, f'eval-{cfg.data.friendly_name}') for chkpt_dir in chkpt_dirs]
		
		summary_files = [
			os.path.join(eval_dir, f) 
			for eval_dir in eval_dirs 
				for f in os.listdir(eval_dir)
					if re.match(score_file_name, f) and
					   f.endswith('-scores.csv.gz')
					   # change this to a filename with the appropriate metric
		]
		
		summary_of_summaries = load_summaries(summary_files)
		log.info(f'Comparing {len(summary_of_summaries.model_id.unique())} models')
		
		if cfg.data.new_verb:
			multi_eval_new_verb(cfg, source_dir, starting_dir, summary_of_summaries)
		elif cfg.data.entail:
			multi_eval_entailments(cfg, source_dir, starting_dir, summary_of_summaries)
		else:
			multi_eval(cfg, source_dir, starting_dir, summary_of_summaries)
			
		# move this stuff so we can plot the cossims for the summaries (this requires instatiating a dummy tuner)
		similarities_files = [
			os.path.join(eval_dir, f)
			for eval_dir in eval_dirs
				for f in os.listdir(eval_dir)
					if re.match(score_file_name, f) and
					   f.endswith('-cossim.csv.gz')
		]
		
		summary_of_similarities = load_summaries(similarities_files)
		summary_of_similarities = summary_of_similarities.drop_duplicates().reset_index(drop=True)
		
		n_models = len(summary_of_similarities.model_id.unique())
		log.info(f'Comparing cosine similarity data for {n_models} models')
		
		summary_of_similarities = multi_eval_cossims(cfg, source_dir, starting_dir, summary_of_similarities)
		
		filename = summary_of_similarities.eval_data.unique()[0] + '-'
		epoch_label = summary_of_similarities.epoch_criteria.unique()[0] if len(summary_of_similarities.epoch_criteria.unique()) == 1 else ''
		if len(summary_of_similarities.model_id.unique()) == 1:
			epoch_label = '-' + epoch_label
			magnitude = floor(1 + np.log10(summary_of_similarities.total_epochs.unique()[0]))
			epoch_label = f'{str(summary_of_similarities.eval_epoch.unique()[0]).zfill(magnitude)}{epoch_label}'
		
		filename += epoch_label + '-cossim.csv.gz'
		
		summary_of_similarities.to_csv(filename, index = False, na_rep = 'NaN')
			
	# Rename the output dir if we had to escape the first character in bert
	if '^bert' in starting_dir:
		logging.shutdown()
		os.chdir('..')
		renamed = starting_dir.replace('^bert', 'bert')
		if not os.path.exists(renamed):
			os.rename(starting_dir, starting_dir.replace('^bert', 'bert'))
		else:
			copy_tree(starting_dir, renamed)
			remove_tree(starting_dir)

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

def save_summary(cfg: DictConfig, save_dir: str, summary: pd.DataFrame) -> None:
	# Get information for saved file names
	dataset_name = cfg.data.friendly_name
	
	eval_epoch = cfg.epoch if len(np.unique(summary.eval_epoch)) > 1 or np.unique(summary.eval_epoch)[0] == 'multiple' else np.unique(summary.eval_epoch)[0]
	
	os.chdir(save_dir)
	summary.to_pickle(f"{dataset_name}-{eval_epoch}-scores.pkl.gz")
	summary.to_csv(f"{dataset_name}-{eval_epoch}-scores.csv.gz", index = False)
	# save these with appropriate file names depending on the metric

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

def multi_eval(cfg: DictConfig, source_dir: str, save_dir: str, summary: pd.DataFrame) -> None:
	raise NotImplementedError('Comparison of non-entailment data not currently supported.')

def multi_eval_entailments(cfg: DictConfig, source_dir: str, save_dir: str, summaries: pd.DataFrame) -> None:
	"""
	Combines entailment summaries over multiple models and plots them
	"""
	# Summarize info here and then figure out how to plot it
	
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
	
	save_summary(cfg, save_dir, summary_of_summaries)
	
	summary_of_summaries['sentence_num'] = 0
	summary_of_summaries = summary_of_summaries.rename({'ex_sentence' : 'sentence'}, axis = 1)
	
	cfg = adjust_cfg(cfg, source_dir, summary_of_summaries)
	
	# Plot the overall results
	tuner = Tuner(cfg)
	log.info(f'Plotting results from {len(summary_of_summaries.model_id.unique())} models')
	tuner.graph_entailed_results(summary_of_summaries, cfg)
	
	acc = tuner.get_entailed_accuracies(summary_of_summaries)
	eval_epoch = cfg.epoch if len(np.unique(acc.eval_epoch)) > 1 or np.unique(acc.eval_epoch)[0] == 'multiple' else np.unique(acc.eval_epoch)[0]
	acc.to_csv(f'{cfg.data.friendly_name}-{eval_epoch}-accuracies.csv.gz', index = False)

def multi_eval_new_verb(cfg: DictConfig, source_dir: str, save_dir: str, summaries: pd.DataFrame) -> None:
	return NotImplementedError('Comparison of new verb data not currently supported.')

def multi_eval_cossims(cfg: DictConfig, source_dir: str, save_dir: str, cossims: pd.DataFrame) -> pd.DataFrame:
	cossims = cossims.copy()
	
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
	# for the most similar tokens, we want to know something the agreement in token choice across models, which means summarizing across tokens rather than models
	# for the target tokens, we want to know something about average similarity within each model, which means summarizing across models and not tokens
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
		
		# most_similars = most_similars.assign(sem = [0 if np.isnan(se) else se for se in most_similars['sem']])
	
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
	breakpoint()
	# cfg = adjust_cfg(cfg, source_dir, cossims)
	#
	# Plot the overall results
	# tuner = Tuner(cfg)
	# log.info(f'Plotting cosine similarities from {len(cossims.model_id.unique())} models')
	# tuner.plot_cossims(cossims)
	
	filename = cossims.eval_data.unique()[0] + '-'
	epoch_label = cossims.epoch_criteria.unique()[0] if len(cossims.epoch_criteria.unique()) == 1 else ''
	if len(cossims.model_id.unique()) == 1:
		epoch_label = '-' + epoch_label
		magnitude = floor(1 + np.log10(cossims.total_epochs.unique()[0]))
		epoch_label = f'{str(cossims.eval_epoch.unique()[0]).zfill(magnitude)}{epoch_label}'
	
	filename += epoch_label + '-cossim_diffs.txt'
	
	if len(cossims.predicted_arg.unique()) > 1:
		with open(filename, 'w', encoding = 'utf-8') as f:
			for predicted_arg, df in cossims.groupby('predicted_arg'):
				df = df.loc[~df.target_group.str.endswith('most similar')]
				means = df.groupby('target_group')['mean'].agg('mean')
				out_group_means = means[[i for i in means.index if not i == predicted_arg]]
				means_diffs = {f'{predicted_arg}-{arg}': means[predicted_arg] - out_group_means[arg] for arg in out_group_means.index}
				if means_diffs:
					for diff in means_diffs:
						log.info(f'Mean cossim {diff} targets for {predicted_arg} across {n_models} models: {"{:.2f}".format(means_diffs[diff])}')
						f.write(f'Mean cossim {diff} targets for {predicted_arg} across {n_models} models: {means_diffs[diff]}\n')
	
	return cossims


if __name__ == "__main__":
	multieval()
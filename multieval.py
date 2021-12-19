# multieval.py
# 
# Application entry point for evaluating and summarizing multiple masked language models.

import os
import re
import sys
import hydra
import logging

import numpy as np
import pandas as pd
import pickle as pkl

from glob import glob
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
	if cfg.epoch == 'None':
		cfg.epoch = None
		score_file_name = cfg.data.friendly_name + '-(([0-9]+)-+)+scores.pkl'
		log.warning('Epoch not specified. If no evaluation has been performed, evaluation will be performed on the final epoch. Otherwise, all epochs on which evaluation has been performed will be loaded for each model.')
	elif cfg.epoch == 'best':
		score_file_name = cfg.data.friendly_name + '-(([0-9]+)-+)+best-scores.pkl'
	else:
		score_file_name = cfg.data.friendy_name + '-' + cfg.epoch + '-scores.pkl'
	
	# Get checkpoint dirs in outputs
	chkpt_dirs = os.path.join(hydra.utils.to_absolute_path(cfg.checkpoint_dir), '**')
	chkpt_dirs = [os.path.split(f)[0] for f in glob(chkpt_dirs, recursive = True) if f.endswith('weights.pkl')]
	
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
		if not (os.path.exists(eval_dir) and any([re.search(score_file_name, f) for f in os.listdir(eval_dir)])):
			
			chkpt_cfg = OmegaConf.load(chkpt_cfg_path)
				
			if not os.path.exists(eval_dir):
				os.mkdir(eval_dir)
			
			os.chdir(eval_dir)
				
			# Eval model
			tuner = Tuner(chkpt_cfg)
			
			if cfg.data.new_verb:
				args_cfg_path = os.path.join(chkpt_dir, 'args.yaml')
				args_cfg = OmegaConf.load(args_cfg_path)
				
				tuner.eval_new_verb(
					eval_cfg = cfg,
					args_cfg = args_cfg,
					checkpoint_dir = chkpt_dir,
					epoch = cfg.epoch	
				)
			elif cfg.data.entail:
				tuner.eval_entailments(
					eval_cfg = cfg,
					checkpoint_dir = chkpt_dir,
					epoch = cfg.epoch
				)
			else:
				tuner.eval(
					eval_cfg = cfg, 
					checkpoint_dir = chkpt_dir,
					epoch = cfg.epoch
				)
				
			# Switch back to the starting dir and copy the eval information to each individual directory
			if not eval_dir == starting_dir:
				logging.shutdown()
				os.chdir(os.path.join(starting_dir, '..'))
				copy_tree(os.path.join(starting_dir, '.hydra'), os.path.join(eval_dir, '.hydra'))
				copy_file(os.path.join(starting_dir, 'multieval.log'), os.path.join(eval_dir, 'multieval.log'))
				os.rename(os.path.join(eval_dir, 'multieval.log'), os.path.join(eval_dir, 'eval.log'))
				os.remove(os.path.join(starting_dir, 'multieval.log'))
				
	
	# If we are comparing the models, get the summary files and run the comparison
	if cfg.compare:
		eval_dirs = [os.path.join(chkpt_dir, f'eval-{cfg.data.friendly_name}') for chkpt_dir in chkpt_dirs]
		
		summary_files = [
			os.path.join(eval_dir, f) 
			for eval_dir in eval_dirs 
				for f in os.listdir(eval_dir)
					if re.match(score_file_name, f)
		]
		
		summary_of_summaries = load_summaries(summary_files)
		
		log.info(f'Comparing {len(summary_of_summaries.model_id.unique())} models')
		
		if cfg.data.new_verb:
			multi_eval_new_verb(cfg, source_dir, starting_dir, summary_of_summaries)
		elif cfg.data.entail:
			multi_eval_entailments(cfg, source_dir, starting_dir, summary_of_summaries)
		else:
			multi_eval(cfg, source_dir, starting_dir, summary_of_summaries)
		
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
		with open(summary_file, 'rb') as f:
			summary = pkl.load(f)
			summaries = summaries.append(summary, ignore_index = True)
	
	return summaries

def save_summary(cfg: DictConfig, save_dir: str, summary: pd.DataFrame) -> None:
	# Get information for saved file names
	dataset_name = cfg.data.friendly_name
	if not cfg.epoch == 'best':
		all_epochs = '-'.join([str(x) for x in sorted(np.unique(summary.eval_epoch).tolist(), key = lambda x: x)])
	else:
		all_epochs = 'best'
	
	os.chdir(save_dir)
	summary.to_pickle(f"{dataset_name}-{all_epochs}-scores.pkl")
	summary.to_csv(f"{dataset_name}-{all_epochs}-scores.csv", index = False)

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
				 'eval_epoch', 'total_epochs',
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
	if cfg.epoch == 'best':
		file = [f for f in os.listdir(os.getcwd()) if re.match(cfg.data.friendly_name + r'-(([0-9]+)-+)+plots\.pdf', f)][0]
		os.rename(file, f'{cfg.data.friendly_name}-best-plots.pdf')
	
	acc = tuner.get_entailed_accuracies(summary_of_summaries)
	if not cfg.epoch == 'best':
		all_epochs = '-'.join([str(x) for x in sorted(np.unique(summary.eval_epoch).tolist(), key = lambda x: x)])
	else:
		all_epochs = 'best'
	
	acc.to_csv(f'{cfg.data.friendly_name}-{all_epochs}-accuracies.csv', index = False)

def multi_eval_new_verb(cfg: DictConfig, source_dir: str, save_dir: str, summaries: pd.DataFrame) -> None:
	return NotImplementedError('Comparison of new verb data not currently supported.')

if __name__ == "__main__":
	multieval()
# multieval.py
# 
# Application entry point for evaluating and summarizing multiple masked language models.

import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from glob import glob
import sys
from distutils.dir_util import copy_tree
import pandas as pd
import pickle as pkl
import re

from core.tuner import Tuner

source_dir = os.getcwd()
config_path = 'conf'
config_name = 'multieval'

@hydra.main(config_path=config_path, config_name=config_name)
def tune(cfg: DictConfig) -> None:
	
	starting_dir = os.path.join(os.getcwd())
	print(OmegaConf.to_yaml(cfg))
	
	# Get the score file name for the current data set to check whether we've already evaluated on it
	score_file_name = cfg.data.name.split('.')[0] + '-scores.pkl'

	# Get checkpoint dirs in outputs
	chkpt_dirs = os.path.join(hydra.utils.to_absolute_path(cfg.checkpoint_dir), '**')
	chkpt_dirs = [os.path.split(f)[0] for f in glob(chkpt_dirs, recursive = True) if f.endswith('model.pt')]
	
	if criteria == 'all': criteria = '' # do this to give us a resonable dir name
	criteria = cfg.criteria.split(',')
	os_path_sep = r'\\\\' if os.name == 'nt' else '/' # hack because windows is terrible
	criteria = [re.sub(r'\^', os_path_sep, c) for c in criteria]
	
	chkpt_dirs = [d for d in chkpt_dirs if all([re.search(c, d) for c in criteria])]
	chkpt_cfg_paths = [os.path.join(chkpt_dir, '.hydra', 'config.yaml') for chkpt_dir in chkpt_dirs]
	
	for chkpt_dir, chkpt_cfg_path in tuple(zip(chkpt_dirs, chkpt_cfg_paths)):
		
		eval_dir = os.path.join(chkpt_dir, 'eval')
		
		# If we haven't already evaluated the model in the directory, evaluate it
		if not (os.path.exists(eval_dir) and score_file_name in os.listdir(eval_dir)):
				
				chkpt_cfg = OmegaConf.load(chkpt_cfg_path)
				
				if not os.path.exists(eval_dir):
					os.mkdir(eval_dir)

				os.chdir(eval_dir)
				
				# Eval model
				tuner = Tuner(chkpt_cfg)
				if cfg.data.entail:
					tuner.eval_entailments(
						eval_cfg = cfg,
						checkpoint_dir = chkpt_dir
					)
				else:
					tuner.eval(
						eval_cfg = cfg, 
						checkpoint_dir=chkpt_dir
					)
				
				# Switch back to the starting dir and copy the eval information to each individual directory
				if not eval_dir == starting_dir:
					os.chdir(os.path.join(starting_dir, '..'))
					copy_tree(starting_dir, eval_dir)
					os.rename(os.path.join(eval_dir, 'multieval.log'), os.path.join(eval_dir, 'eval.log'))
	
	eval_dirs = [os.path.join(chkpt_dir, 'eval') for chkpt_dir in chkpt_dirs]
	summary_files = [os.path.join(eval_dir, f) for eval_dir in eval_dirs 
					 for f in os.listdir(eval_dir)
					 if f == score_file_name]
	
	eval_multi_entailments(cfg, starting_dir, summary_files)
	
def eval_multi_entailments(cfg: DictConfig, save_dir, summary_files):
	"""
	Combines entailment summaries over multiple models to plot them
	"""
	summaries = pd.DataFrame()
	for summary_file in summary_files:
		with open(summary_file, 'rb') as f:
			summary = pkl.load(f)
			summaries = summaries.append(summary, ignore_index = True)
	
	# Summarize info here and then figure out how to plot it
	summary_of_summaries = summaries. \
		groupby(['model_id', 'eval_data', 
				 'sentence_type', 'ratio_name', 
				 'role_position', 'position_num',
				 'model_name', 'masked', 
				 'masked_tuning_style', 'tuning'])['odds_ratio']. \
		agg(['mean', 'sem']). \
		reset_index()
	
	dataset_name = cfg.data.name.split('.')[0]
	
	os.chdir(save_dir)
	summary_of_summaries.to_pickle(f"{dataset_name}-scores.pkl")
	summary_of_summaries.to_csv(f"{dataset_name}-scores.csv", index = False)
	
	num_models = len(summary_of_summaries['model_id'].drop_duplicates())
	for (ratio_name, sentence_type), summary_slice in summary_of_summaries.groupby(['ratio_name', 'sentence_type']):
		means = [round(float(o_r), 2) for o_r in list(summary_slice['mean'].values)]
		sems = [round(float(s_e), 2) for s_e in list(summary_slice['sem'].values)]
		formatted = tuple(zip(means, sems))
		formatted = [f'{t[0]} ({t[1]})' for t in formatted]
		formatted = '[' + ', '.join(formatted) + ']'
		role_position = summary_slice.role_position.unique()[0].replace('_' , ' ')
		print(f'\nMean log odds and standard error of {ratio_name} in {role_position} in {sentence_type}s across {num_models} models:\n\t{formatted}')
	
	print('')
	
	# Load the appropriate config file for model parameters,
	# or use a special file if parameters are being compared across different models
	with open_dict(cfg):
		if len(model_name := summary_of_summaries['model_name'].unique()) == 1:
			model_name = model_name[0]
			model_cfg_path = os.path.join(source_dir, config_path, 'model', f'{model_name}.yaml')
		else:
			model_cfg_path = os.path.join(source_dir, config_path, 'model', 'multi.yaml')
		
		model_cfg = OmegaConf.load(model_cfg_path)
		cfg.model = model_cfg
		
		if len(tuning_name := summary_of_summaries['tuning'].unique()) == 1:
			tuning_name = tuning_name[0]
			tuning_cfg_path = os.path.join(source_dir, config_path, 'tuning', f'{tuning_name}.yaml')
		else:
			tuning_cfg_path = os.path.join(source_dir, config_path, 'tuning', 'multi.yaml')
			
		tuning_cfg = OmegaConf.load(tuning_cfg_path)
		cfg.tuning = tuning_cfg
		
	tuner = Tuner(cfg)
	
	tuner.graph_entailed_results(summary_of_summaries, cfg, multi = True)

if __name__ == "__main__":
	tune()
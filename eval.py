# eval.py
# 
# Application entry point for evaluating and summarizing masked language models.
import os
import re
import time
import hydra
import shutil
import logging

import pandas as pd

from glob import glob
from typing import *
from operator import and_
from functools import reduce
from omegaconf import DictConfig, OmegaConf
from distutils.dir_util import copy_tree
from distutils.file_util import copy_file

from core.tuner import Tuner
from core import tuner_utils
from core import tuner_plots

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
	'dirname', 
	lambda criteria, dataname: criteria.replace(',', '-') + '-' + dataname.split('.')[0]
)

EXPECTED_NUMBER_OF_RESULTS_FILES = {
	'newarg' 	:  9,
	'newverb'	: 18,
}

@hydra.main(config_path='conf', config_name='eval')
def evaluate(cfg: DictConfig) -> None:
	'''
	Evaluates model checkpoints according to the passed config.
	
		params:
			cfg (DictConfig): a DictConfig specifying evaluation parameters.
							  Explanation and defaults can be found in ./conf/eval.yaml.
	'''
	def reset_log_file() -> None:
		'''
		Closes and deletes the log file. 
		Used after an individual model is evaluated to obtain a clean log for the next model.
		'''
		logging.shutdown()
		os.remove('eval.log')
	
	def get_score_file_regex(name: str, epoch: Union[int,str], exp_type: str) -> str:
		'''
		Get the appropriate regex for the files containing eval results.
		
			params:
				name (str)				: the name of the evaluation data being used
				epoch (int,str)			: the epoch where the models are being evaluated
				exp_type (str)			: the type of experiment being evaluated
			
			returns:
				score_file_regex (str)	: a regex used to count the number of eval files in a directory
										  useful to check whether a model has already been evaluated at the current settings.
		'''
		# set up scores file criteria
		if epoch == 'None':
			epoch = None
			expr = '(([0-9]+)-+)+'
			log.warning('Epoch not specified. If no evaluation has been performed, evaluation will be performed on the final epoch. Otherwise, all epochs on which evaluation has been performed will be loaded for each model.')
		elif isinstance(epoch,str) and 'best' in epoch:
			expr = f'(([0-9]+)-+)+{epoch}'
		else:
			expr = epoch
		
		return rf'(\.hydra|eval\.log|({name.split(".")[0]}-{expr}-(accuracies(_diffs)?(_sentences)?\.csv\.gz|tsnes\.csv\.gz|tsne-plots\.pdf|{scores_name}(_diffs)?(_sentences)?-plots\.pdf|{scores_name}(_sentences)?\.csv\.gz|cossims\.csv\.gz|cossims-plots\.pdf|predictions\.csv\.gz|target_counts\.json\.gz|kl_divs\.csv\.gz|kl_divs-hist\.pdf)))'
	
	def get_checkpoint_dirs(d: str, criteria: str) -> List[str]:
		'''
		Finds all subdirectories of d containing model checkpoints that can be evaluated, filtered by criteria.
		
			params:
				d (str)						: the directory whose subdirectories to search for model checkpoints.
				criteria (str)				: a single string formatted as a comma-separated list of strings.
											  a directory will only be included in the returned list if all strings in
											  criteria are found in its full path.
			
			returns:
				checkpoint_dirs (List[str])	: a list of subdirectories of d containing model checkpoints
		'''
		# Get checkpoint dirs in outputs
		checkpoint_dirs_weights = os.path.join(hydra.utils.to_absolute_path(d), '**/weights.pkl.gz')
		checkpoint_dirs_models 	= os.path.join(hydra.utils.to_absolute_path(d), '**/model.pt')
		checkpoint_dirs 		= list(set([os.path.split(p)[0] for p in glob(checkpoint_dirs_weights, recursive=True) + glob(checkpoint_dirs_models, recursive=True)]))
		checkpoint_dirs 		= [d for d in checkpoint_dirs if 'metrics.csv.gz' in os.listdir(d)]
		if not checkpoint_dirs:
			raise ValueError(f'No model information found in "{d}". Did you put in the right directory path?')
		
		# filter paths based on criteria
		criteria = criteria.split(',')
		criteria = [''] if criteria == ['all'] else criteria # if criteria is 'all', don't filter out anything
		os_path_sep = r'\\\\' if os.name == 'nt' else '/' # windows bad >:(
		criteria = [re.sub(r'\^', os_path_sep, c) for c in criteria]
		checkpoint_dirs = sorted([d for d in checkpoint_dirs if all([re.search(c, d) for c in criteria])])
		
		return checkpoint_dirs
	
	def create_and_change_to_eval_dir(checkpoint_dir: str, eval_dir_name: str) -> str:
		'''
		Creates and changes to a model's evaluation directory.
		
			params:
				checkpoint_dir (str)	: the directory containing the model checkpoint
				eval_dir_name (str)		: the name of the evaluation directory to create (without 'eval-' prepended)
		'''
		eval_dir = os.path.join(checkpoint_dir, f'eval-{eval_dir_name}')
		if not os.path.exists(eval_dir):
			os.mkdir(eval_dir)
			
		os.chdir(eval_dir)
		return eval_dir
	
	def copy_config_logs(multieval_dir: str, eval_dir: str) -> None:
		'''
		Copies hydra config files and logs from the main evaluation directory to the individual model's eval directory.
		
			params:
				multieval_dir (str)	: the source directory containing the config and log files
				eval_dir (str)		: the destination directory to move files to
		'''
		if multieval_dir != eval_dir:
		# Switch back to the starting dir and copy the eval information to each individual directory
			if os.path.exists(os.path.join(eval_dir, 'eval.log')):
				os.remove(os.path.join(eval_dir, 'eval.log'))
			
			# exit the directory so we can copy it over
			logging.shutdown()
			os.chdir(os.path.join(multieval_dir, '..'))
			copy_tree(os.path.join(multieval_dir, '.hydra'), os.path.join(eval_dir, '.hydra'))
			copy_file(os.path.join(multieval_dir, 'eval.log'), os.path.join(eval_dir, 'eval.log'))
			os.remove(os.path.join(multieval_dir, 'eval.log'))
			os.chdir(multieval_dir)
	
	def get_dir_name(data: DictConfig, comparison_masking: str) -> str:
		'''
		Gets a formatted directory name for saving results.
		
			params:
				data (dictconfig)		: a dict config containing the name of the eval dataset
				comparison_masking (str): a str specifying how kl divergence is to be calculated 
										  (i.e., how to mask tokens)
			
			returns:
				dir_name (str)			: a directory name to store the results in
		'''
		dir_name = data.split('.')[0]
		
		if comparison_masking:
			dir_name += '-kl' + comparison_masking[0] + 'mask'
		
		return dir_name
	
	# make sure to clean out the log file if we are rerunning in the same dir
	reset_log_file()
	
	print(OmegaConf.to_yaml(cfg, resolve=True))
	
	# Get directory information to use for moving stuff around later
	source_dir 			= hydra.utils.get_original_cwd()
	multieval_dir 		= os.getcwd()
	
	# Get a regex for the score file name so we can just load it if it already exists
	# make this global so we can access it in other functions
	global scores_name
	scores_name			= 'odds_ratios' if cfg.data.exp_type in ['newverb', 'newarg'] else 'scores'
	score_file_regex 	= get_score_file_regex(cfg.data.name, cfg.epoch, cfg.data.exp_type)
	num_expected_files 	= EXPECTED_NUMBER_OF_RESULTS_FILES[cfg.data.exp_type]
	if not cfg.create_plots:
		num_expected_files -= (4 if cfg.data.exp_type == 'newverb' else 3)
	
	checkpoint_dirs 	= get_checkpoint_dirs(cfg.dir, cfg.criteria)
	
	try:
		for i, checkpoint_dir in enumerate(checkpoint_dirs):
			success 	= False
			eval_dir 	= create_and_change_to_eval_dir(checkpoint_dir, get_dir_name(cfg.data.name, cfg.comparison_masking))
			if len([f for f in os.listdir(eval_dir) if re.search(score_file_regex, f)]) < num_expected_files or cfg.rerun:
				tuner 	= Tuner(checkpoint_dir, use_gpu=cfg.use_gpu)
				tuner.evaluate(eval_cfg=cfg)
				copy_config_logs(multieval_dir, eval_dir)
			success		= True
	except KeyboardInterrupt:
		log.warning('Multieval was stopped manually!')
		cfg.summarize 	= False
	
	log.info(f'Evaluation complete for {i if not success else i + 1} models')
	os.chdir(multieval_dir)
	
	if cfg.summarize and len(checkpoint_dirs) > 1:
		summarize(cfg, checkpoint_dirs)
	else:
		os.chdir('..')
		logging.shutdown()
		# need to add a tiny cooldown here to avoid stepping on the OS's toes
		time.sleep(0.5)
		shutil.rmtree(multieval_dir)

def save_summary( 
	summary: pd.DataFrame, 
	suffix: str = None,
	filetypes: List[str] = ['pkl', 'csv']
) -> None:
	'''
	Saves a summary dataframe to disk.
	
		params:
			summary (pd.DataFrame)	: the summary dataframe to save
			suffix (str)			: added to the end of the summary file name
			filetypes (List[str])	: what filetype to save the summary as
	'''
	func_map = dict(
		pkl=lambda df, f: df.to_pickle(f),
		csv=lambda df, f: df.to_csv(f, **{'index': False, 'na_rep': 'NaN'})
	)
	
	filetypes = [filetypes] if isinstance(filetypes, str) else filetypes
	
	if any([f for f in filetypes if not f in ['pkl', 'csv']]):
		log.warning('Invalid filetype provided. Acceptable filetypes are "csv", "pkl". Excluding invalid types.')
		filetypes = [f for f in filetypes if f in ['pkl', 'csv']]
	
	if not filetypes:
		log.warning('No valid filetype provided. Using defaults ["pkl", "csv"].')
		filetypes = ['pkl', 'csv']
	
	# Get information for saved file names
	filename = f'{tuner_utils.get_file_prefix(summary)}-{suffix or scores_name}'
	
	for filetype in filetypes:
		func_map[filetype](summary, f'{filename}.{filetype}.gz')

def summarize(
	cfg: DictConfig,
	checkpoint_dirs: List[str]
) -> None:
	'''
	Loads and combines summaries and passed them to summarize_cossims and summarize_odds_ratios.
	
		params:
			cfg (DictConfig)			: a DictConfig specifying the evaluation parameters.
			checkpoint_dirs (List[str])	: a list of directories containing csvs with cosine similarity and odds ratios data.
	'''
	def find_summaries(checkpoint_dirs: str) -> List[str]:
		eval_dirs 				= [os.path.join(checkpoint_dir, f) for checkpoint_dir in checkpoint_dirs for f in os.listdir(checkpoint_dir) if f.startswith(f'eval-{cfg.data.name.split(".")[0]}')]
		summary_files			= [os.path.join(eval_dir,f) for eval_dir in eval_dirs for f in os.listdir(eval_dir) if f.endswith(f'-{scores_name}.csv.gz')]
		sentences_summary_files	= [os.path.join(eval_dir,f) for eval_dir in eval_dirs for f in os.listdir(eval_dir) if f.endswith(f'-{scores_name}_sentences.csv.gz')]
		cossims_files			= [os.path.join(eval_dir,f) for eval_dir in eval_dirs for f in os.listdir(eval_dir) if f.endswith('-cossims.csv.gz')]
		
		return summary_files, sentences_summary_files, cossims_files
	
	log.info('Loading results files')
	summary_files, sentences_summary_files, cossims_files	= find_summaries(checkpoint_dirs)
	summaries 												= tuner_utils.load_csvs(summary_files)
	
	if sentences_summary_files:
		sentences_summaries 								= tuner_utils.load_csvs(sentences_summary_files)
	else:
		sentences_summaries 								= pd.DataFrame()
	
	cossims 												= tuner_utils.load_csvs(cossims_files, converters={'token': str})
	
	log.info(f'Creating summary of cosine similarity data from {len(cossims_files)} models')
	summarize_cossims(cfg, cossims)
	
	assert cfg.data.exp_type in ['newverb', 'newarg'], f'Currently, multieval only supports comparing data for newverb and newarg experiments.'
	
	log.info(f'Creating summary of {scores_name.replace("_", " ")} data from {len(summary_files)} models')
	summarize_odds_ratios(cfg, summaries)
	
	if not sentences_summaries.empty:
		log.info(f'Creating summary of {scores_name.replace("_", " ")} data for sentences from {len(sentences_summary_files)} models')
		summarize_odds_ratios(cfg, sentences_summaries)
	
	log.info(f'Summarization of data from {summaries.model_id.unique().size} models complete')

def summarize_odds_ratios(
	cfg: DictConfig,
	summaries: pd.DataFrame
) -> None:
	'''
	Combines entailment summaries over multiple models, and outputs a summary of the summaries and accuracies, as well as plots
		
		params:
			cfg (Dict)					: a config file containing information about the experiments evaluated. passed to other functions
			summaries (pd.DataFrame)	: a dataframe concatenating results from several models to summarize
	'''
	excluded_cols = ['sentence_num', 'sentence', 'odds_ratio', 'log_probability', 'other_log_probability']
	agg_kwargs = dict(
		odds_ratio_mean = ('odds_ratio', 'mean'), 
		odds_ratio_sem 	= ('odds_ratio', 'sem')
	)
	
	if cfg.data.exp_type == 'newverb':
		excluded_cols.extend([
			'token_id', 'token', 'token_type', 'odds_ratio_pre_post_difference', 
			'full_ratio_name'
		])
		agg_kwargs.update(dict(
			odds_ratio_pre_post_difference_mean = ('odds_ratio_pre_post_difference', 'mean'),
			odds_ratio_pre_post_difference_sem	= ('odds_ratio_pre_post_difference', 'sem')
		))
		# for the token summaries, where we have info about the individual positions
		# instead of the overall mean of a whole sentence
		if 'position_ratio_name' in summaries.columns:
			agg_kwargs.update(dict(	
				log_probability_mean 				= ('log_probability', 'mean'),
				log_probability_sem 				= ('log_probability', 'sem'),
				other_log_probability_mean 			= ('other_log_probability', 'mean'),
				other_log_probability_sem 			= ('other_log_probability', 'sem'),
			))
	
	included_cols = [c for c in summaries.columns if not c in excluded_cols]
	
	summary_of_summaries = summaries. \
		groupby(included_cols, dropna=False). \
		agg(**agg_kwargs). \
		reset_index()
	
	if cfg.data.exp_type == 'newverb':
		if 'token_type' in summaries.columns:
			for model_id in summary_of_summaries.model_id.unique():
				summary_of_summaries.loc[summary_of_summaries.model_id == model_id, 'token_type'] = tuner_utils.multiplator(summaries.loc[summaries.model_id == model_id, 'token_type'])
		
	# re-add an example of each sentence type to the summary of summaries for plot labels
	summaries.sentence_num 	= summaries.sentence_num.astype(int)
	sentence_examples 		= summaries.loc[summaries.groupby(['model_id','random_seed','sentence_type']).sentence_num.idxmin()]
	sentence_examples 		= sentence_examples[['model_id','random_seed','sentence_type','sentence']]
	sentence_examples 		= sentence_examples.rename(dict(sentence='ex_sentence'), axis=1)
	summary_of_summaries 	= summary_of_summaries.merge(sentence_examples)
	
	save_summary(summary_of_summaries, filetypes=['pkl', 'csv'])
	
	# add/change these back for plotting purposes
	summary_of_summaries['sentence_num'] 	= 0
	summary_of_summaries 					= summary_of_summaries.rename({'ex_sentence' : 'sentence'}, axis=1)
	
	if 'token' in summaries.columns and 'token_id' in summaries.columns:
		summary_of_summaries['token'] 			= tuner_utils.multiplator(summaries.token, multstr='any')
		summary_of_summaries['token_id']		= tuner_utils.multiplator(summaries.token_id)
	
	n_models = len(summary_of_summaries[['model_id', 'random_seed']].drop_duplicates())
	
	# Plot the overall results
	if cfg.data.exp_type == 'newverb' and cfg.create_plots:
		if 'position_ratio_name' in summary_of_summaries.columns:
			log.info(f'Creating {scores_name.replace("_", " ")} differences plots with data from {n_models} models')
			tuner_plots.create_odds_ratios_plots(summary_of_summaries, cfg, plot_diffs=True)
		else:
			log.info(f'Creating {scores_name.replace("_", " ")} differences plots for sentences with data from {n_models} models')
			tuner_plots.create_odds_ratios_plots(summary_of_summaries, cfg, plot_diffs=True, suffix='sentences')
	
	if cfg.create_plots:
		if 'position_ratio_name' in summary_of_summaries.columns:
			log.info(f'Creating {scores_name.replace("_", " ")} plots with data from {n_models} models')
			tuner_plots.create_odds_ratios_plots(summary_of_summaries, cfg)
		else:
			log.info(f'Creating {scores_name.replace("_", " ")} plots for sentences with data from {n_models} models')
			tuner_plots.create_odds_ratios_plots(summary_of_summaries, cfg, suffix='sentences')
	
	acc = tuner_utils.get_odds_ratios_accuracies(summary_of_summaries, cfg)
	acc = tuner_utils.transfer_hyperparameters_to_df(summary_of_summaries, acc)
	save_summary(acc, 'accuracies' if 'position_ratio_name' in summary_of_summaries.columns else 'accuracies_sentences', 'csv')

def summarize_cossims(cfg: DictConfig, cossims: pd.DataFrame) -> None:
	'''
	Combines and plots cosine similarity data from multiple models
	
		params:
			cfg (Dict)				: a config file containing information about the experiments evaluated. passed to other functions
			cossims (pd.DataFrame)	: a dataframe combining cosine similarity results from >1 model to summarize
	'''
	agg_kwargs = dict(
		cossim_mean = ('cossim', 'mean'),
		cossim_sem 	= ('cossim', 'sem'),
		num_points 	= ('cossim', 'size')
	)
	
	groups = [c for c in cossims.columns if not c == 'cossim']
	
	# we summarize the topk most similar tokens and target tokens separately
	# for the most similar tokens, we want to know about the *agreement* 
	# in token choice across models, which means summarizing across token selections rather than model behavior
	topk = cossims[cossims.target_group.str.endswith('most similar')].copy()
	
	if not topk.empty:
		model_token_cols 			= [c for c in topk.columns if not c in ['eval_epoch','token','predicted_arg','cossim']]
		correction_kwargs_cols 		= [c for c in cossims.columns if c.startswith('correction_')]
		all_cols 					= ['eval_epoch','token','predicted_arg','correction'] + correction_kwargs_cols
		duplicated_token_arg_pairs 	= [tuple(pair) for pair in topk[topk[all_cols].duplicated()][all_cols].to_numpy()]
		
		for eval_epoch, token, predicted_arg, correction, *correction_kwargs in duplicated_token_arg_pairs:
			cols_values = tuple(zip([c for c in cossims.columns if c.startswith('correction_')], correction_kwargs))
			condition = reduce(
				and_, 
				[
					topk.eval_epoch == eval_epoch,
					topk.token == token, 
					topk.predicted_arg == predicted_arg,
					topk.correction == correction,
				] +
				[topk[col] == value for col, value in cols_values]
			)
			
			topk.loc[condition, model_token_cols] = (
				topk.loc[condition, model_token_cols]
					.apply(
						lambda col: tuner_utils.multiplator(col), 
						result_type='broadcast'
					)
				)
	
	# for the target tokens, we want to know something about the average between
	# tokens' and their targets' similarity within each model, 
	# which means summarizing model behavior and not token selection
	targets = cossims[~cossims.target_group.str.endswith('most similar')].copy()
	
	if not targets.empty:
		model_token_cols = ['token', 'token_id']
		
		for target_group in targets.target_group.unique():
			targets.loc[targets.target_group == target_group, model_token_cols] = \
			targets.loc[targets.target_group == target_group, model_token_cols].apply(lambda col: tuner_utils.multiplator(col), result_type='broadcast')
	
	cossims = pd.concat([
		df.groupby(groups, dropna=False) \
			.agg(**agg_kwargs) \
			.reset_index() \
			.sort_values(['eval_epoch','predicted_arg','target_group'])
		for df in (topk, targets) if not df.empty
	], ignore_index=True)
	
	save_summary(cossims, 'cossims', 'csv')
	# we can only create cosine similarity plots for target group tokens, and only if there is more than one argument we are comparing
	if (
		any(~cossims.target_group.str.endswith('most similar')) 
		# and not len(cossims[~cossims.target_group.str.endswith('most similar')].predicted_arg.unique()) <= 1
		and cfg.create_plots
	):
		n_models = len(cossims[(cossims.model_id != 'multiple') & (cossims.random_seed != 'multiple')][['model_id', 'random_seed']].drop_duplicates())
		
		log.info(f'Creating cosine similarity plots with data from {n_models} models')
		tuner_plots.create_cossims_plot(cossims)

if __name__ == '__main__':
	
	evaluate()

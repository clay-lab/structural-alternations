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

EXPECTED_NUMBER_OF_RESULT_FILES = {
	'newarg' 	: 9,
	'newverb'	:13,
}

@hydra.main(config_path='conf', config_name='eval')
def evaluate(cfg: DictConfig) -> None:
	
	def reset_log_file():
		logging.shutdown()
		os.remove('eval.log')
	
	def get_score_file_regex(name: str, epoch: Union[int,str], exp_type: str) -> str:
		# set up scores file criteria
		if epoch == 'None':
			epoch = None
			expr = '(([0-9]+)-+)+'
			log.warning('Epoch not specified. If no evaluation has been performed, evaluation will be performed on the final epoch. Otherwise, all epochs on which evaluation has been performed will be loaded for each model.')
		elif isinstance(epoch,str) and 'best' in epoch:
			expr = f'(([0-9]+)-+)+{epoch}'
		else:
			expr = epoch
		
		return rf'(\.hydra|eval\.log|({name.split(".")[0]}-{expr}-(accuracies(_diffs)?\.csv\.gz|tsnes\.csv\.gz|tsne-plots\.pdf|{scores_name}(_diffs)?-plots\.pdf|{scores_name}.(csv|pkl)\.gz|cossims\.csv\.gz|kl_divs\.csv\.gz|kl_divs-hist\.pdf)))'
	
	def get_checkpoint_dirs(d: str, criteria: str) -> List[str]:
		# Get checkpoint dirs in outputs
		checkpoint_dirs_weights = os.path.join(hydra.utils.to_absolute_path(d), '**/weights.pkl.gz')
		checkpoint_dirs_models 	= os.path.join(hydra.utils.to_absolute_path(d), '**/model.pt')
		checkpoint_dirs 		= list(set([os.path.split(p)[0] for p in glob(checkpoint_dirs_weights, recursive=True) + glob(checkpoint_dirs_models, recursive=True)]))
		
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
		eval_dir = os.path.join(checkpoint_dir, f'eval-{eval_dir_name}')
		if not os.path.exists(eval_dir):
			os.mkdir(eval_dir)
		
		os.chdir(eval_dir)
		return eval_dir
	
	def copy_config_logs(multieval_dir: str, eval_dir: str) -> None:
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
	num_expected_files 	= EXPECTED_NUMBER_OF_RESULT_FILES[cfg.data.exp_type]
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
	
	def find_summaries(checkpoint_dirs: str) -> List[str]:
		eval_dirs 	= [os.path.join(checkpoint_dir, f) for checkpoint_dir in checkpoint_dirs for f in os.listdir(checkpoint_dir) if f.startswith(f'eval-{cfg.data.name.split(".")[0]}')]
		summary_files	= [os.path.join(eval_dir,f) for eval_dir in eval_dirs for f in os.listdir(eval_dir) if f.endswith(f'-{scores_name}.pkl.gz')]
		cossims_files	= [os.path.join(eval_dir,f) for eval_dir in eval_dirs for f in os.listdir(eval_dir) if f.endswith('-cossims.csv.gz')]
		
		return summary_files, cossims_files
	
	log.info('Loading results files')
	summary_files, cossims_files	= find_summaries(checkpoint_dirs)
	summaries 						= tuner_utils.load_pkls(summary_files)
	cossims 						= tuner_utils.load_csvs(cossims_files, converters={'token': str})
	
	log.info(f'Creating summary of cosine similarity data from {len(cossims_files)} models')
	summarize_cossims(cfg, cossims)
	
	assert cfg.data.exp_type in ['newverb', 'newarg'], f'Currently, multieval only supports comparing data for newverb and newarg experiments.'
	
	log.info(f'Creating summary of {scores_name.replace("_", " ")} data from {len(summary_files)} models')
	summarize_odds_ratios(cfg, summaries)
	
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
	excluded_cols = ['sentence_num', 'sentence', 'odds_ratio']
	agg_kwargs = dict(
		odds_ratio_mean = ('odds_ratio', 'mean'), 
		odds_ratio_sem 	= ('odds_ratio', 'sem')
	)
	
	if cfg.data.exp_type == 'newverb':
		excluded_cols.extend(['token_id', 'token', 'token_type', 'odds_ratio_pre_post_difference'])
		agg_kwargs.update(dict(
			odds_ratio_pre_post_difference_mean = ('odds_ratio_pre_post_difference', 'mean'),
			odds_ratio_pre_post_difference_sem	= ('odds_ratio_pre_post_difference', 'sem')
		))
	
	included_cols = [c for c in summaries.columns if not c in excluded_cols]
	
	summary_of_summaries = summaries. \
		groupby(included_cols, dropna=False). \
		agg(**agg_kwargs). \
		reset_index()
	
	if cfg.data.exp_type == 'newverb':
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
	summary_of_summaries['token'] 		= tuner_utils.multiplator(summaries.token, multstr='any')
	summary_of_summaries['token_id']	= tuner_utils.multiplator(summaries.token_id)
	
	n_models = len(summary_of_summaries[['model_id', 'random_seed']].drop_duplicates())
	
	# Plot the overall results
	if cfg.data.exp_type == 'newverb' and cfg.create_plots:
		log.info(f'Creating {scores_name.replace("_", " ")} differences plots with data from {n_models} models')
		tuner_plots.create_odds_ratios_plots(summary_of_summaries, cfg, plot_diffs=True)
	
	if cfg.create_plots:
		log.info(f'Creating {scores_name.replace("_", " ")} plots with data from {n_models} models')
		tuner_plots.create_odds_ratios_plots(summary_of_summaries, cfg)
	
	acc = tuner_utils.get_odds_ratios_accuracies(summary_of_summaries, cfg)
	acc = tuner_utils.transfer_hyperparameters_to_df(summary_of_summaries, acc)
	save_summary(acc, 'accuracies', 'csv')

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
		
		model_token_cols 			= [c for c in topk.columns if not c in ['token','predicted_arg','cossim']]
		duplicated_token_arg_pairs 	= [tuple(pair) for pair in topk[topk[['token','predicted_arg']].duplicated()][['token','predicted_arg']].to_numpy()]
		
		for token, predicted_arg in duplicated_token_arg_pairs:
			topk.loc[(topk.token == token) & (topk.predicted_arg == predicted_arg), model_token_cols] = \
			topk.loc[(topk.token == token) & (topk.predicted_arg == predicted_arg), model_token_cols].apply(lambda col: tuner_utils.multiplator(col), result_type='broadcast')
	
	# for the target tokens, we want to know something about the average between
	# tokens' and their targets' similarity within each model, which means summarizing model behavior and not token selection
	targets = cossims[~cossims.target_group.str.endswith('most similar')].copy()
	
	if not targets.empty:
		
		model_token_cols = ['token','token_id']
		
		for target_group in targets.target_group.unique():
			targets.loc[targets.target_group == target_group, model_token_cols] = \
			targets.loc[targets.target_group == target_group, model_token_cols].apply(lambda col: tuner_utils.multiplator(col), result_type='broadcast')
	
	cossims = pd.concat([
		df.groupby(groups,dropna=False) \
			.agg(**agg_kwargs) \
			.reset_index() \
			.sort_values(['predicted_arg','target_group'])
		for df in (topk, targets) if not df.empty
	], ignore_index=True)
	
	save_summary(cossims, 'cossims', 'csv')
	# we can only create cosine similarity plots for target group tokens, and only if there is more than one argument we are comparing
	if (
		any(~cossims.target_group.str.endswith('most similar')) 
		and not len(cossims[~cossims.target_group.str.endswith('most similar')].predicted_arg.unique()) <= 1
		and cfg.create_plots
	):
		n_models = len(cossims[(cossims.model_id != 'multiple') & (cossims.random_seed != 'multiple')][['model_id', 'random_seed']].drop_duplicates())
		
		log.info(f'Creating cosine similarity plots with data from {n_models} models')
		tuner_plots.create_cossims_plot(cossims)


if __name__ == '__main__':
	
	evaluate()

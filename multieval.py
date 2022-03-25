# multieval.py
# 
# Application entry point for evaluating and summarizing multiple masked language models.
import os
import re
import hydra
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

OmegaConf.register_new_resolver('criterianame', lambda criteria: criteria.replace(',', '-'))

@hydra.main(config_path='conf', config_name='multieval')
def multieval(cfg: DictConfig) -> None:
	
	def reset_log_file():
		logging.shutdown()
		os.remove('multieval.log')
	
	def get_score_file_name(name: str, epoch: Union[int,str] = None, exp_type: str):
		# set up scores file criteria
		if epoch == 'None':
			epoch = None
			expr = '(([0-9]+)-+)+'
			log.warning('Epoch not specified. If no evaluation has been performed, evaluation will be performed on the final epoch. Otherwise, all epochs on which evaluation has been performed will be loaded for each model.')
		elif isinstance(epoch,str) and 'best' in epoch:
			expr = f'(([0-9]+)-+)+{epoch}'
		else:
			expr = epoch
		
		return f'(.hydra|eval.log|({name}-{expr}-(accuracies.csv.gz|tsne.csv.gz|tsne-plots.pdf|{scores_name}(_diffs)?-plots.pdf|{scores_name}.(csv|pkl).gz|cossims.csv.gz)))'
	
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
		checkpoint_dirs = [d for d in checkpoint_dirs if all([re.search(c, d) for c in criteria])]
		
		return checkpoint_dirs
	
	def create_and_change_to_eval_dir(checkpoint_dir: str) -> None:
		eval_dir = os.path.join(checkpoint_dir, f'eval-{cfg.data.friendly_name}')
		if not os.path.exists(eval_dir):
			os.mkdir(eval_dir)
		
		os.chdir(eval_dir)
	
	def copy_config_logs(starting_dir: str, eval_dir: str) -> None:
		if starting_dir != eval_dir:
		# Switch back to the starting dir and copy the eval information to each individual directory
			if os.path.exists(os.path.join(eval_dir, 'eval.log')):
				os.remove(os.path.join(eval_dir, 'eval.log'))
			
			logging.shutdown()
			os.chdir(os.path.join(starting_dir, '..')) # exit the directory so we can copy it over
			copy_tree(os.path.join(starting_dir, '.hydra'), 		os.path.join(eval_dir, '.hydra'))
			copy_file(os.path.join(starting_dir, 'multieval.log'), 	os.path.join(eval_dir, 'multieval.log'))
			
			os.rename(os.path.join(eval_dir, 'multieval.log'), 		os.path.join(eval_dir, 'eval.log'))
			
			os.remove(os.path.join(starting_dir, 'multieval.log'))
			os.chdir(starting_dir)
	
	# if we are rerunning the script after stopping, we want to clean out the log file
	reset_log_file()
	
	print(OmegaConf.to_yaml(cfg, resolve=True))
	
	# Get directory information to use for moving stuff around later
	source_dir 			= hydra.utils.get_original_cwd(), '..'
	starting_dir 		= os.getcwd()

	# Get a regex for the score file name so we can just load it if it already exists
	# make this global so we can access it later
	global scores_name
	scores_name			= 'odds_ratios' if cfg.data.exp_type in ['newverb', 'newarg'] else 'scores'
	score_file_name 	= get_score_file_name(cfg.data.friendly_name, cfg.epoch)
	num_expected_files 	= 9 if self.exp_type == 'newarg' else 11 if self.exp_type == 'newverb' else 1
	checkpoint_dirs 	= get_checkpoint_dirs()
	
	for checkpoint_dir in checkpoint_dirs:
		create_and_change_to_eval_dir(checkpoint_dir)
		# If we haven't already evaluated the model in the directory, evaluate it
		if len([f for f in os.listdir(eval_dir) if re.search(score_file_name, f)]) < expected_num_files:
			tuner = Tuner(checkpoint_dir)
			tuner.evaluate(eval_cfg=cfg, checkpoint_dir=checkpoint_dir)
			copy_config_logs(starting_dir, eval_dir)
	
	log.info(f'Evaluation complete for {len(checkpoint_dirs)} models')
	
	if cfg.summarize:
		summarize(cfg, checkpoint_dirs)

def load_summaries(summary_files: List[str]) -> pd.DataFrame:
	pkl_files = [f for f in summary_files if f.endswith('.pkl.gz')]
	csv_files = [f for f in summary_files if f.endswith('.csv.gz')]
	
	pkl_summaries = tuner_utils.load_pkls(pkl_files)
	csv_summaries = tuner_utils.load_csvs(csv_files, converters={'token': str})
	
	summaries = pd.concat([pkl_summaries, csv_summaries], ignore_index=True)
	summaries = summaries.sort_values('model_id').reset_index(drop=True)
	
	return summaries

def save_summary(
	save_dir: str, 
	summary: pd.DataFrame, 
	suffix: str = None,
	filetypes: List[str] = ['pkl', 'csv']
) -> None:
	
	func_map = {
		'pkl': lambda f: pd.DataFrame.to_pickle(f)
		'csv': lambda f: pd.DataFrame.to_csv(f, index=False, na_rep='NaN')
	}

	original_dir = os.getcwd()
	
	filetypes = [filetype] if isinstance(filetype, str) else filetype
	
	if any([f for f in filetypes if not f in ['pkl', 'csv']]):
		log.warning('Invalid filetype provided. Acceptable filetypes are "csv", "pkl". Excluding invalid types.')
		filetypes = [f for f in filetypes if f in ['pkl', 'csv']]
	
	if not filetypes:
		log.warning('No valid filetype provided. Using defaults ["pkl", "csv"].')
		filetypes = ['pkl', 'csv']
	
	# Get information for saved file names
	filename = f'{tuner_utils.get_file_prefix(summary)}-{suffix or scores_name}'
	
	os.chdir(save_dir)
	
	for filetype in filetypes:
		func_map[filetype](f'{filename}.{filetype}.gz')
	
	os.chdir(original_dir)

def summarize(
	cfg: DictConfig,
	checkpoint_dirs: List[str]
) -> None:

	def find_summaries(checkpoint_dirs: str) -> List[str]:
		eval_dirs = [os.path.join(checkpoint_dir, f'eval-{cfg.data.friendly_name}') for checkpoint_dir in checkpoint_dirs]
	
		glob_eval_dirs 			= [eval_dir + f'/*-{scores_name}.pkl.gz' for eval_dir in eval_dirs]
		glob_cossim_dirs		= [eval_dir + f'/*-cossims.csv.gz' for eval_dir in eval_dirs]
	
		summary_files 			= glob(glob_eval_dirs)
		cossims_files 			= glob(glob_cossim_dirs)
	
		return summary_files, cossims_files
	
	summary_files, cossims_files = find_summaries(checkpoint_dirs)
	
	summary_of_summaries 	= load_summaries(summary_files)
	summary_of_cossims 		= load_summaries(cossims_files)

	log.info(f'Creating summary of cosine similarity data from {cossims.model_id.unique().size} models')
	summarize_cossims(cfg, source_dir, save_dir, cossims)
	
	log.info(f'Creating summary of {scores_name.replace("_", " ")} data from {summary.model_id.unique().size} models')
	if cfg.data.exp_type in ['newverb', 'newarg']:
		summarize_odds_ratios(cfg, source_dir, save_dir, summary)
	else:
		raise NotImplementedError('Currently, multieval only supports comparing data for newverb and newarg experiments.')
	
	log.info(f'Summarization of data from {summary.model_id.unique().size} models complete')

def summarize_odds_ratios(
	cfg: DictConfig, 
	source_dir: str, 
	save_dir: str, 
	summaries: pd.DataFrame
) -> None:
	'''
	Combines entailment summaries over multiple models and plots them
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
		groupby(included_cols). \
		agg(**agg_kwargs). \
		reset_index()
	
	if cfg.data.exp_type == 'newverb':
		for model_id in summary_of_summaries.model_id.unique():
			summary_of_summaries.loc[summary_of_summaries.model_id == model_id]['token_type'] = tuner_utils.multiplator(summaries.loc[summaries.model_id == model_id].token_type)
	
	# re-add an example of each sentence type to the summary of summaries for plot labels
	summaries.sentence_num 	= summaries.sentence_num.astype(int)
	sentence_examples 		= summaries.loc[summaries.groupby(['model_id','sentence_type']).sentence_num.idxmin()]
	sentence_examples 		= sentence_examples[['model_id','sentence_type','sentence']]
	sentence_examples 		= sentence_examples.rename(dict(sentence='ex_sentence'), axis=1)
	summary_of_summaries 	= summary_of_summaries.merge(sentence_types_nums)
	
	save_summary(save_dir, summary_of_summaries)
	
	# change these back for plotting purposes
	summary_of_summaries['sentence_num'] 	= 0
	summary_of_summaries 					= summary_of_summaries.rename({'ex_sentence' : 'sentence'}, axis=1)
	
	n_models = summary_of_summaries.model_id.unique().size
	
	# Plot the overall results
	if cfg.data.exp_type == 'newverb':
		log.info(f'Creating {scores_name.replace("_", " ")} differences plots with data from {n_models} models')
		tuner_plots.create_odds_ratios_plots(summary_of_summaries, cfg, plot_diffs=True)
	
	log.info(f'Creating {scores_name.replace("_", " ")} plots with data from {n_models} models')
	tuner_plots.create_odds_ratios_plots(summary_of_summaries, cfg)
	
	acc = tuner_utils.get_odds_ratios_accuracies(summary_of_summaries)
	save_summary(save_dir, acc, 'accuracies', 'csv')

def summarize_cossims(
	cfg: DictConfig, 
	source_dir: str, 
	save_dir: str, 
	cossims: pd.DataFrame
) -> None:
	'''
	Combines and plots cosine similarity data from multiple models.
	'''
	'''
	def format_tokens_ids_for_comparisons(cossims: pd.DataFrame) -> pd.DataFrame:
		if not (len(cossims.model_name.unique()) == 1 and cossims.model_name.unique()[0] == 'roberta'):
			roberta_cossims = cossims[~(cossims.target_group.str.endswith('most similar')) & (cossims.model_name == 'roberta')].copy()
			num_tokens_in_cossims = len(roberta_cossims.token.unique())
			
			roberta_cossims['token'] = [re.sub(chr(288), '', token) for token in roberta_cossims.token]
			num_tokens_after_change = len(roberta_cossims.token.unique())
			if num_tokens_in_cossims != num_tokens_after_change:
				# this isn't going to actually get rid of any info, but it's worth logging
				# we only check this for the target tokens, because the situation is common enough with found tokens that it's not really worth mentioning
				log.warning('RoBERTa cossim target tokens were used with and without preceding spaces. This may complicate comparing results to BERT models.')
			
			# first, replace the ones that don't start with spaces before with a preceding ^
			cossims.loc[(cossims.model_name == 'roberta') & ~(cossims.token.str.startswith(chr(288))), 'token'] = \
				cossims[(cossims.model_name == 'roberta') & ~(cossims.token.str.startswith(chr(288)))].token.str.replace(r'^(.)', r'^\1', regex=True)
			
			# then, replace the ones with the preceding special character (since we are mostly using them in the middle of sentences)
			cossims.loc[(cossims.model_name == 'roberta') & (cossims.token.str.startswith(chr(288))), 'token'] = \
			[re.sub(chr(288), '', token) for token in cossims[(cossims.model_name == 'roberta') & (cossims.token.str.startswith(chr(288)))].token]
			
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
				cossims.loc[cossims.token == token].token_id = 'multiple'
		
		return cossims
	'''
	agg_kwargs = dict(
		cossim_mean = ('cossim', 'mean'),
		cossim_sem 	= ('cossim', 'sem'),
		num_points 	= ('cossim', 'size')
	)
	
	groups = [c for c in most_similars.columns if not c == 'cossim']
	
	# cossims = format_tokens_ids_for_comparisons(cossims)
	
	# we summarize the most similar tokens and target tokens separately
	# for the most similar tokens, we want to know something the agreement 
	# in token choice across models, which means summarizing across tokens rather than models
	
	# for the target tokens, we want to know something about average similarity within each model, 
	# which means summarizing across models and not tokens
	most_similars = cossims[cossims.target_group.str.endswith('most similar')].copy()
	if not most_similars.empty:
		for token in most_similars.token.unique():
			for predicted_arg in most_similars[most_similars.token == token].predicted_arg.unique():
				for column in [c for c in most_similars.columns if not c in ['token', 'predicted_arg']]:
					most_similars.loc[(most_similars.token == token) & (most_similars.predicted_arg == predicted_arg), column] = \
						tuner_utils.multiplator(most_similars.loc[(most_similars.token == token) & (most_similars.predicted_arg == predicted_arg), column])
		
		most_similars = most_similars. \
			groupby(groups) \
			agg(**agg_kwargs). \
			reset_index(). \
			sort_values(['predicted_arg', 'target_group'])
	
	targets = cossims[~cossims.target_group.str.endswith('most similar')].copy()
	if not targets.empty:
		for target_group in targets.target_group.unique():
			targets.loc[targets.target_group == target_group].token = tuner_utils.multiplator(targets.loc[targets.target_group == target_group].token)
		
		targets = targets. \
			groupby(groups) \
			agg(**agg_kwargs). \
			reset_index(). \
			sort_values(['predicted_arg', 'target_group'])
	
	cossims = pd.concat([targets, most_similars], ignore_index=True)
	
	save_summary(save_dir, cossims, 'cossims', 'csv')
	
	n_models = cossims[cossims.model_id != 'multiple'].model_id.unique().size
	
	# we can only create cosine similarity plots for target group tokens, and only if there is more than one argument we are comparing
	if any(~cossims.target_group.str.endswith('most similar')) and not len(cossims.predicted_arg.unique()) <= 1:
		log.info(f'Creating cosine similarity plots with data from {n_models} models')
		tuner_plots.create_cossims_plot(cossims)


if __name__ == '__main__':
	
	multieval()
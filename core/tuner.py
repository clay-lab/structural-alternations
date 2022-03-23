# tuner.py
# 
# Tunes a model on training data and provides functions for evaluation
import os
import re
import gzip
import hydra
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import logging
import itertools

import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import torch.nn as nn

from math import floor
from copy import deepcopy
from tqdm import trange, tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import *
from omegaconf import DictConfig, OmegaConf, open_dict, ListConfig
from transformers import logging as lg
from transformers import AutoModelForMaskedLM
from sklearn.manifold import TSNE

from . import tuner_plots
from . import tuner_utils
from .tuner_utils import none

lg.set_verbosity_error()

log = logging.getLogger(__name__)

class Tuner:

	# START Computed Properties
	
	@property
	def mixed_tuning_data(self) -> Dict:
		'''Returns a dict with roberta-style masked sentences, inputs, and masked token indices.'''
		return self.__get_formatted_datasets()[self.tuning]
	
	@property
	def word_embeddings(self) -> nn.parameter.Parameter:
		'''Returns the model's word embedding weights'''
		return getattr(self.model, self.model_name).embeddings.word_embeddings.weight
	
	@property
	def added_token_weights(self) -> Dict[str,torch.Tensor]:
		'''Returns the weights of the added token(s)'''
		added_token_weights = {}
		for token in self.tokens_to_mask:
			token_id = self.tokenizer.convert_tokens_to_ids(token)
			assert token_id != self.unk_token_id, f'Added token {token} was not added correctly!'
			added_token_weights[token] = self.word_embeddings[token_id,:].clone()
		
		return added_token_weights
	
	# END Computed Properties
	
	# START Private Functions
	
	def __log_debug_predictions(self, epoch: int, total_epochs: int) -> None:
		'''
		Prints a log message used during debugging. Currently only usable with newverb experiments.
		
			params:
				epoch (int)			: which epoch the model is at
				total_epochs (int)	: the total number of epochs the model was trained for, or max_epochs
		'''
		log.info('')
		self.predict_sentences(
			info = f'epoch {str(epoch).zfill(len(str(total_epochs)))}', 
			sentences = [
				 'The local [MASK] will step in to help.',
				 'The [MASK] will blork the [MASK].',
				f'The {self.cfg.tuning.args["[subj]"][0]} will [MASK] the {self.cfg.tuning.args["obj"][0]}.',
				 'The [MASK] will [MASK] the [MASK].',
			], 
			output_fun=log.info
		)
	
	def __format_strings_with_tokens_for_display(self, data: 'any') -> 'any':
		
		return tuner_utils.format_strings_with_tokens_for_display(data, self.tokens_to_mask, self.model_name, self.string_id)
	
	def __format_data_for_tokenizer(self, data: str) -> List[str]:
		
		def format_string_for_tokenizer(s: str) -> str:
			s = s.lower() if 'uncased' in self.string_id else s
			s = tuner_utils.strip_punct(s) if self.strip_punct else s
			s = s.replace(self.mask_token.lower(), self.mask_token)
			return s

		return tuner_utils.apply_to_all_of_type(data, str, format_string_for_tokenizer)
		
	def __get_formatted_datasets(
		self, 
		mask_args: bool = False, 
		masking_style: str = None, 
		datasets: Union[Dict, DictConfig] = None
	) -> Dict:
		'''
		Returns a dictionary with formatted inputs, labels, and mask_token_indices (if they exist) for datasets
		
			params:
				mask_args (bool)		: whether to mask arguments only useful in newverb experiments
				masking_style (str)		: 'always' to mask all tokens in [self.tokens_to_mask] (+ arguments if mask_args)
									  	  'none' to return unmasked data
				datasets (Dict-like)	: which datasets to generated formatted data for
			
			returns:
				formatted_data (dict)	: a dict with, for each dataset, sentences, inputs, (+ masked_token_indices if masking_style != 'none')
		'''
		to_mask = self.tokenizer.convert_tokens_to_ids(self.tokens_to_mask)
		
		if datasets is None:
			datasets = {self.tuning: {'data': OmegaConf.to_container(self.cfg.tuning.data)}}
			if mask_args:
				datasets[self.tuning].update({'data': self.verb_tuning_data})
		
		if (not np.isnan(self.mask_args) and self.mask_args) or mask_args:
			to_mask += list(self.cfg.tuning.args.keys())
		
		# this is so we don't overwrite the original datasets as we do this
		datasets = deepcopy(datasets)
		
		# we need to convert everything to primitive types to feed them to the tokenizer
		if isinstance(datasets, DictConfig):
			datasets = OmegaConf.to_container(datasets)
		
		formatted_data = {}
		for dataset in datasets:
			inputs, labels, masked_token_indices = self.create_inputs(sentences=datasets[dataset]['data'], to_mask=to_mask, masking_style=masking_style)
			formatted_data.update({dataset: {'sentences': datasets[dataset]['data'], 'inputs': inputs, 'labels': labels, 'masked_token_indices': masked_token_indices}})
		
		return formatted_data
	
	def __collect_results(
		self, outputs: 'MaskedLMOutput',
		masked_token_indices: List[Dict[str,int]],
		eval_groups: Dict = None,
	) -> List:
		'''
		Returns a list of dicts with results based on model outputs
			
			params:
				outputs (MaskedLMOutput) 	: model outputs
				masked_token_indices (list)	: list of dicts with mappings from string token to integer positions 
										  	  of the masked token locations corresponding to that token 
										  	  for each sentence in the outputs
				eval_groups (dict)			: dict mapping a token group to a list of tokens to evaluate
			
			returns:
				results (list)				: list of dicts with results for each token for each sentence
		'''
		def get_output_metrics(outputs: 'MaskedLMOutput') -> Tuple:
			logits = outputs.logits
			probabilities = nn.functional.softmax(logits, dim=-1)
			log_probabilities = nn.functional.log_softmax(logits, dim=-1)
			surprisals = -(1/torch.log(torch.tensor(2.))) * nn.functional.log_softmax(logits, dim=-1)
			predicted_ids = torch.argmax(log_probabilities, dim=-1)
			
			return logits, probabilities, log_probabilities, surprisals, predicted_ids
		
		results = []
		
		metrics = tuple(zip(masked_token_indices, self.tuning_data['sentences'], *get_output_metrics(outputs)))
		
		if eval_groups is None:
			eval_groups = self.tokens_to_mask
		
		if isinstance(eval_groups,list):
			eval_groups = {token: [token] for token in eval_groups}
		
		for arg_type in eval_groups:
			for arg in eval_groups[arg_type]:
				for sentence_num, (arg_indices, sentence, logits, probs, logprobs, surprisals, predicted_ids) in enumerate(metrics):
					if arg_type in arg_indices:
						arg_token_id = self.tokenizer.convert_tokens_to_ids(arg)
						if arg_token_id == self.unk_token_id:
							raise ValueError(f'Argument "{arg}" was not tokenized correctly! Try using something different instead.')
						
						exp_logprob = logprobs[arg_indices[arg_type],arg_token_id]
						
						common_args = {
							'arg type'			: arg_type,
							'token id'			: arg_token_id,
							'token'				: arg,
							'sentence'			: sentence,
							'sentence num'		: sentence_num,
							'predicted sentence': self.tokenizer.decode(predicted_ids),
							'predicted ids'		: ' '.join([str(i.item()) for i in predicted_ids]),
							'logit'				: logits[arg_indices[arg_type],arg_token_id],
							'probability'		: probs[arg_indices[arg_type],arg_token_id],
							'log probability'	: exp_logprob,
							'surprisal'			: surprisals[arg_indices[arg_type],arg_token_id],
						}
						
						if self.exp_type == 'newverb':
							common_args.update({'args group': self.cfg.tuning.which_args})
						
						other_positions = [(arg_position, arg_index) for arg_position, arg_index in arg_indices.items() if not arg_position == arg_type]
						
						if other_positions:
							for arg_position, arg_index in other_positions:
								logprob = logprobs[arg_index,arg_token_id]
								odds_ratio = exp_logprob - logprob
							
							results.append({
								'odds ratio': odds_ratio,
								'ratio name': arg_type + '/' + arg_position,
								**common_args
							})
						else:
							results.append({**common_args})
		
		return results
	
	def __add_hyperparameters_to_summary_df(self, df: pd.DataFrame) -> pd.DataFrame:
		'''
		Adds hyperparameters to a summary dataframe.
			
			params:
				df (pd.DataFrame): a dataframe to add hyperparameters to
			
			returns:
				df (pd.DataFrame): the dataframe with hyperparameters added in columns
		'''
		exclude = ['mask_token', 'mask_token_id', 'save_full_model', 'checkpoint_dir']
		
		included_vars = [var for var in vars(self) if not var in exclude]
		included_vars = [var for var in included_vars if type(vars(self)[var]) in (str,int,float,bool,np.nan)]
		sorted_vars = sorted([var for var in included_vars], key=lambda item: re.sub(r'^(model)', '0\\1', item))
		
		for var in sorted_vars:
			df[var] = vars(self)[var]
		
		return df
	
	def __restore_original_random_seed(self) -> None:
		'''Restores the original random seed used to generate weights for the novel tokens'''
		if hasattr(self, 'seed'):
			return
		
		for f in ['tune.log', 'weights.pkl.gz']:
			path = os.path.join(self.checkpoint_dir, f)
			if not os.path.isfile(path):
				path = f'{os.path.sep}..{os.path.sep}'.join(os.path.split(path))
		
			try:
				if f == 'tune.log':
					with open(path, 'r') as logfile_stream:
						logfile = logfile_stream.read()
					
					self.random_seed = int(re.findall(r'Seed set to ([0-9]*)\n', logfile)[0])
				elif f == 'weights.pkl.gz':
					with gzip.open(path, 'rb') as weightsfile_stream:
						weights = pkl.load(weightsfile_stream)
			
					self.random_seed = weights['random_seed']
			except (IndexError, FileNotFoundError):
				pass
		
		log.error(f'Seed not found in log file or weights file in {os.path.split(path)[0]}!')
		return
	
	def __eval(self, eval_cfg: DictConfig) -> None:
		'''
		Evaluates a model without any fine-tuning
			
			params: eval_cfg (DictConfig): evaluation config options
		'''
		self.model.eval()
		
		inputs = self.load_eval_file(eval_cfg)['inputs']
		
		with torch.no_grad():	
			log.info('Evaluating model on testing data')
			outputs = self.model(**inputs)
		
		results = self.__collect_results(inputs, eval_cfg.data.eval_groups, outputs)
		summary = self.summarize_results(results)
		
		log.info('Creating aconf and entropy plots')
		tuner_plots.graph_results(summary, eval_cfg)
	
	def __evaluate_newtoken_experiment(self, eval_cfg: DictConfig) -> None:
		'''
		Computes model performance on data using odds ratios metrics.
		Used with newverb and newarg experiments
		
			params: eval_cfg (DictConfig): evaluation config options
		'''
		def add_odds_ratios_differences_to_summary(summary: pd.DataFrame) -> pd.DataFrame:
			'''
			Convert the pre- and post-tuning results into a pandas.DataFrame
			that contains the post- minus pre-fine-tuning difference in the odds ratios
			for each sentence/argument
			
				params:
					summary (pd.DataFrame):	dataframe with odds ratios for epoch 0 and a distinct eval epoch
				
				returns:
					summary (pd.DataFrame): dataframe with difference between odds ratios @ epoch 0 and @ eval epoch added
			'''
			summary_zero = summary[summary.eval_epoch == 0].reset_index(drop=True)
			summary_eval = summary[summary.eval_epoch != 0]
			
			if not summary_eval.empty:
				summary_eval = summary_eval.reset_index(drop=True)
				assert all(summary_eval[[c for c in summary_eval.columns if not c in ['odds_ratio', 'eval_epoch']]] == summary_zero[[c for c in summary_zero.columns if not c in ['odds_ratio', 'eval_epoch']]]), \
					'Pre- and post-fine-tuning results do not match!'
					
				summary_eval['odds_ratio_pre_post_difference'] = summary_eval.odds_ratio - summary_zero.odds_ratio
				summary = summary_eval
			else:
				summary_zero['odds_ratio_pre_post_difference'] = np.nan
				summary = summary_zero
			
			return summary
		
		self.model.eval()
		
		data = self.load_eval_file(eval_cfg)
		
		summary = self.get_odds_ratios_summary(epoch=eval_cfg.epoch, eval_cfg=eval_cfg, data=data)
		
		if eval_cfg.data.exp_type == 'newverb':
			summary_zero = self.get_odds_ratios_summary(epoch=0, eval_cfg=eval_cfg, data=data)
			summary = pd.concat([summary_zero, summary], ignore_index=True)
			summary = add_odds_ratios_differences_to_summary(summary, eval_cfg)
		
		file_prefix = tuner_utils.get_file_prefix(summary)
		
		log.info(f'SAVING TO: {os.getcwd().replace(hydra.utils.get_original_cwd(), "")}')
		summary.to_pickle(f'{file_prefix}-odds_ratios.pkl.gz')
		
		# tensors are saved as text in csv, but we want to save them as numbers
		summary_csv = summary.copy()
		for c in ['odds_ratio', 'odds_ratio_pre_post_difference']:
			if c in summary_csv.columns:
				summary_csv[c] = summary_csv[c].astype(float)
		
		summary_csv.to_csv(f'{file_prefix}-odds_ratios.csv.gz', index=False, na_rep='NaN')
		
		cossims_args = dict(topk=eval_cfg.k)
		if eval_cfg.data.exp_type == 'newarg':
			cossims_args.update(dict(targets=eval_cfg.data.masked_token_targets))
		
		cossims = self.get_cossims(**cossims_args)
		cossims = tuner_utils.transfer_hyperparameters_to_df(summary, cossims)
		predicted_roles = {v: k for k, v in self.__format_data_for_tokenizer(eval_cfg.data.eval_groups).items()}
		cossims['predicted_role'] = [predicted_roles[arg.replace(chr(288), '')] for arg in cossims.predicted_arg]
		
		if 'masked_token_target_labels' in eval_cfg.data:
			target_group_labels = self.__format_data_for_tokenizer(eval_cfg.data.masked_token_target_labels)
			cossims['target_group_label'] = [
				target_group_labels[group.replace(chr(288), '')] 
				if not group.endswith('most similar') and group.replace(chr(288), '') in target_group_labels 
				else group for group in cossims.target_group
			]
			
		if not cossims[~cossims.target_group.str.endswith('most similar')].empty:
			log.info('Creating cosine similarity plots')
			self.create_cossims_plot(cossims)
		
		cossims.to_csv(f'{file_prefix}-cossims.csv.gz', index=False, na_rep='NaN')
		
		log.info('Creating t-SNE plot(s)')
		tsne_args = dict(n=eval_cfg.num_tsne_words)
		if 'masked_token_targets' in eval_cfg.data:
			tsne_args.update(dict(targets=eval_cfg.data.masked_token_targets))
		
		if 'masked_token_target_labels' in eval_cfg.data:
			tsne_args.update(dict(target_group_labels=target_group_labels))
		
		tsnes = self.get_tsnes(**tsne_args)
		tsnes = tuner_utils.transfer_hyperparameters_to_df(summary, tsnes)
		self.create_tsnes_plots(tsnes)
		
		if eval_cfg.data.exp_type == 'newverb':
			log.info('Creating odds ratios differences plots')
			self.create_odds_ratios_plots(summary, eval_cfg, plot_diffs=True)
		
		log.info('Creating odds ratios plots')
		self.create_odds_ratios_plots(summary, eval_cfg)
		
		if eval_cfg.data.exp_type == 'newverb':
			acc = self.get_odds_ratios_accuracies(summary, eval_cfg, get_diffs_accuracies=True)
			acc.to_csv(f'{file_prefix}-accuracies_diffs.csv.gz', index=False, na_rep='NaN')
		
		acc = self.get_odds_ratios_accuracies(summary, eval_cfg)
		acc = tuner_utils.transfer_hyperparameters_to_df(summary, acc)
		acc.to_csv(f'{file_prefix}-accuracies.csv.gz', index=False, na_rep='NaN')
		
		log.info('Evaluation complete')
		print('')
	
	# END Private Functions
	
	def __init__(self, cfg_or_path: Union[DictConfig,str]) -> 'Tuner':
		'''
		Creates a tuner object, loads argument/dev sets, and sets class attributes
		
			params:
				cfg_or_path (DictConfig or str): if dictconfig, a dictconfig specifying a Tuner configuration
												 if str, a directory created by a Tuner when tune() is run
			
			returns:
				the created Tuner object
		'''
		def load_args() -> None:
			'''Loads correct argument set for newverb experiments'''
			if self.cfg.tuning.exp_type == 'newverb':
				with open_dict(self.cfg):
					self.cfg.tuning.args = self.cfg.tuning[self.model_name] if self.cfg.tuning.which_args == 'model' else self.cfg.tuning[self.cfg.tuning.which_args]
		
		def load_dev_sets() -> None:
			'''Loads dev sets using specified criteria'''
			if self.cfg.dev == 'best_matches':
				criteria = self.cfg.tuning.name.split('_')
				candidates = os.listdir(os.path.join(hydra.utils.get_original_cwd(), 'conf', 'tuning'))
				candidates = [candidate.replace('.yaml', '').split('_') for candidate in candidates]
				
				# Find all the tuning sets that differ from the current one by one parameter, and grab those as our best matches
				candidates = [candidate for candidate in candidates if len(set(criteria) - set(candidate)) == 1 and candidate[0] == criteria[0]]
				
				# additionally filter out any manually excluded best_matches
				if 'dev_exclude' in self.cfg:
					self.cfg.dev_exclude = tuner_utils.listify(self.cfg.dev_exclude)
					for exclusion in self.cfg.dev_exclude:
						candidates = [candidate for candidate in candidates if not exclusion in candidate]
				
				# join them together
				candidates = ['_'.join(candidate) for candidate in candidates]
				self.cfg.dev = candidates
			
			dev_sets = tuner_utils.listify(self.cfg.dev)
			self.cfg.dev = {}
			
			with open_dict(self.cfg):
				for dev_set in dev_sets:
					dev = OmegaConf.load(os.path.join(hydra.utils.get_original_cwd(), 'conf', 'tuning', dev_set + '.yaml'))
					if not all(token in self.cfg.tuning.to_mask for token in dev.to_mask):
						log.warn(f'Not all dev tokens to mask from {dev_set} are in the training set! This is probably not what you intended. Removing this dataset from the dev data.')
					
					self.cfg.dev.update({dev.name: dev})
		
		def setattrs() -> None:
			'''Sets static model attributes'''
			
			log.info(f'Initializing Model:\t{self.cfg.model.base_class} ({self.cfg.model.string_id})')
			self.model 								= AutoModelForMaskedLM.from_pretrained(self.cfg.model.string_id, **self.cfg.model.model_kwargs)
			
			resolved_cfg = OmegaConf.to_container(self.cfg, resolve=True)
			for k, v in resolved_cfg['hyperparameters'].items():
				
				setattr(self, k, v)
			
			if not isinstance(self.unfreezing,int):
				unfreezing_epochs_per_layer 		= re.findall(r'[0-9]+', self.unfreezing)
				unfreezing_epochs_per_layer			= int(unfreezing_epochs_per_layer[0]) if unfreezing_epochs_per_layer else 1
				self.unfreezing 					= re.sub(r'[0-9]*', '', self.unfreezing) if not self.unfreezing == 'none' else np.nan
				self.unfreezing_epochs_per_layer 	= self.unfreezing_epochs_per_layer if self.unfreezing == 'gradual' else np.nan
			
			self.model_name 						= self.model.config.model_type
			self.model_id 							= os.path.split(self.checkpoint_dir)[-1] + '-' + self.model_name[0]
			self.string_id 							= self.model.config.name_or_path
			
			# this is temporary so we can format the new tokens according to the model specifications
			# the formatting functions fixes mask tokens that are converted to lower case, so it needs something to refer to
			# this is redefined a little ways down immediately after initalizing the tokenizer
			self.mask_token 						= ''
			
			tokens 									= self.__format_data_for_tokenizer(self.cfg.tuning.to_mask)
			if self.model_name == 'roberta':
				tokens = tuner_utils.format_roberta_tokens_for_tokenizer(tokens)
			else:
				# we use ^ in display-facing notation to signify tokens without a preceding space
				# these are not used in bert and distilbert, so we exclude them
				tokens = [token for token in tokens if not token.startswith('^')]
			
			self.tokens_to_mask						= tokens
			
			log.info(f'Initializing Tokenizer:\t{self.cfg.model.tokenizer} ({self.cfg.model.string_id})')
			self.tokenizer 							= tuner_utils.create_tokenizer_with_added_tokens(self.cfg.model.string_id, self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs)
			self.model.resize_token_embeddings(len(self.tokenizer))
			
			self.mask_token 						= self.tokenizer.mask_token
			self.mask_token_id 						= self.tokenizer.convert_tokens_to_ids(self.mask_token)
			self.unk_token_id 						= self.tokenizer.convert_tokens_to_ids([self.tokenizer.unk_token_id])
			
			self.tuning 							= self.cfg.tuning.name
			self.exp_type 							= self.cfg.tuning.exp_type
			self.mask_args							= self.mask_args if self.exp_type == 'newverb' else np.nan
			self.reference_sentence_type 			= self.cfg.tuning.reference_sentence_type
			self.masked 							= self.masked_tuning_style != 'none' 
			
			self.tuning_data 						= self.__get_formatted_datasets(masking_style='none')[self.tuning]
			self.masked_tuning_data 				= self.__get_formatted_datasets(masking_style='always')[self.tuning]
			self.dev_data 							= self.__get_formatted_datasets(masking_style='none', datasets=self.cfg.dev)
			self.masked_dev_data 					= self.__get_formatted_datasets(masking_style='always', datasets=self.cfg.dev)
			
			if self.exp_type == 'newverb':
				to_replace = self.cfg.tuning.args
				
				args, values = zip(*to_replace.items())
				replacement_combinations = itertools.product(*list(to_replace.values()))
				to_replace_dicts = [dict(zip(args, t)) for t in replacement_combinations]
				
				sentences = []
				for d in to_replace_dicts:
					for sentence in self.tuning_data:
						for arg, value in d.items():
							sentence = sentence.replace(arg, value)
						
						sentences.append(sentence)
				
				self.verb_tuning_data 				= sentences
				self.masked_dev_argument_data 		= self.__get_formatted_datasets(mask_args=True, masking_style='always', datasets=self.cfg.dev)
				self.args_group 					= self.cfg.tuning.which_args
				
		self.cfg = OmegaConf.load(os.path.join(cfg_or_path, '.hydra', 'config.yaml')) if isinstance(cfg_or_path, str) else cfg_or_path
		self.checkpoint_dir = cfg_or_path if isinstance(cfg_or_path, str) else os.getcwd()
		self.save_full_model = False
		
		load_dev_sets()
		load_args()
		setattrs()
	
	def __repr__(self) -> str:
		'''Return a string that eval() can be called on to create an identical Tuner object'''
		return 'tuner.Tuner(' + repr(self.cfg) + ')'
	
	def __str__(self) -> str:
		'''Return a formatted string for printing'''
		return f'Tuner object @ {self.checkpoint_dir} with config:\n' + OmegaConf.to_yaml(self.cfg, resolve=True)
	
	def __call__(self, *args, **kwargs):
		'''Calls predict sentences to generate model predictions'''
		return self.predict_sentences(*args, **kwargs)
	
	
	def tune(self) -> None:
		'''
		Fine-tunes the model on the provided tuning data. 
		Saves updated weights/model state, metrics, and plots of metrics to disk.
		'''
		def unfreeze_all_params() -> None:
			'''Unfreezes all model parameters, ensures full model is saved to disk'''
			self.save_full_model = True
			log.warning(f'You are using {self.unfreezing} unfreezing, which requires saving the full model state instead of just the weights of the new tokens.')
			log.warning('Only the initial model state and the state with the lowest mean dev loss will be retained and available for evaluation.')
			
			for name, param in self.model.named_parameters():
				param.requires_grad = True
				assert param.requires_grad, f'{name} is frozen!'
		
		def freeze_to_layer(n: int = None) -> None:
			'''
			Freezes model layers up to n
			
				params:
					n (int): if positive, the highest layer to freeze
							 if negative, freezes to model.num_hidden_layers - n
			'''
			if n is None:
				n = self.model.config.num_hidden_layers
			
			if abs(n) > self.model.config.num_hidden_layers:
				log.warning(f'You are trying to freeze to hidden layer {n}, but the model only has {self.model.config.num_hidden_layers}. Freezing all hidden layers.')
				n = self.model.config.num_hidden_layers
			
			# allow specifying from the end of the model
			n = self.model.config.num_hidden_layers + n if n < 0 else n
			
			layers_to_freeze = [f'layer.{x}.' for x in range(n)]
			layers_to_unfreeze = [f'layer.{x}.' for x in range(n,self.model.config.num_hidden_layers)]
			
			if any(layers_to_unfreeze):
				if not self.save_full_model:
					self.save_full_model = True
					log.warning(f'You are using {"layer " + str(self.unfreezing) if isinstance(self.unfreezing,int) else self.unfreezing} unfreezing, which requires saving the full model state instead of just the weights of the new tokens.')
					log.warning('Only the initial model state and the state with the lowest mean dev loss will be retained and available for evaluation.')
			
			# this is so we only print the log message if we are actually changing parameters
			if (
				any(	param.requires_grad for name, param in self.model.named_parameters() if any(layer in name for layer in layers_to_freeze  )) or
				any(not param.requires_grad for name, param in self.model.named_parameters() if any(layer in name for layer in layers_to_unfreeze))
			): 
				if len(layers_to_freeze) == self.model.config.num_hidden_layers:
					log.info('Freezing model parameters')
				else:	
					log.info(f'Freezing model parameters to layer {n}')
			
			for name, param in self.model.named_parameters():
				# always freeze everything except the word embeddings and the layers, and also freeze the specified layers
				if ('word_embeddings' not in name and 'layer' not in name) or ('layer' in name and any(layer in name for layer in layers_to_freeze)):
					param.requires_grad = False
					assert not param.requires_grad, f'{name} is not frozen!'
				else:
					param.requires_grad = True
					assert param.requires_grad, f'{name} is frozen!'
		
		def initialize_added_token_weights() -> None:
			'''Initializes the token weights to random values to provide variability in model tuning, saves random seed'''
			with torch.no_grad():
				model_embedding_weights = self.word_embeddings
				model_embedding_dim = self.word_embeddings.shape[-1]
				num_new_tokens = len(self.tokens_to_mask)
				new_embeds = nn.Embedding(num_new_tokens, model_embedding_dim)
				
				std, mean = torch.std_mean(model_embedding_weights)
				log.info(f'Initializing new token(s) with random data drawn from N({mean:.2f}, {std:.2f})')
				
				# we do this here manually because otherwise running multiple models using multirun was giving identical results
				# we set this right before initializing the weights for reproducability
				if 'seed' in self.cfg:
					self.random_seed = self.cfg.seed
				elif 'which_args' in self.cfg.tuning and self.cfg.tuning.which_args in ['model', self.model_name, 'best_average', 'most_similar'] and f'{self.model_name}_seed' in self.cfg.tuning:
					self.random_seed = self.cfg.tuning[f'{self.model_name}_seed']
				else:
					self.random_seed = int(torch.randint(2**32-1, (1,)))
				
				tuner_utils.set_seed(self.random_seed)
				log.info(f'Seed set to {self.random_seed}')
				
				nn.init.normal_(new_embeds.weight, mean=mean, std=std)
				
				for i, token in enumerate(self.tokens_to_mask):
					token_id = self.tokenizer.convert_tokens_to_ids(token)
					self.word_embeddings[token_id] = new_embeds.weight[i]
		
		def save_weights(weights: Dict) -> None:
			'''Saves dictionary of weights to disk'''
			with gzip.open('weights.pkl.gz', 'wb') as f:
				pkl.dump(weights, f)
		
		def get_tuner_inputs_labels() -> Tuple:
			'''
			Returns inputs and labels used for fine-tuning
			
				returns:
					tuple consisting of inputs, labels, dev_inputs, dev_labels, masked_inputs, and masked_dev_inputs
			'''
			if self.masked:
				inputs_data = self.masked_tuning_data if self.masked_tuning_style == 'always' else self.mixed_tuning_data
			elif not self.masked:
				inputs_data = self.verb_tuning_data if self.exp_type == 'newverb' else self.tuning_data

			dev_inputs_data = self.masked_dev_data
			
			labels_data = self.verb_tuning_data if self.exp_type == 'newverb' else self.tuning_data
			dev_labels_data = self.dev_data
			
			inputs = inputs_data['inputs']
			labels = labels_data['inputs']['input_ids']
			
			dev_inputs = {dataset: dev_inputs_data[dataset]['inputs'] for dataset in dev_inputs_data}
			dev_labels = {dataset: dev_labels_data[dataset]['inputs']['input_ids'] for dataset in dev_labels_data}
			
			# used to calculate metrics during training
			masked_inputs = self.masked_tuning_data['inputs']
			
			masked_dev_data = self.masked_dev_data
			masked_dev_inputs = {dataset: masked_dev_data[dataset]['inputs'] for dataset in self.masked_dev_data}
						
			return inputs, labels, dev_inputs, dev_labels, masked_inputs, masked_dev_inputs
		
		def zero_grad_for_non_added_tokens() -> None:
			'''Sets gradients to zero for tokens other than the newly added tokens'''
			nz_grad = {}
			for token in self.tokens_to_mask:
				token_id = self.tokenizer.convert_tokens_to_ids(token)
				if token_id == self.unk_token_id:
					raise ValueError(f'Added token {token} was not added correctly!')
				
				nz_grad[token_id] = self.word_embeddings.grad[token_id].clone()
			
			# Zero out all gradients of word_embeddings in-place
			self.word_embeddings.grad.data.fill_(0) # note that fill_(None) doesn't work here
			
			# Replace the original gradients at the relevant token indices
			for token_to_mask in nz_grad:
				self.word_embeddings.grad[token_to_mask] = nz_grad[token_to_mask]
		
		def verify_word_embeddings() -> None:
			'''Checks that all word embeddings except for the ones for the new tokens have not changed'''
			new_embeddings = self.word_embeddings.clone()
			num_changed_params = torch.round(torch.sum(torch.mean(torch.ne(self.old_embeddings, new_embeddings) * 1., dim = -1))) # use torch.round to attempt to fix rare floating point rounding error
			num_expected_to_change = len(self.tokens_to_mask)
			assert num_changed_params == num_expected_to_change, f'Exactly {num_expected_to_change} embeddings should have been updated, but {num_changed_params} were!'
		
		def record_epoch_metrics(
			epoch: int, outputs: 'MaskedLMOutput', delta: float, 
			dataset_name: str, metrics: List, tb_loss_dict: Dict, tb_metrics_dict: Dict, 
			best_losses: Dict, patience_counters: Dict, masked_argument_inputs: 'BatchEncoding' = None
		) -> None:
			'''
			Records metrics for a tuning epoch for a dataset in the passed arguments
			
				params:
					epoch (int)								: the epoch for which metrics are being recorded
					outputs (MaskedLMOutput)				: the outputs to collect metrics from
					delta (float)							: mean loss must improve by delta to reset patience
					dataset_name (str)						: the name of dataset for which metrics are being recorded
					metrics (list) 							: list of metrics to append new metrics to
					tb_loss_dict (dict)						: dict with losses for each epoch to add to tensorboard
					tb_metrics_dict (dict)					: dict with metrics for each epoch/token to add to tensorboard
					best_losses (dict)						: dict containing the best loss for each dataset up to the current epoch
					patience_counters (dict)				: dict containing current patience for each dataset
					masked_argument_inputs (BatchEncoding)	: inputs used to collect odds ratios for arguments in newverb experiments
			'''
			def get_mean_epoch_metrics(
				results: Dict, 
				eval_groups: Union[List,Dict] = None,
				metrics: List[str] = ['log probability', 'surprisal', 'odds ratio']
			) -> Dict:
				'''
				Calculates mean metrics for results at an epoch
				
					params:
						results (dict) 		: dict containing results to get means over
						eval_groups (dict) 	: a dict/list specifying which groups to get means over
						metrics 			: which metrics to get means for
						
					returns:
						epoch_metrics (dict): dict containing mean metrics for current epoch for each group in eval_groups
				'''
				if eval_groups is None:
					eval_groups = self.tokens_to_mask
				
				epoch_metrics = {}
				for metric in metrics:
					if any(metric in r for r in results):
						epoch_metrics[metric] = {}
						for arg_type in eval_groups:
							if any(metric in r for r in results if r['arg type'] == arg_type):
								epoch_metrics[metric][arg_type] = float(torch.mean(torch.tensor([r[metric] for r in results if r['arg type'] == arg_type])))
								# if we have more than one token that fits into this group, we get means for each of them separately as well
								if isinstance(eval_groups,dict) and arg_type in eval_groups and isinstance(eval_groups[arg_type],list) and len(eval_groups[arg_type]) > 1:
									for token in eval_groups[arg_type]:
										if any(metric in r for r in results if r['token'] == token):
											epoch_metrics[metric][arg_type].update({f'{token} ({arg_type})': float(torch.mean(torch.tensor([r[metric] for r in results if r['token'] == token])))})
					
				return epoch_metrics
			
			dataset_name = dataset_name.replace('_', ' ')
			dataset_type = dataset_name.replace('(masked, no dropout)', '(train)')
			dataset_type = re.sub(r'.*?(\w*)(?=\)|$)', '\\1', dataset_type)
			
			metrics_dict = {'epoch': epoch + 1, 'dataset': dataset_name, 'dataset_type': dataset_type}
			metrics.append({**metrics_dict, 'metric': 'loss', 'value': outputs.loss.item()})
			
			tb_loss_dict.update({dataset_name: outputs.loss})
			if outputs.loss.item() < best_losses[dataset_name] - delta:
				best_losses[dataset_name] = outputs.loss.item()
				patience_counters[dataset_name] = self.patience
			else:
				patience_counters[dataset_name] -= 1
				patience_counters[dataset_name] = max(patience_counters[dataset_name], 0)
			
			metrics.append({**metrics_dict, 'metric': 'remaining patience', 'value': patience_counters[dataset_name]})
			
			train_results = self.__collect_results(outputs=outputs, masked_token_indices=self.masked_tuning_data['masked_token_indices'])
			epoch_metrics = get_mean_epoch_metrics(results=train_results)
			
			if self.exp_type == 'newverb' and masked_argument_inputs is not None:
				newverb_outputs = self.model(**masked_argument_inputs)
				newverb_results = self.__collect_results(outputs=newverb_outputs, masked_token_indices=self.masked_tuning_data['sentence_arg_indices'], eval_groups=self.cfg.tuning.args)
				newverb_epoch_metrics = get_mean_epoch_metrics(results=newverb_results, eval_groups=self.cfg.tuning.args)
				epoch_metrics = {metric: {**epoch_metrics.get(metric, {}), **newverb_epoch_metrics.get(metric, {})} for metric in set(epoch_metrics.keys()).union(set(newverb_epoch_metrics.keys()))}
			
			for metric in epoch_metrics:
				tb_metrics_dict[metric] = {}
				for token in epoch_metrics[metric]:
					tb_metrics_dict[metric][token] = {dataset_name: epoch_metrics[metric][token]}
					metrics.append({**metrics_dict, 'metric': f'{token} mean {metric} in expected position', 'value': epoch_metrics[metric][token]})
		
		def add_tb_epoch_metrics(
			epoch: int, writer: SummaryWriter, 
			tb_loss_dict: Dict, dev_losses: List[float], 
			tb_metrics_dict: Dict
		) -> None:
			'''
			Adds metrics from dicts to tensorboard writer
			
				params:
					epoch (int) 			: which epoch to add metrics for
					writer (SummaryWriter) 	: a tensorboard SummaryWriter to add metrics to
					tb_loss_dict (dict)		: dict containing loss information
					dev_losses (list)		: list containing losses for each dev set
					tb_metrics_dict (dict) 	: dict containing metrics information
			'''
			writer.add_scalars('loss', tb_loss_dict, epoch)
			writer.add_scalar('loss/mean dev', np.mean(dev_losses), epoch)
			writer.add_scalar('loss/mean dev lower ci', np.mean(dev_losses) - np.std(dev_losses), epoch)
			writer.add_scalar('loss/mean dev upper ci', np.mean(dev_losses) + np.std(dev_losses), epoch)
			
			for metric in tb_metrics_dict:
				for token in tb_metrics_dict[metric]:
					writer.add_scalars(f'{token} mean {metric} in expected position', tb_metrics_dict[metric][token], epoch)
					
					dev_only_token_metric = [tb_metrics_dict[metric][token][dataset] for dataset in tb_metrics_dict[metric][token] if not dataset.endswith('(train)')]
					writer.add_scalar(f'{token} mean {metric} in expected position/mean dev', np.mean(dev_only_token_metric), epoch)
					writer.add_scalar(f'{token} mean {metric} in expected position/mean dev lower ci', np.mean(dev_only_token_metric) - np.std(dev_only_token_metric), epoch)
					writer.add_scalar(f'{token} mean {metric} in expected position/mean dev upper ci', np.mean(dev_only_token_metric) + np.std(dev_only_token_metric), epoch)	
		
		def add_tb_labels(epoch: int, writer: SummaryWriter, tb_metrics_dict: Dict) -> None:
			'''
			Adds labels to tensorboard summarywriter
				
				params:
					epoch (int) 			: which epoch to add labels for
					writer (SummaryWriter) 	: a tensorboard SummaryWriter to add labels to
					tb_metrics_dict (dict) 	: dict containing metrics to add labels to
			'''
			# note that we do not plot means in the pdfs if using only the no dropout training set as a dev set
			# but we DO include them in the tensorboard plots. this is because that allows us to include the 
			# hyperparameters info in the tensorboard log in SOME way without requiring us to create a directory
			# name that contains all of it (which results in names that are too long for the filesystem)
			model_label = f'{self.model_name} {self.tuning.replace("_", " ")}, '
			if self.exp_type == 'newverb':
				model_label += f'args group: {self.cfg.tuning.which_args}, '
			
			model_label += f'masking: {self.masked_tuning_style}, ' if self.masked else 'unmasked, '
			model_label += 'mask args, ' if self.mask_args else ''
			model_label += f'{"no punctuation" if self.strip_punct else "punctuation"}, '
			model_label += f'epochs={epoch+1} (min={self.min_epochs}, max={self.max_epochs}), '
			model_label += f'pat={self.patience} (\u0394={self.delta})'
			model_label += f'unfreezing={self.unfreezing}'
			if self.unfreezing == 'gradual':
				model_label += f'({self.unfreezing_epochs_per_layer})'
			
			# Aggregate the plots and add a helpful label
			# note that tensorboard does not support plotting means and CIs automatically even when aggregating
			# Thus, we only do this manually for the average dev loss, since plots of other means are in the PDF
			# and are less likely to be useful given the effort it would take to manually construct them
			metrics_labels = {'mean dev loss' : ['Margin', ['loss/mean dev', 'loss/mean dev lower ci', 'loss/mean dev upper ci']]}
			for metric in tb_metrics_dict:
				for token in tb_metrics_dict[metric]:
					metrics_labels[f'mean dev {token} mean {metric} in expected position'] = \
						['Margin', 
							[f'{token} mean {metric} in expected position/mean dev', 
							 f'{token} mean {metric} in expected position/mean dev lower ci', 
							 f'{token} mean {metric} in expected position/mean dev upper ci']
						]
			
			layout = {model_label: metrics_labels}
			
			writer.add_custom_scalars(layout)
		
		initialize_added_token_weights()
		
		# Store weights pre-training so we can inspect the initial status later
		saved_weights = {'random_seed': self.random_seed, 0: self.added_token_weights}
		
		if not self.tuning_data or (self.exp_type == 'newverb' and self.unfreezing is not None):
			log.info(f'Saving randomly initialized weights')
			save_weights(saved_weights)
			if not self.tuning_data:	
				return
		
		# Collect Hyperparameters
		lr  		= self.lr
		epochs 		= self.max_epochs
		min_epochs 	= self.min_epochs
		patience 	= self.patience
		delta 		= self.delta
		optimizer 	= torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0)
		
		# Store the old embeddings so we can verify that only the new ones get updated
		self.old_embeddings = self.word_embeddings.clone()
		
		if self.unfreezing == 'complete':
			unfreeze_all_params()
		elif not isinstance(self.unfreezing, int):
			freeze_to_layer(self.model.config.num_hidden_layers)
		elif isinstance(self.unfreezing, int):
			freeze_to_layer(self.unfreezing)
		
		inputs, labels, dev_inputs, dev_labels, masked_inputs, masked_dev_inputs = get_tuner_inputs_labels()
		
		hyperparameters_str =  f'lr={lr}, min_epochs={min_epochs}, max_epochs={epochs}, '
		hyperparameters_str += f'patience={patience}, \u0394={delta}, unfreezing={None if np.isnan(self.unfreezing) else self.unfreezing}'
		hyperparameters_str += f'{self.unfreezing_epochs_per_layer}' if self.unfreezing == 'gradual' else ''
		hyperparameters_str += f', mask_args={self.mask_args}' if not np.isnan(self.mask_args) else ''
		
		log.info(f'Training model @ "{os.getcwd().replace(hydra.utils.get_original_cwd(), "")}" ({hyperparameters_str})')
		
		datasets = [self.tuning + ' (train)', 
					self.tuning + ' (masked, no dropout)'] + \
				   [dataset + ' (dev)' for dataset in self.cfg.dev]
		
		metrics = []
		writer = SummaryWriter()
		
		with logging_redirect_tqdm(), trange(epochs) as t:
			
			patience_counter = 0
			patience_counters = {d.replace('_', ' '): patience for d in datasets}
			
			best_mean_loss = np.inf
			best_losses = {d.replace('_', ' '): np.inf for d in datasets}
			
			best_epoch = 0
			
			try:
				for epoch in t:
					
					if self.unfreezing == 'gradual':
						self.freeze_to_layer(max(-self.model.config.num_hidden_layers, -(floor(epoch/self.unfreezing_epochs_per_layer)+1)))
					
					# debug
					if 'debug' in self.cfg and self.cfg.debug and self.exp_type == 'newverb':
						self.__print_debug_predictions(epoch, total_epochs)
					
					self.model.train()
					
					optimizer.zero_grad(set_to_none=True) # this is supposed to be faster than .zero_grad()
					
					if self.masked_tuning_style == 'roberta':
						inputs = self.mixed_tuning_data['inputs']
					
					train_outputs = self.model(**inputs, labels=labels)
					train_loss = train_outputs.loss
					train_loss.backward()
					
					tb_loss_dict, tb_metrics_dict = {}, {}
					
					record_epoch_metrics(
						epoch, train_outputs, delta, 
						self.tuning + ' (train)', metrics, 
						tb_loss_dict, tb_metrics_dict,
						best_losses, patience_counters
					)
					
					if not self.unfreezing == 'complete':
						zero_grad_for_non_added_tokens()
					
					optimizer.step()
					
					if not self.unfreezing == 'complete':
						verify_word_embeddings()
					
					saved_weights[epoch+1] = self.added_token_weights
					
					# evaluate the model on the dev set(s) and log results
					self.model.eval()
					
					with torch.no_grad():
						dev_losses = []
						for dataset in dev_inputs:
							dev_outputs = self.model(**dev_inputs[dataset], labels=dev_labels[dataset])
							dev_loss = dev_outputs.loss
							dev_losses += [dev_loss.item()]
							
							record_epoch_metrics(
								epoch, dev_outputs, delta, 
								self.cfg.dev[dataset].name + ' (dev)', metrics, 
								tb_loss_dict, tb_metrics_dict, 
								best_losses, patience_counters,
								self.masked_dev_argument_data[dataset]['inputs'] if self.exp_type == 'newverb' else None
							)
						
						# Compute loss on masked training data without dropout; this is most representative of the testing procedure, so we can use it to determine the best epoch
						no_dropout_train_outputs = self.model(**masked_inputs, labels=labels)
						no_dropout_train_loss = no_dropout_train_outputs.loss
						
						dev_losses += [no_dropout_train_loss.item()]
						
						record_epoch_metrics(
							epoch, no_dropout_train_outputs, delta, 
							self.tuning + ' (masked, no dropout)', metrics, 
							tb_loss_dict, tb_metrics_dict, 
							best_losses, patience_counters, 
							self.masked_tuning_data('arguments')['inputs'] if self.exp_type == 'newverb' else None
						)
						
					add_tb_epoch_metrics(epoch, writer, tb_loss_dict, dev_losses, tb_metrics_dict)
					
					if np.mean(dev_losses) < best_mean_loss - delta:
						best_mean_loss = np.mean(dev_losses)
						patience_counter = 0
						best_epoch = epoch + 1
						if self.save_full_model:
							# need to use a deepcopy, else this changes and we overwrite the best state
							best_model_state_dict = deepcopy(self.model.state_dict())
					else:
						patience_counter += 1
						patience_counter = min(patience, patience_counter)
						
					metrics.append({'epoch': epoch + 1, 'dataset': 'overall', 'dataset_type': 'overall', 'metric': 'remaining patience overall', 'value': patience - patience_counter})
					writer.add_scalars('remaining patience', {**patience_counters, 'overall': patience - patience_counter}, epoch)
					t.set_postfix(pat=patience - patience_counter, avg_dev_loss='{0:5.2f}'.format(np.mean(dev_losses)), train_loss='{0:5.2f}'.format(train_loss.item()))
					
					if patience_counter >= patience and epoch + 1 >= min_epochs:
						log.info(f'Mean dev loss has not improved by {delta} in {patience_counter} epochs (min_epochs={min_epochs}). Halting training at epoch {epoch}.')
						break
			
			except KeyboardInterrupt:
				log.warning(f'Training halted manually at epoch {epoch+1}')
				pass
			
		# debug
		if 'debug' in self.cfg and self.cfg.debug and self.exp_type == 'newverb':
			self.__print_debug_predictions(epoch, total_epochs)
			log.info('')
		
		add_tb_labels(epoch, writer, tb_metrics_dict)
		
		if not self.save_full_model:
			# we do minus two here because we've saved the randomly initialized weights @ 0 and the random seed
			log.info(f'Saving weights for random initializations and each of {len(saved_weights)-2} training epochs')
			save_weights(saved_weights)
		else:
			log.info(f'Saving model state with lowest avg dev loss (epoch={best_epoch}) to disk')
			with open(os.path.join(self.checkpoint_dir, 'model.pt'), 'wb') as f:
				torch.save(best_model_state_dict, f)
		
		metrics = pd.DataFrame(metrics)
		metrics = self.__add_hyperparameters_to_summary_df(metrics)
		breakpoint()
		metrics.metric = self.__format_strings_with_tokens_for_display(metrics.metric)
		metrics = metrics.sort_values(
			by=['dataset','metric','epoch'], 
			key=lambda col: col.astype(str).str.replace(r'(.*\(train\))', '0\\1', regex=True) \
										   .str.replace(r'(.*\(masked, no dropout\))', '1\\1', regex=True) \
										   .str.rjust(len(str(max(metrics.epoch)))) \
										   .str.lower()
		).reset_index(drop=True)
		
		log.info('Saving metrics')
		metrics.to_csv(os.path.join(self.checkpoint_dir, 'metrics.csv.gz'), index=False, na_rep='NaN')
		
		log.info('Plotting metrics')
		self.create_metrics_plots(metrics)
		
		writer.flush()
		writer.close()
	
	def create_inputs(
		self,
		sentences: List[str] = None,
		to_mask: Union[List[int],List[str]] = None,
		masking_style: str = 'always'
	) -> Tuple['BatchEncoding', torch.Tensor, List[Dict]]:
		'''
		Creates masked model inputs from a batch encoding or a list of sentences, with tokens in to_mask masked
				
			params:
				sentences (list) 				: a list of sentences to get inputs for
				to_mask (list)					: list of token_ids or token strings to mask in the inputs
				masking_style (str)				: if 'always', always replace to_mask token_ids with mask tokens
												  if 'none', do nothing
												  if None, return bert/roberta-style masked data
			
			returns:
				masked_inputs (BatchEncoding)	: the inputs with the tokens in to_mask replaced with mask, 
												  original, or random tokens, dependent on masking_style
				labels (tensor)					: a tensor with the target labels
				masked_token_indices (list)		: a list of dictionaries mapping the (display-formatted) tokens in to_mask to their 
												  original position(s) in each sentence (since they are no longer in the masked sentences)
		'''
		
		if to_mask is None:
			to_mask = self.tokens_to_mask
		
		to_mask = [self.tokenizer.convert_tokens_to_ids(token) if isinstance(token,str) else token for token in to_mask]
		assert not any(token_id == self.unk_token_id for token_id in to_mask), 'At least one token to mask is not in the model vocabulary!'
		
		inputs = self.__format_data_for_tokenizer(sentences)
		if not tuner_utils.verify_tokenization_of_sentences(self.tokenizer, inputs, self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs):
			log.error('Added tokens affected the tokenization of sentences!')
			return
		
		inputs 					= self.tokenizer(inputs, return_tensors='pt', padding=True)
		labels 					= inputs['input_ids'].clone().detach()
		
		to_mask_indices 		= [np.where([token_id in to_mask for token_id in sentence])[-1].tolist() for sentence in inputs['input_ids']]
		to_mask_ids				= [[int(token_id) for token_id in sentence if token_id in to_mask] for sentence in inputs['input_ids']]
		
		masked_token_indices 	= []
		for token_ids, token_locations in zip(to_mask_ids, to_mask_indices):
			for token_id, token_location in zip(token_ids, token_locations):
				token = self.tokenizer.convert_ids_to_tokens(token_id)
				masked_token_indices.append({token: token_location})
		
		masked_inputs = inputs.copy()
		if masking_style != 'none':
			for i, (tokenized_sentence, indices) in enumerate(zip(inputs['input_ids'], to_mask_indices)):
				for index in indices:
					# even when using bert/roberta style tuning, we sometimes need access to the data with everything masked
					r = np.random.random()
					# Roberta tuning regimen: 
					# masked tokens are masked 80% of the time,
					# original 10% of the time, 
					# and random word 10% of the time
					if (r < 0.8 or masking_style == 'always') and not masking_style == 'none':
						replacement = self.mask_token_id
					elif 0.8 <= r < 0.9 or masking_style == 'none':
						replacement = inputs['input_ids'][i][index]
					elif 0.9 <= r:
						replacement = np.random.choice(list(self.tokenizer.get_vocab().values()))
					
					masked_inputs['input_ids'][i][index] = replacement
		
		return masked_inputs, labels, masked_token_indices
	
	def restore_weights(self, epoch: Union[int,str] = 'best_mean') -> Tuple[int,int]:
		'''
		Restores model weights @ the specified epoch
		
			params:
				epoch (int or str) 				: if int, the epoch to restore weights from
									  			  if str, a description of which epoch to pick (best_mean or best_sumsq)
			
			returns:
				epoch (int), total_epochs (int) : the epoch to which the model was restored, 
									  			  and the total number of epochs for which the model was trained	
		'''
		# if we have saved the full model and we are not on epoch zero, prefer that
		# this occurs when not freezing all parameters except the added token weights
		model_path = os.path.join(self.checkpoint_dir, 'model.pt')
		metrics = pd.read_csv(os.path.join(self.checkpoint_dir, 'metrics.csv.gz'))
		total_epochs = max(metrics.epoch)
		loss_df = metrics[(metrics.metric == 'loss') & (~metrics.dataset.str.endswith(' (train)'))]
		if os.path.isfile(model_path) and not epoch == 0:
			# we use the metrics file to determine the epoch at which the full model was saved
			# note that we have not saved the model state at each epoch, unlike with the weights
			# this is a limitation of the gradual unfreezing approach
			epoch = tuner_utils.get_best_epoch(loss_df, method = 'mean')
			
			log.info(f'Restoring model state from epoch {epoch}/{total_epochs}')
			
			with open(model_path, 'rb') as f:
				self.model.load_state_dict(torch.load(f))
		else:
			# we do this because we need to make sure that when we are restoring from 0, we start at the correct state
			# this is now required because we have added gradual unfreezing to our pipeline
			# nothing will go wrong if we are restoring from a later epoch for another model, because in that case, all that changes is the weights
			self.tokenizer = tuner_utils.create_tokenizer_with_added_tokens(self.string_id, self.cfg.tuning.to_mask, **self.cfg.model.tokenizer_kwargs)
			self.model = AutoModelForMaskedLM.from_pretrained(self.string_id, **self.cfg.model.model_kwargs)
			self.model.resize_token_embeddings(len(self.tokenizer))
			
			weights_path = os.path.join(self.checkpoint_dir, 'weights.pkl.gz')
			with gzip.open(weights_path, 'rb') as f:
				weights = pkl.load(f)
			
			if epoch == None or epoch in ['max', 'total', 'highest', 'last', 'final']:
				epoch = total_epochs
			elif 'best' in str(epoch):
				epoch = tuner_utils.get_best_epoch(loss_df, method = epoch)
			
			log.info(f'Restoring saved weights from epoch {epoch}/{total_epochs}')
			
			with torch.no_grad():
				for token in weights[epoch]:
					token_id = self.tokenizer.convert_tokens_to_ids(token)
					self.word_embeddings[token_id] = weights[epoch][token]
		
		# return the epoch and total_epochs to help if we didn't specify it
		return epoch, total_epochs
	
	def predict_sentences(
		self,
		sentences: List[str] = None,
		info: str = '',
		output_fun: Callable = print
	) -> List[Dict]:
		'''
		Returns the model's predictions for each sentence in sentences
		
			params:
				sentences (list) : list of sentences to get predictions for
				info (str)		 : used when outputting results
				output_fun (fun) : how to display results
			
			returns:
				results (dict)	 : dict with inputs sentences, predicted_sentences, and model outputs 
								   for each sentence in sentences
		'''
		restore_training = False
		if self.model.training:
			restore_training = True
			log.warning('Cannot predict in training mode. Setting to eval mode temporarily.')
			self.model.eval()
		
		if sentences is None:
			sentences = f'The local {self.mask_token} will step in to help.'
			log.info(f'No sentence was provided. Using default sentence "{sentences}"')
		
		sentences = self.__format_data_for_tokenizer(sentences)
		
		with torch.no_grad():
			outputs = self.model(**self.tokenizer(sentences, return_tensors='pt', padding=True))
		
		logprobs = nn.functional.log_softmax(outputs.logits, dim=-1)
		predicted_ids = torch.squeeze(torch.argmax(logprobs, dim=-1))
		predicted_sentences = [self.tokenizer.decode(predicted_sentence_ids) for predicted_sentence_ids in predicted_ids]
		
		if output_fun is not None:
			for sentence, predicted_sentence in zip(sentences, predicted_sentences):
				output_fun(f'{info + " " if info else ""}input: {sentence}, prediction: {predicted_sentence}')
		
		if restore_training:
			self.model.train()
		
		return {'inputs': sentences, 'predictions': predicted_sentences, 'outputs': outputs}
	
	
	# dimensionality reductions
	def get_cossims(
		self, tokens: List[str] = [], 
		targets: Dict[str,str] = {}, topk: int = 50
	) -> pd.DataFrame:
		'''
		Returns a dataframe containing information about the k most similar tokens to tokens
		If targets is provided, also includes infomation about the cossim of the tokens to 
		the targets they are mapped to
			
			params:
				tokens (list) 			: list of tokens to get cosine similarities for
				targets (dict)			: for each token in tokens, which tokens to get cosine similarities for
				topk (int)				: how many of the most similar tokens to tokens to record
			
			returns:
				cossims (pd.DataFrame)	: dataframe containing information about cosine similarities for each token/target combination + topk most similar tokens
		'''
		def format_tokens_targets(tokens: List[str] = None, targets: Dict[str,List[str]] = {}) -> Tuple[List[str], Dict[str,List[str]]]:
			'''
			Formats tokens and targets according to the conventions of different model tokenizers
			
				params:
					tokens (list) 					: list of tokens to format
					targets (dict)					: dict containing tokens to format
				
				returns:
					tokens (list), targets (dict)	: tokens and targets formatted according to model conventions
			'''
			
			if tokens is None:
				tokens = self.tokens_to_mask
			
			tokens = self.__format_data_for_tokenizer(tokens)
			
			if targets:
				targets = self.__format_data_for_tokenizer(targets)
			
			# if we are training roberta, we only currently care about the cases with spaces in front for masked tokens
			# otherwise, try to do something sensible with other tokens
			# if they exist, use them
			# if they have a space in front, replace it with a chr(288)
			# if they don't exist, but a version with a space in front does, use that
			if self.model_name == 'roberta':
				tokens = [t for t in tokens if t.startswith(chr(288) and t in self.tokens_to_mask) or not t in self.tokens_to_mask]
				tokens = [t if tuner_utils.verify_tokens_exist(t) else chr(288) + t if tuner_utils.verify_tokens_exist(chr(288) + t) else None for t in tokens]
				tokens = [t for t in tokens if t is not None]
				tokens = tuner_utils.format_roberta_tokens_for_display(tokens)
				
				# filter the keys in targets ...
				targets = {k if k in tokens else chr(288) + key if chr(288) + key in tokens else '' : v for key, v in targets.items()}
				targets = {k: v for k, v in targets.items() if k}
				targets = {k if tuner_utils.verify_tokens_exist(k) else chr(288) + k if tuner_utils.verify_tokens_exist(chr(288) + key) else key : 
						   v if tuner_utils.verify_tokens_exist(v) else chr(288) + v if tuner_utils.verify_tokens_exist(chr(288) + v) else [] for key, v in targets.items()}
				targets = {k : v for k, v in targets.items() if targets[key]}
				targets = tuner_utils.format_roberta_tokens_for_display(targets)
				
				# ... and the values
				for k in targets:
					targets[k] = [t for t in targets[key] if t.startswith(chr(288) and t in self.tokens_to_mask) or not t in self.tokens_to_mask]
					targets[k] = [chr(288) + t if key.startswith(chr(288)) else t for t in targets[k]] # if the key has a preceding space, then we're only interested in predictions for tokens with preceding spaces
					targets[k] = [t if tuner_utils.verify_tokens_exist(t) else None for t in targets[k]]
					targets[k] = [t for t in targets[k] if t is not None]
					targets[k] = tuner_utils.format_roberta_tokens_for_display(targets[k])
				
				targets = {k: v for k, v in targets.items() if all(targets[k])}
			else:
				tokens = [t for t in tokens if tuner_utils.verify_tokens_exist(t)]
				targets = {k : v for k, v in targets.items() if tuner_utils.verify_tokens_exist(key)}
				for k in targets:
					targets[k] = [t for t in targets[k] if tuner_utils.verify_tokens_exist(t)]
				
				targets = {k: v for k, v in targets.items() if all(targets[k])}
			
			return tokens, targets
		
		def update_cossims(
			cossims: List[Dict], values: List[float],
			included_ids: List[int] = [], excluded_ids: List[int] = [], 
			k: int = None, target_group: str = ''
		) -> None:
			'''
			Updates the cossims list
				
				params:
					cossims (list)			: list of dicts containing information about cosine similarities
					values (list)			: cosine similarities for each token in the model's vocabulary
					included_ids (list)		: which cosine similarities to include in cossims
					excluded_ids (list)		: which cosine similarities to exclude from cossims
					k (int)					: how many of the most similar tokens to include
					target_group (str)		: which target_group cosine similarites are being recorded for
			'''
			if not included_ids:
				included_ids = list(range(len(values)))
			
			excluded_ids = tuner_utils.listify(excluded_ids)
			included_ids = tuner_utils.listify(included_ids)
			included_ids = set(included_ids).difference(set(excluded_ids))
			
			if k is None:
				k = len(included_ids)
			
			k = min(k, len(included_ids))
			
			cossims.extend([{
				'predicted_arg'	: token,
				'token_id' 		: i,
				'target_group'	: target_group, 
				'token'			: self.tokenizer.convert_ids_to_tokens(i), 
				'cossim'		: cossim
			} for i, cossim in enumerate(values) if i in included_ids][:k])
		
		tokens, targets = format_tokens_targets(tokens, targets)
		
		cos = nn.CosineSimilarity(dim=-1)
		cossims = []
		
		for token in tokens:
			token_id = self.tokenizer.convert_tokens_to_ids(token)
			token_embedding = self.word_embeddings[token_id]
			token_cossims = cos(token_embedding, self.word_embeddings)
			included_ids = torch.topk(token_cossims, k=topk+1).indices.tolist() # add one so we can leave out the identical token
			token_cossims = token_cossims.tolist()
			
			update_cossims(cossims=cossims, values=token_cossims, included_ids=included_ids, excluded_ids=token_id, k=topk, target_group=f'{topk} most similar')
			
			if token in targets:
				target_ids = [token_id for token_id in self.tokenizer.convert_tokens_to_ids(targets[token]) if token_id != self.unk_token_id]
				update_cossims(cossims=cossims, values=token_cossims, included_ids=target_ids, target_group=token)
				
				out_groups = {k: v for k, v in targets.items() if not k == token}
				for out_group_token in out_groups:
					out_group_target_ids = [token_id for token_id in self.tokenizer.convert_tokens_to_ids(targets[out_group_token]) if token_id != self.unk_token_id]
					update_cossims(cossims=cossims, values=token_cossims, included_ids=out_group_target_ids, target_group=out_group_token)
		
		cossims = pd.DataFrame(cossims)
		
		return cossims
	
	def get_tsnes(
		self, n: int = None, targets: Dict[str,List[str]] = None,
		target_group_labels: Dict[str,str] = None,
		ndims: int = 2, random_tsne_state: int = 0, 
		learning_rate: str = 'auto', 
		init: str = 'pca', **tsne_kwargs
	) -> pd.DataFrame:
		'''
		Fits a TSNE to the model's word embeddings and returns the results
			
			params:
				n (int) 					: how many of the first n filtered tokens in the tokenizer to include
				targets (dict)				: maps token groups to targets to get TSNEs for
				target_group_labels (dict)	: names of the target groups (if different from the tokens)
				ndims (int)					: how many dims to fit the tsne to
				random_tsne_state (int)		: passed to TSNE()
				learning_rate (str)			: passed to TSNE()
				init (str)					: passed to TSNE()
				**tsne_kwargs (dict)		: passed to TSNE()
			
			returns:
				tsnes (pd.DataFrame)		: dataframe with information about tsne components
		'''
		def get_formatted_tsne_targets(n: int, targets: Dict, added_words: List[str]) -> Dict:
			'''
			Returns tsne targets formatted according to model conventions
			
				params:
					n (int) 			: how many of the first n good candidates to include
					targets (dict)		: also include everything manually specified in here
					added_words (list)	: additional words to include
				
				returns:
					targets (dict)		: dict mapping token groups to targets formatted according to model conventions
			'''
			target_values = list(itertools.chain(*list(targets.values())))
			tokenizer_keys = tuple(self.tokenizer.get_vocab().keys())
			formatted_keys = [k.replace(chr(288), '').lower() for k in tokenizer_keys] # we convert to lower b/c that's how we compare them to the dataset words
			
			pos = 'verbs' if self.exp_type == 'newverb' else 'nouns'
			
			with open(os.path.join(hydra.utils.get_original_cwd(), 'conf', pos + '.txt'), 'r') as f:
				candidates = [w.lower().strip() for w in f.readlines()]
			
			candidates 		+= added_words
			target_values 	+= added_words if target_values else []
			
			names_sets = []
			if n is not None:
				names_sets.append((f'first {n}', candidates))
			
			if targets is not None:
				names_sets.append(('targets', target_values))
			
			targets = {}
			for name, candidate_set in names_sets:
				targets[name] = {}
				filtered_keys = [k for k in formatted_keys if k in candidate_set]
				selected_keys = [tokenizer_keys[tokenizer_keys.index(formatted_keys[formatted_keys.index(k)])] for k in filtered_keys]
				if name == f'first {n}':
					selected_keys = [k for k in selected_keys if selected_keys.index(k) < n or k in added_words]
				
				targets[name]['tokens'] = {k: self.tokenizer.convert_tokens_to_ids(k) for k in selected_keys}
				if self.model_name == 'roberta':
					# if we are using roberta, filter to tokens that start with a preceding space and are not followed by a capital letter (to avoid duplicates))
					targets[name]['tokens'] = {k: v for k, v in targets[name].items() if k.startswith(chr(288)) and not re.search(fr'^{chr(288)}[A-Z]', k)}
				
				targets[name]['embeddings'] = {k: self.word_embeddings[v].reshape(1,-1) for k, v in targets[name]['tokens'].items()}
				targets[name]['words'] 		= list(targets[name]['tokens'].keys())
			
			return targets
		
		masked_token_targets = self.__format_data_for_tokenizer(targets)
		added_words = self.tokens_to_mask
		targets = get_formatted_tsne_targets(n=n, targets=targets, added_words=added_words)
		
		if target_group_labels is not None:
			target_group_labels = self.__format_data_for_tokenizer(target_group_labels)
		else:
			target_group_labels = self.__format_data_for_tokenizer({k: k for k in self.tokens_to_mask})
		
		tsne_results = []
		for group in targets:
			
			# we create the TSNE object inside the loop to use the same random state for reproducibility
			tsne = TSNE(ndims, random_state=random_tsne_state, learning_rate=learning_rate, init=init, **tsne_kwargs)
			
			with torch.no_grad():
				vectors = torch.cat([embedding for embedding in targets[group]['embeddings'].values()])
				two_dim = tsne.fit_transform(vectors)
			
			for token, tsne1, tsne2 in zip(targets[group]['words'], two_dim[:,0], two_dim[:,1]):
				
				target_group = 'novel token' \
								if token in added_words \
								else group if group == f'first {n}' \
								else [target_group for target_group in masked_token_targets if token in masked_token_targets[target_group]][0]
				
				target_group_label = target_group_labels[target_group] if target_group in target_group_labels else target_group
				
				tsne_results.append({
					'token' 			: token,
					'tsne1'				: tsne1,
					'tsne2'				: tsne2,
					'token_category'	: 'novel' if token in added_words else 'existing',
					'token_id'			: self.tokenizer.convert_tokens_to_ids(token),
					'tsne_type'			: group,
					'target_group'		: target_group,
					'target_group_label': target_group_label
				})
		breakpoint()
		tsne_results = pd.DataFrame(tsne_results)
		
		return tsne_results
	
	
	# evaluation functions
	def evaluate(self, eval_cfg: DictConfig) -> None:
		# this is just done so we can record it in the results
		self.__restore_original_random_seed()
		
		if eval_cfg.data.exp_type in ['newverb', 'newarg']:
			self.__evaluate_newtoken_experiment(eval_cfg=eval_cfg)
		else:
			self.__eval(eval_cfg=eval_cfg)
	
	def load_eval_file(self, eval_cfg: DictConfig) -> Dict:
		'''
		Loads a file from the specified path, returning a Dict of sentence types for model evaluation.
			
			params: 
				eval_cfg (DictConfig) : dictconfig containing evaluation configuration options
			
			returns:
				types_sentences (dict): dict with sentences, inputs, and arg_indices 
										for each sentence type in the eval data file
		'''
		resolved_path = os.path.join(hydra.utils.get_original_cwd(), 'data', eval_cfg.data.name + '.data')
			
		with open(resolved_path, 'r') as f:
			raw_input = [line.strip() for line in f]
		
		sentences = [[s.strip() for s in r.split(' , ')] for r in raw_input]
		sentences = self.__format_data_for_tokenizer(sentences)
		transposed_sentences = list(map(list, zip(*sentences)))
		
		if self.exp_type in ['newverb', 'newarg']:
			sentence_types = eval_cfg.data.sentence_types
		else:
			# dummy value
			sentence_types = range(len(sentences))
		
		assert len(eval_cfg.data.sentence_types) == len(transposed_sentences), 'Number of sentence types does not match in data config and data!'
		
		to_mask = list(self.cfg.tuning.args.keys()) if self.exp_type == 'newverb' else self.tokens_to_mask
		
		types_sentences = {}
		for sentence_type, sentence_type_group in zip(sentence_types, transposed_sentences):
			breakpoint()
			types_sentences[sentence_type] = {}
			types_sentences[sentence_type]['sentences'] = sentence_type_group
			
			masked_inputs, _, sentence_arg_indices = self.create_inputs(sentences=sentence_type_group, to_mask=to_mask, masking_style='always')
			types_sentences[sentence_type]['inputs'] = masked_inputs
			types_sentences[sentence_type]['sentence_arg_indices'] = sentence_arg_indices
		
		# flatten the dict if we are not using sentence types
		if not self.exp_type in ['newverb', 'newarg']:
			tmp_dict = {}
			for sentence_type in types_sentences:
				tmp_dict.update(**types_sentences[sentence_type])
			
			types_sentences = tmp_dict
		
		return types_sentences
	
	def summarize_results(self, results: Dict) -> Dict:
		'''
		Summarizes model results
		
			params:
				results (dict): a dict containing results
			
			returns:
				results (dict): dict containing summary of results
		'''
		summary = {}
		
		# Define theme and recipient ids
		ricket = 'RICKET' if not 'uncased' in self.string_id else 'ricket'
		thax = 'THAX' if not 'uncased' in self.string_id else 'thax'
		
		ricket = self.tokenizer.convert_tokens_to_ids(ricket)
		thax = self.tokenizer.convert_tokens_to_ids(thax)
		
		assert ricket != self.unk_token_id, 'RICKET was not correctly added to the tokenizer!'
		assert thax != self.unk_token_id, 'RICKET was not correctly added to the tokenizer!'
		
		# Cumulative log probabilities for <token> in <position>
		theme_in_theme = []
		theme_in_recipient = []
		recipient_in_theme = []
		recipient_in_recipient = []
		
		# Confidence in predicting <token> over the alternative
		ricket_confidence = []
		thax_confidence = []
		
		# Confidence that position is an <animacy> noun
		animate_confidence = []
		inanimate_confidence = []
		
		# Entropies in various positions
		theme_entropy = []
		recipient_entropy = []
		
		for result in results:
			for token in result:
				scores = result[token]['mean grouped log probability']
				probabilities = result[token]['probabilities']
				categorical_distribution = Categorical(probs=probabilities)
				entropy = categorical_distribution.entropy()
				
				if target == ricket:
					theme_in_recipient.append(scores['theme'])
					recipient_in_recipient.append(scores['recipient'])
					recipient_entropy.append(entropy)
					ricket_confidence.append(scores['recipient'] - scores['theme'])
					animate_confidence.append(scores['animate'] - scores['inanimate'])
				elif target == thax:
					theme_in_theme.append(scores['theme'])
					recipient_in_theme.append(scores['recipient'])
					theme_entropy.append(entropy)
					thax_confidence.append(scores['theme'] - scores['recipient'])
					inanimate_confidence.append(scores['animate'] - scores['inanimate'])
		
		summary['theme'] = {
			'entropy' : theme_entropy,
			'animacy_conf' : inanimate_confidence,
			'token_conf' : thax_confidence
		}
		
		summary['recipient'] = {
			'entropy' : recipient_entropy,
			'animacy_conf' : animate_confidence,
			'token_conf' : ricket_confidence
		}
		
		return summary
	
	def get_odds_ratios_summary(
		self, 
		epoch: int, 
		eval_cfg: DictConfig, 
		data: Dict = None, 
		return_type: str = 'df'
	) -> Union[pd.DataFrame,Dict]:
		'''
		Returns a dataframe containing a summary of odds ratios data
			
			params:
				epoch (int)									: which epoch to evaluate
				eval_cfg (DictConfig)						: a dictconfig containing evaluation config options
				data (dict)									: a dict consisting of sentences, inputs, and arg_indices
															  to evaluate model performance on
				return_type (str)							: if 'df', returns a dataframe. else returns a list of dicts
			
			returns:
				odds_ratios_summary (pd.DataFrame or list)	: dataframe or list of dicts containing a summary of the odds ratios data
		'''
		# use data if provided so we don't have to reload it, but load it automatically if not
		data = self.load_eval_file(eval_cfg) if data is None else data
		
		epoch, total_epochs = self.restore_weights(epoch)
		
		# get hyperparameters/config info to add to the summary
		eval_parameters = {
			'eval_data' 	: eval_cfg.data.split('.')[0],					
			'epoch_criteria': eval_cfg.epoch if isinstance(eval_cfg.epoch, str) else 'manual',
			'eval_epoch' 	: epoch,
			'total_epochs' 	: total_epochs,
		}
		
		# debug
		if 'debug' in eval_cfg and eval_cfg.debug:
			self.__print_debug_predictions(epoch, total_epochs)
			log.info('')
		
		if eval_cfg.data.exp_type == 'newverb':	
			which_args = self.cfg.tuning.which_args if not self.cfg.tuning.which_args == 'model' else self.model_name
			
			args = self.cfg.tuning.args
			if 'added_args' in eval_cfg.data:
				if which_args in eval_cfg.data.added_args:
					args = {arg_type: args[arg_type] + eval_cfg.data.added_args[which_args][arg_type] for arg_type in args}
		else:
			args = {arg: [arg] for arg in self.tokens_to_mask}
			tokens_to_roles = {v: k for k, v in eval_cfg.data.eval_groups.items()}
			tokens_to_roles = self.__format_data_for_tokenizer(tokens_to_roles)
			
			if self.model_name == 'roberta':
				tokens_to_roles.update({chr(288) + token: tokens_to_roles[token] for token in tokens_to_roles.copy()})
		
		log.info(f'Evaluating model on testing data')
		odds_ratios_summary = []
		for sentence_type in tqdm(data):
			with torch.no_grad():
				sentence_type_outputs = self.model(**data[sentence_type]['inputs'])
			
			sentence_type_logprobs = nn.functional.log_softmax(sentence_type_outputs.logits, dim=-1)
			
			for arg_type in args:
				for arg in args[arg_type]:
					for sentence_num, (arg_indices, sentence, logprob) in enumerate(zip(data[sentence_type]['sentence_arg_indices'], data[sentence_type]['sentences'], sentence_type_logprobs)):
						arg_name = chr(288) + arg if self.model_name == 'roberta' and not sentence.startswith(arg_type) else arg
						arg_token_id = self.tokenizer.convert_tokens_to_ids(arg_name)
						assert arg_token_id != self.unk_token_id, f'Argument {arg_name} was not tokenized correctly! Try using a different one instead.'
						
						positions = sorted(list(arg_indices.keys()), key = lambda arg_type: arg_indices[arg_type])
						positions = {p: positions.index(p) + 1 for p in positions}	
						
						for arg_position, arg_index in [(arg_position, arg_index) for arg_position, arg_index in arg_indices.items() if not arg_position == arg_type]:
							log_odds = logprob[arg_index,arg_token_id]
							exp_log_odds = logprob[arg_indices[arg_type],arg_token_id]
							odds_ratio = exp_log_odds - log_odds
							
							if eval_cfg.data.exp_type == 'newverb':
								token_type = {'token_type': 'tuning' if arg in self.cfg.tuning.args[arg_type] else 'eval_only'}
							else:
								token_type = {'role_position': tokens_to_roles[arg_type]}
							
							odds_ratios_summary.append({
								'odds_ratio' 			: odds_ratio,
								'ratio_name' 			: arg_type + '/' + arg_position,
								'position_ratio_name' 	: f'position {positions[arg_type]}/position {positions[arg_position]}',
								'token_id' 				: arg_token_id,
								'token' 				: arg_name,
								**token_type,
								'sentence' 				: sentence,
								'sentence_type' 		: sentence_type,
								'sentence_num' 			: sentence_num,
								**eval_parameters
							})
		
		if return_type.lower() in ['df', 'pd', 'dataframe', 'pd.dataframe']:
			odds_ratios_summary = pd.DataFrame(odds_ratios_summary)
			breakpoint()
			odds_ratios_summary = self.__add_hyperparameters_to_summary_df(odds_ratios_summary)
		
		return odds_ratios_summary
	
	
	# convenience functions for plots/accuracies (implemented in tuner_utils and tuner_plots)
	def create_metrics_plots(self, *args, **kwargs) -> None:
		'''
		Calculates which metrics to plot using identical y axes and which to plot on the same figure, and plots metrics
		
			params:
				*args (list)	: passed to tuner_plots.create_metrics_plot-
				**kwargs (dict)	: passed to tuner_plots.create_metrics_plots
		'''
		ignore_for_ylims = self.tokens_to_mask
		dont_plot_separately = []
		if self.exp_type == 'newverb':
			ignore_for_ylims += list(itertools.chain(*[[arg_type, f'({arg_type})'] for arg_type in list(self.cfg.tuning.args.keys())])) + \
								list(itertools.chain(*[self.cfg.tuning.args[arg_type] for arg_type in self.cfg.tuning.args]))
			
			for arg_type in self.cfg.tuning.args:
				for arg in self.cfg.tuning.args[arg_type]:
					dont_plot_separately.append(m for m in metrics.metric.unique() if m.startswith(f'{arg} ({arg_type})'))
		
		tuner_plots.create_metrics_plots(*args, **kwargs, ignore_for_ylims=ignore_for_ylims, dont_plot_separately=dont_plot_separately)
	
	def create_cossims_plot(self, *args, **kwargs) -> None:
		'''
		Calls tuner_plots.create_cossims_plot
		
			params:
				*args (list)	: passed to tuner_plots.create_cossims_plot
				**kwargs (dict)	: passed to tuner_plots.create_cossims_plot
		'''
		tuner_plots.create_cossims_plot(*args, **kwargs)
	
	def create_tsnes_plots(self, *args, **kwargs) -> None:
		'''
		Calls tuner_plots.create_tsnes_plots
		
			params:
				*args (list)	: passed to tuner_plots.create_tsnes_plot
				**kwargs (dict)	: passed to tuner_plots.create_tsnes_plot
		'''
		tuner_plots.create_tsnes_plots(*args, **kwargs)
	
	def create_odds_ratios_plots(self, *args, **kwargs) -> None:
		'''
		Calls tuner_plots.create_odds_ratios_plots
		
			params:
				*args (list)	: passed to tuner_plots.create_odds_ratios_plots
				**kwargs (dict)	: passed to tuner_plots.create_odds_ratios_plots
		'''
		tuner_plots.create_odds_ratios_plots(*args, **kwargs)
	
	def get_odds_ratios_accuracies(self, *args, **kwargs) -> pd.DataFrame:
		'''
		Calls tuner_utils.get_odds_ratios_accuracies, returns a dataframe containing accuracy information
		
			params:
				*args (list)		: passed to tuner_utils.get_odds_ratios_accuracies
				**kwargs (list)		: passed to tuner_utils.get_odds_ratios_accuracies
			
			returns:
				acc (pd.DataFrame)	: dataframe containing accuracy information for each novel token in each sentence
		'''
		return tuner_utils.get_odds_ratios_accuracies(*args, **kwargs)
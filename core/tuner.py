# tuner.py
# 
# Tunes a model on training data and provides functions for evaluation
import os
import re
import sys
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
import torch.nn.functional as F

from math import floor, sqrt
from copy import deepcopy
from tqdm import trange, tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import *
from .mixout.module import MixLinear
from omegaconf import DictConfig, OmegaConf, open_dict, ListConfig
from contextlib import suppress

from transformers import logging as lg
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.manifold import TSNE

from . import tuner_plots
from . import tuner_utils
from . import kl_baseline_loss
from .tuner_utils import none
from .mixout.module import MixLinear

lg.set_verbosity_error()

log = logging.getLogger(__name__)

class Tuner:
	
	# START Computed Properties
	
	@property
	def mixed_tuning_data(self) -> Dict:
		'''Returns a dict with roberta-style masked sentences, inputs, and masked token indices.'''
		return self.__get_formatted_datasets(masking_style=self.masked_tuning_style)[self.tuning]
	
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
			assert tuner_utils.verify_tokens_exist(self.tokenizer, token_id), f'Added token {token} was not added correctly!'
			added_token_weights[token] = self.word_embeddings[token_id,:].clone()
		
		return added_token_weights
	
	# END Computed Properties
	
	
	# START Private Functions
	
	# during tuning
	def __create_inputs(
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
		
		to_mask = [self.tokenizer.convert_tokens_to_ids(token) if isinstance(token,str) else token for token in tuner_utils.listify(to_mask)]
		assert not any(token_id == self.unk_token_id for token_id in to_mask), 'At least one token to mask is not in the model vocabulary!'
		
		inputs = self.__format_data_for_tokenizer(sentences)
		if not tuner_utils.verify_tokenization_of_sentences(self.tokenizer, inputs, self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs):
			log.error('Added tokens affected the tokenization of sentences!')
			return
		
		inputs 					= self.tokenizer(inputs, return_tensors='pt', padding=True).to(self.device)
		labels 					= inputs['input_ids'].clone().detach()
		
		to_mask_indices 		= [np.where([token_id in to_mask for token_id in sentence])[-1].tolist() for sentence in inputs['input_ids']]
		to_mask_ids				= [[int(token_id) for token_id in sentence if token_id in to_mask] for sentence in inputs['input_ids']]
		
		masked_token_indices 	= []
		for token_ids, token_locations in zip(to_mask_ids, to_mask_indices):
			masked_token_indices_for_sentence = {}
			for token_id, token_location in zip(token_ids, token_locations):
				token = self.tokenizer.convert_ids_to_tokens(token_id)
				
				num = 1
				while token in masked_token_indices_for_sentence:
					token = f'{token}{num}'
				masked_token_indices_for_sentence.update({token: token_location})
			
			masked_token_indices.append(masked_token_indices_for_sentence)
		
		masked_inputs 	= deepcopy(inputs)
		if masking_style != 'none':
			for i, (tokenized_sentence, indices) in enumerate(zip(inputs['input_ids'], to_mask_indices)):
				masked_inputs['input_ids'][i] = tuner_utils.mask_input(
													inputs=masked_inputs['input_ids'][i], 
													indices=indices, 
													masking_style=masking_style, 
													tokenizer=self.tokenizer
												)
		
		return masked_inputs, labels, masked_token_indices
	
	def __get_formatted_datasets(
		self, 
		mask_args: bool = False, 
		masking_style: str = None, 
		datasets: Union[Dict, DictConfig] = None,
		eval_cfg: DictConfig = None,
	) -> Dict:
		'''
		Returns a dictionary with formatted inputs, labels, and mask_token_indices (if they exist) for datasets
		
			params:
				mask_args (bool)		: whether to mask arguments (only used in newverb experiments)
				masking_style (str)		: 'always' to mask all tokens in [self.tokens_to_mask] (+ arguments if mask_args)
									  	  'none' to return unmasked data
									  	  'eval' is equivalent to 'always' for newarg experiments. for new verb experiments, it produces a dataset where each unique masked sentence is used only once
									  	  this speeds up eval since it means we don't need to run the model on identical sentences repeatedl
				datasets (Dict-like)	: which datasets to generated formatted data for
				eval_cfg (DictConfig)	: used during evaluation to add generalization arguments to the newverb experiments
			
			returns:
				formatted_data (dict)	: a dict with, for each dataset, sentences, inputs, (+ masked_token_indices if masking_style != 'none')
		'''
		to_mask = self.tokenizer.convert_tokens_to_ids(self.tokens_to_mask)
		
		if datasets is None:
			if self.exp_type != 'newverb' or masking_style != 'eval':
				datasets = {self.tuning: {'data': OmegaConf.to_container(self.cfg.tuning.data)}}
			else:
				datasets = {self.tuning: {'data': OmegaConf.to_container(self.original_verb_tuning_data)}}
		
		if ((not np.isnan(self.mask_args) and self.mask_args) or mask_args) and self.exp_type == 'newverb':
			if masking_style != 'eval':
				args 	=  tuner_utils.flatten(list(self.args.values()))
				to_mask += self.tokenizer.convert_tokens_to_ids(args)
				assert none(token_id == self.mask_token_id for token_id in to_mask), 'The selected arguments were not tokenized correctly!'
			else:
				args 					= self.__format_strings_with_tokens_for_display(self.args)
				if eval_cfg is not None and 'added_args' in eval_cfg.data and self.args_group in eval_cfg.data.added_args:
					args 				= {arg_type: args[arg_type] + eval_cfg.data.added_args[self.args_group][arg_type] for arg_type in args}
				
				# we're manually replacing the arguments with mask tokens and adding them back later to speed up evaluation
				# this is how we're evaluating anyway, so it doesn't make sense to ask the model for its thoughts on the same
				# input 36 different times.
				to_mask 				= self.mask_token # + self.tokens_to_mask. not masking the new verb is more like the eval context, so the metrics are more useful
				# adding [verb] here allows us to find the mask position of the verb when getting topk predictions during eval
				gf_regex 				= re.sub(r'(\[|\])', '\\ \\1', '|'.join([arg for arg in args])).replace(' ', '') + r'|\[verb\]'
				masked_arg_indices 		= {dataset: [re.findall(rf'({gf_regex})', sentence) for sentence in datasets[dataset]['data']] for dataset in datasets}
				for dataset in datasets:
					for j, _ in enumerate(datasets[dataset]['data']):
						for gf in list(args.keys()) + ['[verb]']:
							datasets[dataset]['data'][j] = datasets[dataset]['data'][j].replace(gf, self.mask_token)
		
		# this is so we don't overwrite the original datasets as we do this
		datasets = deepcopy(datasets)
		
		# we need to convert everything to primitive types to feed them to the tokenizer
		if isinstance(datasets, DictConfig):
			datasets = OmegaConf.to_container(datasets)
		
		formatted_data = {}
		for dataset in datasets:
			inputs, labels, masked_token_indices = self.__create_inputs(sentences=datasets[dataset]['data'], to_mask=to_mask, masking_style=masking_style if masking_style != 'eval' else 'always')
			formatted_data.update({dataset: {'sentences': datasets[dataset]['data'], 'inputs': inputs, 'labels': labels, 'masked_token_indices': masked_token_indices}})
		
		# if we are doing a newverb experiment, we only gave the model the masked data once to avoid reevaluating it redundantly.
		# now we determine the correct mapping of grammatical functions to masked token positions for evaluation
		if ((not np.isnan(self.mask_args) and self.mask_args) or mask_args) and masking_style == 'eval' and self.exp_type == 'newverb':
			for dataset in formatted_data:
				gf_masked_token_indices = []
				masked_token_indices = [{k: v for k, v in masked_token_indices.items() if self.mask_token in k} for masked_token_indices in formatted_data[dataset]['masked_token_indices']]
				for token_indices_map, gfs in zip(masked_token_indices, masked_arg_indices[dataset]):
					gf_masked_token_indices.append({gfs[i]: token_indices_map[key] for i, key in enumerate(token_indices_map)})
				
				formatted_data[dataset]['masked_token_indices'] = gf_masked_token_indices
		
		return formatted_data
	
	def __generate_filled_verb_data(self, sentences: List[str], to_replace: Dict[str,List[str]]) -> List[str]:
		'''
		Generates sentences with every combination of arguments in a newverb experiment
		
			params:
				sentences (list)			: a list of sentences containing placeholders for grammatical functions in self.args.keys()
				to_replace (dict)			: a dictionary mapping grammatical function placeholders to tokens they are to be replaced with
			
			returns:
				generated_sentences (list)	: a list of sentences generated by replacing every grammatical function
											  with every argument of that type, and with every other grammatical function replaced
											  by every argument of that type
		'''
		if not self.exp_type == 'newverb':
			return sentences
		
		args, values 				= zip(*to_replace.items())
		replacement_combinations 	= itertools.product(*list(to_replace.values()))
		to_replace_mappings			= [dict(zip(args, t)) for t in replacement_combinations]
		
		generated_sentences = []
		for mapping in to_replace_mappings:
			for sentence in sentences:
				for arg, value in mapping.items():
					sentence = sentence.replace(arg, value)
				
				generated_sentences.append(sentence)
		
		return generated_sentences
	
	# formatting of results
	def __format_strings_with_tokens_for_display(self, data: 'any') -> 'any':
		'''
		Formats strings containing tokenizer-formatted tokens for display
		
			params:
				data (any)	: a data structure (possibly infinitely nested) containing some string(s)
			
			returns:
				data structured in the same way as the input data, with added tokens formatted for display
		'''
		tokens_to_format = self.tokens_to_mask
		if self.exp_type == 'newverb':
			# need to use deepcopy so we don't add newverb arguments to the tokens to mask attribute here
			tokens_to_format = deepcopy(tokens_to_format)
			tokens_to_format += list(self.args.values())
			tokens_to_format = tuner_utils.flatten(tokens_to_format)
		
		# in newverb experiments, we only want to uppercase the added tokens, not the argument tokens
		tokens_to_uppercase = self.tokens_to_mask
		
		return tuner_utils.format_strings_with_tokens_for_display(
			data=data, 
			tokenizer_tokens=tokens_to_format, 
			tokens_to_uppercase=tokens_to_uppercase, 
			model_name=self.model_name, 
			string_id=self.string_id
		)
	
	def __format_data_for_tokenizer(self, data: str) -> str:
		'''
		Formats a data structure with strings (including mask tokens) in a way that makes it possible to use with self's tokenizer
		
			params:
				data (str) : a (possibly nested) data structure containing strings
			
			returns:
				output in the same structure as data, with strings formatted according to tokenizer requirements
		'''
		return tuner_utils.format_data_for_tokenizer(data=data, mask_token=self.mask_token, string_id=self.string_id, remove_punct=self.strip_punct)
		
	def __format_tokens_for_tokenizer(self, tokens: 'any') -> 'any':
		'''Pipelines formatting tokens for models'''
		formatted_tokens	 = self.__format_data_for_tokenizer(tokens)
		if self.model_name == 'roberta':
			formatted_tokens = tuner_utils.format_roberta_tokens_for_tokenizer(formatted_tokens)
		else:
			formatted_tokens = tuner_utils.apply_to_all_of_type(formatted_tokens, str, lambda x: x if not x.startswith('^') else None)
		
		return formatted_tokens
	
	def __add_hyperparameters_to_summary_df(self, df: pd.DataFrame) -> pd.DataFrame:
		'''
		Adds hyperparameters to a summary dataframe.
			
			params:
				df (pd.DataFrame): a dataframe to add hyperparameters to
			
			returns:
				df (pd.DataFrame): the dataframe with hyperparameters added in columns
		'''
		exclude = [
			'mask_token', 'mask_token_id', 'unk_token_id', 
			'save_full_model', 'checkpoint_dir', 'load_full_model', 'device',
			'use_kl_baseline_loss', 'original_cwd',
		]
		
		included_vars = [var for var in vars(self) if not var in exclude]
		included_vars = [var for var in included_vars if type(vars(self)[var]) in (str,int,float,bool,np.nan)]
		sorted_vars = sorted([var for var in included_vars], key=lambda item: re.sub(r'^(model)', '0\\1', item))
		
		for var in sorted_vars:
			df[var] = vars(self)[var]
		
		return df
	
	# evaluation
	def __log_debug_predictions(
		self, 
		epoch: int, 
		total_epochs: int = 0,
		additional_sentences: List[str] = None
	) -> Dict:
		'''
		Prints a log message used during debugging. The log displays predictions for a baseline sentence,
		as well as sentences taken from or similar to the fine-tuning data, depending on the experiment type.
		
			params:
				epoch (int)			: which epoch the model is at
				total_epochs (int)	: the total number of epochs the model was trained for, or max_epochs
			
			returns:
				results (Dict)		: a dictionary containing the sentence, model inputs, top prediction, and model output for each passed sentence
		'''
		additional_sentences = [] if additional_sentences is None else additional_sentences
		
		if self.exp_type == 'newverb':
			sentences = [
				f'The local {self.mask_token} will step in to help.',
				f'The {self.mask_token} will {self.__format_strings_with_tokens_for_display(self.tokens_to_mask[0])} the {self.mask_token}.',
				f'The {self.__format_strings_with_tokens_for_display(self.args["[subj]"][0])} will {self.mask_token} the {self.__format_strings_with_tokens_for_display(self.args["[obj]"][0])}.',
				f'The {self.mask_token} will {self.mask_token} the {self.mask_token}.',
			]
		else:
			sentences = [f'The local {self.mask_token} will step in to help.'] + self.tuning_data['sentences'][:2] + self.tuning_data['sentences'][-2:]
			for i, sentence in enumerate(sentences):
				for token in self.__format_strings_with_tokens_for_display(self.tokens_to_mask):
					sentence = sentence.replace(token, self.mask_token)
				
				sentences[i] = sentence
		
		sentences += additional_sentences
		
		log.info('')
		results = self.predict_sentences(
			info=f'epoch {str(epoch).zfill(len(str(total_epochs)))}', 
			sentences=sentences,
			output_fun=log.info
		)
		
		return results
	
	def __load_eval_predictions_data(self, eval_cfg: DictConfig) -> Dict:
		'''
		Loads the predictions data included in the eval cfg according to the settings
		
			params:
				eval_cfg (dictconfig)	: a dict config containing a list of sentences to get predictions for under eval_cfg.data.prediction_sentences
			
			returns:
				data (dict)				: a dict containing the formatted predictions sentences from the eval cfg, ready to put into the model
		'''
		if 'prediction_sentences' in eval_cfg.data:
			sentences 			= OmegaConf.to_container(eval_cfg.data.prediction_sentences)
		else:
			sentences 			= []
		
		if eval_cfg.debug:
			if self.exp_type == 'newverb':
				gfs 			= list(self.args.keys())
				debug_sentences = [
									f'The local {gfs[0]} has stepped in to help.',
									f'The {gfs[0]} has {self.__format_strings_with_tokens_for_display(self.tokens_to_mask[0])} the {gfs[1]}.',
									f'The {self.__format_strings_with_tokens_for_display(self.args["[subj]"][0])} has [verb] the {self.__format_strings_with_tokens_for_display(self.args["[obj]"][0])}.',
									f'The {gfs[0]} has [verb] the {gfs[1]}.',
								]
			else:
				debug_sentences	= [f'The local {self.__format_strings_with_tokens_for_display(self.tokens_to_mask[0])} will step in to help.'] + self.tuning_data['sentences'][:2] + self.tuning_data['sentences'][-2:]
		
			sentences 			= debug_sentences + sentences
		
		# remove duplicates while preserving order
		# this way we don't evaluate duplicates twice	
		sentences 		= list(dict.fromkeys(sentences))
		
		if sentences:
			data		= tuner_utils.unlistify(self.__get_formatted_datasets(
							mask_args=True, 
							masking_style='eval', 
							datasets={'prediction_sentences': {'data': sentences}},
							eval_cfg=eval_cfg,
						))
		else:
			data 		= {}
		
		return data
	
	def __get_eval_predictions(
		self,
		summary: pd.DataFrame,
		eval_cfg: DictConfig,
		output_fun: Callable = print
	) -> Dict:
		'''
		Gets and returns model predictions for debug sentences and additional sentences.
		
			params:
				summary (pd.DataFrame)	: a dataframe containing information used to determine the prediction message and the output file name
				eval_cfg (dictconfig)	: a dictconfig containing the sentences to get and save predictions for
			
			returns:
				results (Dict)			: a dictionary containing the input, model input, top prediction, and model outputs for each sentence
		'''
		prediction_data = self.__load_eval_predictions_data(eval_cfg=eval_cfg)
		
		if prediction_data:
			results 	= self.predict_sentences(
							info=f'epoch {str(tuner_utils.multiplator(summary.eval_epoch)).zfill(len(str(tuner_utils.multiplator(summary.total_epochs))))}',
							sentences=prediction_data['sentences'],
							output_fun=output_fun
						)
			
			output_fun('')
			
			return results
		else:
			return {}
	
	def __collect_results(
		self, 
		outputs: 'MaskedLMOutput',
		masked_token_indices: List[Dict[str,int]],
		sentences: List[str] = None,
		eval_groups: Dict = None,
	) -> List:
		'''
		Returns a list of dicts with results based on model outputs
			
			params:
				outputs (MaskedLMOutput) 	: model outputs
				masked_token_indices (list)	: list of dicts with mappings from string token to integer positions 
										  	  of the masked token locations corresponding to that token 
										  	  for each sentence in the outputs
				sentences (list)			: the sentences that were used to generate the outputs.
											  if no sentences are provided, it is assumed that the outputs were generated by the model's tuning data
				eval_groups (dict)			: dict mapping a token group to a list of tokens to evaluate
			
			returns:
				results (list)				: list of dicts with results for each token for each sentence
		'''
		def get_output_metrics(outputs: 'MaskedLMOutput') -> Tuple:
			logits 				= outputs.logits
			probabilities 		= F.softmax(logits, dim=-1)
			log_probabilities 	= F.log_softmax(logits, dim=-1)
			surprisals 			= -(1/torch.log(torch.tensor(2.))) * F.log_softmax(logits, dim=-1)
			predicted_ids 		= torch.argmax(log_probabilities, dim=-1)
			
			return logits, probabilities, log_probabilities, surprisals, predicted_ids
		
		if eval_groups is None:
			eval_groups = self.tokens_to_mask
		
		if isinstance(eval_groups,list):
			eval_groups = {token: [token] for token in eval_groups}
		
		tokens_to_type_labels = {token: label for label in eval_groups for token in eval_groups[label]}
		assert not len(tokens_to_type_labels.keys()) < len(eval_groups.values()), 'Tokens should only be used in a single eval group!'
		
		eval_group_masked_token_indices = [
			{
				k: v 
				for k, v in sentence_masked_token_indices.items()
				if k in tokens_to_type_labels.keys() or (k in tokens_to_type_labels.values() and self.exp_type == 'newverb')
			} 
			for sentence_masked_token_indices in masked_token_indices
		]
		
		if sentences is None:
			sentences = self.tuning_data['sentences']
		
		sentences = tuner_utils.listify(sentences)
		
		results = []
		metrics = tuple(zip(eval_group_masked_token_indices, sentences, *get_output_metrics(outputs)))
		
		for sentence_num, (token_indices, sentence, logits, probs, logprobs, surprisals, predicted_ids) in enumerate(metrics):
			for token in token_indices:
				# this lets us get away with doing fewer eval passes during new verb experiment on the masked data,
				# since we don't have to run functionally identically masked sentences for each combination of arguments
				# when the model never actually sees those arguments
				if self.exp_type == 'newverb' and token in self.args.keys():
					tokens = self.args[token]
				else:
					tokens = [token]
				
				for tok in tokens:
					token_id = self.tokenizer.convert_tokens_to_ids(tok)
					
					assert token_id != self.unk_token_id, f'Token "{tok}" was not tokenized correctly! Try using something different instead.'
					
					exp_logprob = logprobs[token_indices[token],token_id]
					
					common_args = {
						'arg type'			: tokens_to_type_labels[tok],
						'token id'			: token_id,
						'token'				: tok,
						'sentence'			: sentence,
						'sentence num'		: sentence_num,
						'predicted sentence': self.tokenizer.decode(predicted_ids),
						'predicted ids'		: ' '.join([str(i.item()) for i in predicted_ids]),
						'logit'				: logits[token_indices[token],token_id],
						'probability'		: probs[token_indices[token],token_id],
						'log probability'	: exp_logprob,
						'surprisal'			: surprisals[token_indices[token],token_id],
					}
					
					if self.exp_type == 'newverb':
						common_args.update({'args group': self.args_group})
					
					# we only want to consider other masked positions that contain tokens in the eval groups for odds ratios
					# for new verbs, we want to consider the other positions
					other_eval_tokens = [
						(other_token, token_index)
						for other_token, token_index in token_indices.items()
						if 	not other_token == token 
						   	and (
						   			(
										other_token in tokens_to_type_labels
										and tokens_to_type_labels[other_token] in tuner_utils.flatten(list(eval_groups.keys()))
									)
									or
									(
										self.exp_type == 'newverb'
										and other_token in eval_groups.keys()
									)
							)
					]
					
					if other_eval_tokens:
						for other_token, other_token_index in other_eval_tokens:
							logprob 	= logprobs[other_token_index,token_id]
							odds_ratio 	= exp_logprob - logprob
							
							positions = sorted(list(token_indices.keys()), key=lambda token: token_indices[token])
							positions = {p: positions.index(p) + 1 for p in positions}	
							
							if self.exp_type == 'newverb' and other_token in eval_groups.keys():
								ratio_name = f'{tokens_to_type_labels[tok]}/{other_token}'
							else:
								ratio_name = f'{tokens_to_type_labels[tok]}/{tokens_to_type_labels[other_token]}'
							
							results.append({
								'odds ratio'			: odds_ratio,
								'ratio name'			: ratio_name,
								'position ratio name'	: f'position {positions[token]}/position {positions[other_token]}',
								**common_args
							})
					else:
						results.append({**common_args})
		
		return results
	
	def __restore_original_random_seed(self) -> None:
		'''Restores the original random seed used to generate weights for the novel tokens'''
		if hasattr(self, 'random_seed'):
			return
		
		for f in ['tune.log', 'weights.pkl.gz']:
			path = os.path.join(self.checkpoint_dir, f)
			if not os.path.isfile(path):
				path = f'{os.path.sep}..{os.path.sep}'.join(os.path.split(path))
		
			with suppress(IndexError, FileNotFoundError):
				if f == 'tune.log':
					with open(path, 'r') as logfile_stream:
						logfile = logfile_stream.read()
					
					self.random_seed = int(re.findall(r'Seed set to ([0-9]*)\n', logfile)[0])
					break
				elif f == 'weights.pkl.gz':
					with gzip.open(path, 'rb') as weightsfile_stream:
						weights = pkl.load(weightsfile_stream)
			
					self.random_seed = weights['random_seed']
					break
		
		if not hasattr(self, 'random_seed'):
			log.error(f'Original random seed not found in log file or weights file in {os.path.split(path)[0]}!')
			
	def __load_format_dataset(
		self, 
		dataset_loc: str,
		split: str = 'train', 
		data_field: str = 'text',
		n_examples: int = None
	) -> Tuple:
		'''
		Loads and formats a huggingface dataset for use with the Tuner
		
			params:
				dataset_loc (str)		: the location of the dataset
				data_field (str)		: the field in the dataset that contains the actual examples
				string_id (str)			: a string_id corresponding to a huggingface pretrained tokenizer
										  used to determine appropriate formatting
				fmt (str)				: the file format the dataset is saved in
				n_examples (int)		: how many (random) examples to draw from the dataset
										  if not set, the full dataset is returned
			
			returns:
				dataset (Dataset)		: a dataset that has been formatted for use with the tokenizer,
										  possible with punctuation stripped
		'''
		return tuner_utils.load_format_dataset(
			dataset_loc 		= dataset_loc,
			split 				= split,
			data_field 			= data_field,
			n_examples 			= n_examples,
			string_id 			= self.string_id, 
			remove_punct 		= self.strip_punct,
			tokenizer_kwargs 	= self.cfg.model.tokenizer_kwargs,
		)
	
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
		
		data 				= self.load_eval_file(eval_cfg)
		
		if eval_cfg.data.exp_type == 'newverb':
			summary_zero 		= self.get_odds_ratios_summary(epoch=0, eval_cfg=eval_cfg, data=data)
			predictions_zero 	= {0: self.__get_eval_predictions(summary=summary_zero, eval_cfg=eval_cfg, output_fun=log.info)}
		
		summary				= self.get_odds_ratios_summary(epoch=eval_cfg.epoch, eval_cfg=eval_cfg, data=data)
		predictions 		= {tuner_utils.multiplator(summary.eval_epoch): self.__get_eval_predictions(summary=summary, eval_cfg=eval_cfg, output_fun=log.info)}
		
		if eval_cfg.data.exp_type == 'newverb':
			summary 		= pd.concat([summary_zero, summary], ignore_index=True)
			summary 		= add_odds_ratios_differences_to_summary(summary)
			predictions 	= {**predictions_zero, **predictions}
		
		file_prefix = tuner_utils.get_file_prefix(summary)
		
		log.info(f'Saving to "{os.getcwd().replace(self.original_cwd, "")}"')
		summary.to_pickle(f'{file_prefix}-odds_ratios.pkl.gz')
		
		# tensors are saved as text in csv, but we want to save them as numbers
		summary_csv = summary.copy()
		for c in ['odds_ratio', 'odds_ratio_pre_post_difference']:
			if c in summary_csv.columns:
				summary_csv[c] = summary_csv[c].astype(float)
		
		summary_csv.to_csv(f'{file_prefix}-odds_ratios.csv.gz', index=False, na_rep='NaN')
		
		if not eval_cfg.topk_mask_token_predictions:
			with open_dict(eval_cfg):
				eval_cfg.topk_mask_token_predictions = len(tuner_utils.flatten(list(self.args.values()))) if self.exp_type == 'newverb' else 20
		
		if any(predictions[epoch] for epoch in predictions):
			if eval_cfg.debug:
				save_predictions 								= deepcopy(predictions)
				for epoch in save_predictions:
					save_predictions[epoch]['model_inputs'] 	= {k: v.clone().detach().cpu() for k, v in save_predictions[epoch]['model_inputs'].items()}
					save_predictions[epoch]['outputs'].logits 	= save_predictions[epoch]['outputs'].logits.clone().detach().cpu()
				
				with gzip.open(f'{file_prefix}-predictions.pkl.gz', 'wb') as out_file:
					pkl.dump(save_predictions, out_file)
			
			topk_mask_token_predictions = self.get_topk_mask_token_predictions(predictions=predictions, eval_cfg=eval_cfg)
			topk_mask_token_predictions = tuner_utils.transfer_hyperparameters_to_df(summary, topk_mask_token_predictions)
			topk_mask_token_predictions.to_csv(f'{file_prefix}-predictions.csv.gz', index=False, na_rep='NaN')
			
		cossims_args 		= dict(topk=eval_cfg.k)
		if eval_cfg.data.exp_type == 'newarg':
			cossims_args.update(dict(targets=eval_cfg.data.masked_token_targets))
		
		predicted_roles 	= {v: k for k, v in eval_cfg.data.eval_groups.items()}
		target_group_labels = {k: v for k, v in eval_cfg.data.masked_token_target_labels.items()} if 'masked_token_target_labels' in eval_cfg.data else {}
		
		groups 				= ['predicted_arg', 'target_group']
		group_types 		= ['predicted_role', 'target_group_label']
		group_labels 		= [predicted_roles, target_group_labels]
		cossims_args.update(dict(groups=groups, group_types=group_types, group_labels=group_labels))
		
		cossims 			= self.get_cossims(**cossims_args)
		cossims 			= tuner_utils.transfer_hyperparameters_to_df(summary, cossims)
	
		if not cossims[~cossims.target_group.str.endswith('most similar')].empty:
			log.info('Creating cosine similarity plots')
			self.create_cossims_plot(cossims)
		
		cossims.to_csv(f'{file_prefix}-cossims.csv.gz', index=False, na_rep='NaN')
		
		log.info('Creating t-SNE plot(s)')
		tsne_args 			= dict(
								n=eval_cfg.num_tsne_words, 
								n_components=2, 
								random_state=0, 
								learning_rate='auto', 
								init='pca'
							)
		if 'masked_token_targets' in eval_cfg.data:
			tsne_args.update(dict(targets=eval_cfg.data.masked_token_targets))
		
		if 'masked_token_target_labels' in eval_cfg.data:
			tsne_args.update(dict(target_group_labels=target_group_labels))
		
		tsnes 				= self.get_tsnes(**tsne_args)
		tsnes 				= tuner_utils.transfer_hyperparameters_to_df(summary, tsnes)
		self.create_tsnes_plots(tsnes)
		
		tsnes.to_csv(f'{file_prefix}-tsnes.csv.gz', index=False, na_rep='NaN')
		
		if eval_cfg.data.exp_type == 'newverb':
			odds_ratios_plot_kwargs = dict(
										scatterplot_kwargs=dict(
											text='token', 
											text_color={
												'colname': 'token_type', 
												'eval only': 'blue', 
												'tuning': 'black'
											}
										)
									)
			log.info('Creating odds ratios differences plots')
			self.create_odds_ratios_plots(summary, eval_cfg, plot_diffs=True, **odds_ratios_plot_kwargs)
		else:
			odds_ratios_plot_kwargs = {}
		
		log.info('Creating odds ratios plots')
		self.create_odds_ratios_plots(summary, eval_cfg, **odds_ratios_plot_kwargs)
		
		if eval_cfg.data.exp_type == 'newverb':
			acc = self.get_odds_ratios_accuracies(summary, eval_cfg, get_diffs_accuracies=True)
			acc.to_csv(f'{file_prefix}-accuracies_diffs.csv.gz', index=False, na_rep='NaN')
		
		acc = self.get_odds_ratios_accuracies(summary, eval_cfg)
		acc = tuner_utils.transfer_hyperparameters_to_df(summary, acc)
		acc.to_csv(f'{file_prefix}-accuracies.csv.gz', index=False, na_rep='NaN')
		
		# we should only do the comparison if anything has been unfrozen.
		# otherwise, it doesn't make sense since the results will be the same.
		if eval_cfg.comparison_dataset:
			if isinstance(self.unfreezing,float) and np.isnan(self.unfreezing):
				log.warning('A baseline comparison dataset was provided, but the model parameters were not unfrozen!')
				log.warning('Because probability distributions will be identical when comparing without new tokens, this is probably not what you meant to do.')
				log.warning('Proceeding anyway, but maybe change your command next time?')
			
			log.info('Comparing model distributions to baseline')
			kl_divs = self.compare_model_performance_to_baseline(eval_cfg)
			kl_divs = tuner_utils.transfer_hyperparameters_to_df(summary, kl_divs)
			kl_divs.to_csv(f'{file_prefix}-kl_divs.csv.gz', index=False, na_rep='NaN')
			
			log.info('Creating KL divergences plot')
			self.create_kl_divs_plot(kl_divs)
		
		log.info('Evaluation complete')
		print('')
	
	# END Private Functions
	
	
	# START Class Functions
	
	def __init__(self, cfg_or_path: Union[DictConfig,str], use_gpu: bool = None) -> 'Tuner':
		'''
		Creates a tuner object, loads argument/dev sets, and sets class attributes
		
			params:
				cfg_or_path (DictConfig or str)	: if dictconfig, a dictconfig specifying a Tuner configuration
												  if str, a directory created by a Tuner when tune() is run
				use_gpu	(bool)					: used during evaluation. useful when loading a model trained on cpu on gpu, or vice versa
			
			returns:
				the created Tuner object
		'''
		def load_args() -> None:
			'''Loads correct argument set for newverb experiments'''
			if self.cfg.tuning.exp_type == 'newverb':
				with open_dict(self.cfg):
					self.cfg.tuning.args = self.cfg.tuning[self.cfg.model.friendly_name] if self.cfg.tuning.which_args == 'model' else self.cfg.tuning[self.cfg.tuning.which_args]
		
		def load_dev_sets() -> None:
			'''Loads dev sets using specified criteria'''
			if self.cfg.dev == 'best_matches':
				criteria = self.cfg.tuning.name.split('_')
				candidates = os.listdir(os.path.join(self.original_cwd, 'conf', 'tuning'))
				candidates = [candidate.replace('.yaml', '').split('_') for candidate in candidates]
				
				# Find all the tuning sets that differ from the current one by one parameter, and grab those as our best matches
				candidates = [candidate for candidate in candidates if candidate[0] == criteria[0]]
				candidates = [candidate for candidate in candidates if len(set(criteria) - set(candidate)) == 1]
				
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
					dev = OmegaConf.load(os.path.join(self.original_cwd, 'conf', 'tuning', dev_set + '.yaml'))
					if not all(token in self.cfg.tuning.to_mask for token in dev.to_mask):
						log.warn(f'Not all dev tokens to mask from {dev_set} are in the training set! This is probably not what you intended. Removing this dataset from the dev data.')
					
					self.cfg.dev.update({dev.name: dev})
		
		def setattrs() -> None:
			'''Sets static model attributes'''
			log.info(f'Initializing Model:\t{self.cfg.model.base_class} ({self.cfg.model.string_id})')
			self.model 								= AutoModelForMaskedLM.from_pretrained(self.cfg.model.string_id, **self.cfg.model.model_kwargs)
			self.model.to(self.device)
			
			resolved_cfg = OmegaConf.to_container(self.cfg, resolve=True)
			for k, v in resolved_cfg['hyperparameters'].items():
				
				setattr(self, k, v)
			
			if not isinstance(self.unfreezing,int):
				unfreezing_epochs_per_layer 		= re.findall(r'[0-9]+', self.unfreezing) if 'gradual' in self.unfreezing else None
				unfreezing_epochs_per_layer			= int(unfreezing_epochs_per_layer[0]) if unfreezing_epochs_per_layer else 1
				self.unfreezing 					= self.unfreezing if 'mixout' in self.unfreezing else re.sub(r'[0-9]*', '', self.unfreezing) if not self.unfreezing == 'none' else np.nan
				if 'mixout' in str(self.unfreezing):
					assert re.search(r'\.[0-9]+$', self.unfreezing), 'You must provide a probability for mixout freezing!'
				
				self.unfreezing_epochs_per_layer 	= self.unfreezing_epochs_per_layer if self.unfreezing == 'gradual' else np.nan
			
			self.model_name 						= self.model.config.model_type
			self.model_id 							= os.path.split(self.checkpoint_dir)[-1] + '-' + self.model_name[0]
			self.string_id 							= self.model.config.name_or_path
			
			# this is temporary so we can format the new tokens according to the model specifications
			# the formatting functions fixes mask tokens that are converted to lower case, so it needs something to refer to
			# this is redefined a little ways down immediately after initializing the tokenizer
			self.mask_token 						= ''
			
			tokens 									= self.__format_tokens_for_tokenizer(self.cfg.tuning.to_mask)
			if self.model_name == 'roberta':
				tokens 								= tuner_utils.format_roberta_tokens_for_tokenizer(tokens)
			
			self.tokens_to_mask						= tokens
			
			log.info(f'Initializing Tokenizer:\t{self.cfg.model.tokenizer}   ({self.cfg.model.string_id})')
			self.tokenizer 							= tuner_utils.create_tokenizer_with_added_tokens(self.cfg.model.string_id, self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs)
			self.model.resize_token_embeddings(len(self.tokenizer))
			
			self.mask_token 						= self.tokenizer.mask_token
			self.mask_token_id 						= self.tokenizer.convert_tokens_to_ids(self.mask_token)
			self.unk_token_id 						= self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
			
			self.tuning 							= self.cfg.tuning.name
			self.exp_type 							= self.cfg.tuning.exp_type
			self.mask_args							= self.mask_args if self.exp_type == 'newverb' else np.nan
			self.reference_sentence_type 			= self.cfg.tuning.reference_sentence_type
			self.masked 							= self.masked_tuning_style != 'none' 
			
			mask_args 								= True if not np.isnan(self.mask_args) and self.mask_args else False
			if self.exp_type == 'newverb':
				self.original_verb_tuning_data		= deepcopy(self.cfg.tuning.data)
				with open_dict(self.cfg):
					self.cfg.tuning.data 			= self.__generate_filled_verb_data(self.cfg.tuning.data, self.cfg.tuning.args)
				
				self.original_verb_dev_data			= deepcopy(self.cfg.dev)
				with open_dict(self.cfg.dev):	
					for dataset in self.cfg.dev:
						self.cfg.dev[dataset].data 	= self.__generate_filled_verb_data(self.cfg.dev[dataset].data, self.cfg.tuning.args)
				
				self.args 							= {k: self.__format_tokens_for_tokenizer(v) for k, v in self.cfg.tuning.args.items()}
			
			self.tuning_data 						= self.__get_formatted_datasets(masking_style='none')[self.tuning]
			self.masked_tuning_data 				= self.__get_formatted_datasets(mask_args=mask_args, masking_style='always')[self.tuning]
			self.dev_data 							= self.__get_formatted_datasets(masking_style='none', datasets=self.cfg.dev)
			self.masked_dev_data 					= self.__get_formatted_datasets(mask_args=mask_args, masking_style='always', datasets=self.cfg.dev)
			
			# even if we are not masking arguments for training, we need them for dev sets
			if self.exp_type == 'newverb':
				self.masked_argument_data 			= self.__get_formatted_datasets(mask_args=True, masking_style='eval')[self.tuning]
				self.masked_dev_argument_data 		= self.__get_formatted_datasets(mask_args=True, masking_style='eval', datasets=self.original_verb_dev_data)
				self.args_group 					= self.cfg.tuning.which_args if not self.cfg.tuning.which_args == 'model' else self.model_name
			
			if self.use_kl_baseline_loss:
				if not (isinstance(self.unfreezing,(int,float)) and np.isnan(self.unfreezing)):
					if not self.cfg.kl_loss_params.scaleby == 0:
						for k, v in self.cfg.kl_loss_params.items():
							setattr(self, ('kl_' if 'kl' not in k else '') + k, v)
					else:
						log.warning('You set "use_kl_baseline_loss=True", but set "kl_loss_params.scaleby=0"!')
						log.warning('This is no different from setting "kl_baseline_loss=False", but would use extra computation time.')
						log.warning('For this reason, not using KL baseline loss to avoid wasting time.')
						self.use_kl_baseline_loss 	= False
				else:
					log.warning('You set "use_kl_baseline_loss=True", but you are not unfreezing any model parameters!')
					log.warning('Model predictions when excluding new tokens would not change compared to baseline.')
					log.warning('For this reason, not using KL baseline loss to avoid wasting time.')
					self.use_kl_baseline_loss 		= False
		
		# this lets us load a tuner without having to go through hydra.main first.
		# it's more convenient when we want to evaluate the model interactively
		try:
			self.original_cwd 		= hydra.utils.get_original_cwd()
		except ValueError:
			self.original_cwd 		= os.getcwd()
		
		self.cfg 					= OmegaConf.load(os.path.join(self.original_cwd, cfg_or_path, '.hydra', 'config.yaml')) if isinstance(cfg_or_path, str) else cfg_or_path
		
		# allowing an optional argument to specify the gpu when calling Tuner helps us when evaluating with/without a gpu regardless of the tuning setup
		if use_gpu is not None:
			with open_dict(self.cfg):
				self.cfg.use_gpu 	= use_gpu
		
		# too little memory to use gpus locally, but we can specify to use them on the cluster with use_gpu=true
		self.device					= 'cuda' if torch.cuda.is_available() and self.cfg.use_gpu else 'cpu'
		
		if self.device == 'cuda':
			log.info(f'Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
		
		self.checkpoint_dir 		= os.path.join(self.original_cwd, cfg_or_path) if isinstance(cfg_or_path, str) else os.getcwd()
		
		# switch over to the checkpoint dir for the purpose of organizing results if we're not already in a subdirectory of it
		if not self.checkpoint_dir in os.getcwd():
			os.chdir(self.checkpoint_dir)
		
		self.save_full_model 		= False
		self.load_full_model 		= False
		
		load_dev_sets()
		load_args()
		setattrs()
		self.set_model_freezing()
	
	def __repr__(self) -> str:
		'''Return a string that eval() can be called on to create an identical Tuner object'''
		return 'tuner.Tuner(' + repr(self.cfg) + ')'
	
	def __str__(self) -> str:
		'''Return a formatted string for printing'''
		return f'Tuner object @ {self.checkpoint_dir} with config:\n' + OmegaConf.to_yaml(self.cfg, resolve=True)
	
	def __call__(self, *args: Tuple, **kwargs: Dict) -> Dict:
		'''
		Calls predict sentences to generate model predictions
		
			params:
				args (tuple)	: passed to self.predict_sentences
				kwargs (dict)	: passed to self.predict_sentences
		'''
		return self.predict_sentences(*args, **kwargs)
	
	# END Class Functions
	
	
	# main tuning functionality
	def tune(self) -> None:
		'''
		Fine-tunes the model on the provided tuning data. 
		Saves updated weights/model state, metrics, and plots of metrics to disk.
		'''
		def save_weights(weights: Dict) -> None:
			'''
			Saves dictionary of weights to disk
			
				params:
					weights (dict): a dictionary of the form {epoch_number: {token_name: weight}} to save to disk
			'''
			# always save the weights on cpu for ease of use
			# sometimes we want to evaluate on a cpu-only machine even after running on a gpu
			# this allows that
			weights_cpu = {}
			for epoch in weights:
				if isinstance(weights[epoch],dict):
					# this is when we save the weights, which are each mapped to a string
					weights_cpu[epoch] = {}
					weights_cpu[epoch] = {k: v.clone().detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in weights[epoch].items()}
				else:
					# the else is triggered when we save the random seed, which is an int
					weights_cpu[epoch] = weights[epoch]
			
			with gzip.open('weights.pkl.gz', 'wb') as f:
				pkl.dump(weights_cpu, f)
		
		def get_tuner_inputs_labels() -> Tuple:
			'''
			Returns inputs and labels used for fine-tuning
			
				returns:
					tuple consisting of inputs, labels, dev_inputs, dev_labels, masked_inputs, and masked_dev_inputs
			'''
			if self.masked:
				inputs_data 	= self.masked_tuning_data if self.masked_tuning_style == 'always' else self.mixed_tuning_data
			elif not self.masked:
				inputs_data 	= self.tuning_data
			
			labels_data 		= self.tuning_data
			
			dev_inputs_data 	= self.masked_dev_data
			dev_labels_data 	= self.dev_data
			
			inputs 				= inputs_data['inputs']
			labels 				= labels_data['inputs']['input_ids']
			
			dev_inputs 			= {dataset: dev_inputs_data[dataset]['inputs'] for dataset in dev_inputs_data}
			dev_labels 			= {dataset: dev_labels_data[dataset]['inputs']['input_ids'] for dataset in dev_labels_data}
			
			# used to calculate metrics during training
			masked_inputs 		= self.masked_tuning_data['inputs']
			masked_dev_inputs 	= {dataset: dev_inputs_data[dataset]['inputs'] for dataset in dev_inputs_data}
			
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
											epoch_metrics[metric][f'{token} ({arg_type})'] = float(torch.mean(torch.tensor([r[metric] for r in results if r['token'] == token])))
				
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
			
			results 		= self.__collect_results(outputs=outputs, masked_token_indices=self.masked_tuning_data['masked_token_indices'])
			epoch_metrics 	= get_mean_epoch_metrics(results=results)
			
			if self.exp_type == 'newverb' and masked_argument_inputs is not None:
				newverb_outputs 		= self.model(**masked_argument_inputs)
				newverb_results 		= self.__collect_results(
											outputs=newverb_outputs,
											sentences=self.masked_argument_data['sentences'],
											masked_token_indices=self.masked_argument_data['masked_token_indices'], 
											eval_groups=self.args
										)
				newverb_epoch_metrics 	= get_mean_epoch_metrics(results=newverb_results, eval_groups=self.args)
				epoch_metrics 			= {metric: {**epoch_metrics.get(metric, {}), **newverb_epoch_metrics.get(metric, {})} for metric in set(epoch_metrics.keys()).union(set(newverb_epoch_metrics.keys()))}
			
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
				model_label += f'args group: {self.args_group}, '
			
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
		
		def sort_metrics(col: pd.Series) -> pd.Series:
			'''
			Sorts metrics for display
				
				params:
					col (pd.Series)	: a column to sort
				
				returns:
					col (pd.Series)	: the column values formatted for use as a sort key
			'''
			col = deepcopy(col)
			col = col.astype(str).tolist()
			
			tokens_to_mask 	= self.__format_strings_with_tokens_for_display(deepcopy(self.tokens_to_mask))
			if hasattr(self, 'args'):
				args 		= self.__format_strings_with_tokens_for_display(deepcopy(self.args))
			else:
				args 		= {}
			
			for i, _ in enumerate(col):
				if '(train)' or '(masked, no dropout)' in col[i]:
					col[i] = re.sub(r'(.*\(train\))', '0\\1', col[i])
					col[i] = re.sub(r'(.*\(masked, no dropout\))', '1\\1', col[i])
				
				with suppress(Exception):
					_ = int(col[i])
					col[i] = str(col[i]).rjust(len(str(max(metrics.epoch))))
				
				# + 2 for loss and remaining patience
				num_tokens_to_mask 	= len(tokens_to_mask) + 2
				num_args 			= len(args)
				total_tokens		= num_tokens_to_mask + len(tuner_utils.GF_ORDER) + num_args
				arg_values			= tuner_utils.flatten(list(args.values()))
				
				num_extender		= lambda x, y = 0: str(x+y).zfill(len(str(total_tokens)))
				gf_replacer 		= lambda s, a, n: s.replace(a, num_extender(tuner_utils.GF_ORDER.index(a), n))
				
				col[i] = re.sub(r'^loss$', f'{num_extender(0)}loss', col[i])
				col[i] = re.sub(r'^remaining patience$', f'{num_extender(1)}remaining patience', col[i])
				
				# whoever decided that "subj" should be alphabetically sorted after "obj"?! >:(
				if any(token in col[i] for token in tokens_to_mask):
					col[i] = num_extender(num_tokens_to_mask) + col[i]
				elif arg_values is not None and any(arg in col[i] for arg in args) and not any(token in col[i] for token in arg_values):
					for arg in args:
						col[i] = gf_replacer(col[i], arg, num_args)
					
					# move the index to the front of the string for sorting
					col[i] = re.sub(r'(.*?)([0-9]+)(.*?)', '\\2\\1\\3', col[i])
				elif arg_values is not None and any(token in col[i] for token in arg_values):
					for gf in tuner_utils.GF_ORDER:
						col[i] = gf_replacer(col[i], gf, num_args+num_tokens_to_mask)
						
					col[i] = re.sub(r'(.*?)([0-9]+)(.*?)', '\\2\\1\\3', col[i])
			
			return pd.Series(col)
		
		def initialize_added_token_weights() -> None:
			if 'seed' in self.cfg:
				self.random_seed 					= self.cfg.seed
			elif (
					'which_args' in self.cfg.tuning and 
					self.args_group in [self.model_name, 'best_average', 'most_similar'] and 
					f'{self.model_name}_seed' in self.cfg.tuning
				):
				self.random_seed 					= self.cfg.tuning[f'{self.model_name}_seed']
			else:
				self.random_seed 					= int(torch.randint(2**32-1, (1,)))
			
			getattr(self.model, self.model_name).embeddings.word_embeddings.weight = \
				tuner_utils.reinitialize_token_weights(
					word_embeddings=self.word_embeddings, 
					tokens_to_initialize=self.tokens_to_mask,
					tokenizer=self.tokenizer,
					device=self.device, 
					seed=self.random_seed,
				)
		
		def compute_loss(outputs: 'MaskedLMOutput', eval: bool = False) -> torch.Tensor:
			'''
			Computes loss on language model outputs according to the config settings
			
				params:
					outputs (MaskedLMOutput): outputs from a masked language model
					eval (bool)				: whether the model is being evaluated or trained on the basis of
											  the loss. for eval/dev sets, we never use the KL baseline loss,
											  since we are interested in measuring overfitting to the generalization sets
				
				returns:
					loss (torch.Tensor)		: if using KL divergence loss, add KL divergence loss to the original model's outputs loss
											  KL divergence loss is defined at class initialization according to passed options
											  otherwise, return the original model loss
			'''
			if self.use_kl_baseline_loss and not eval:
				outputs.loss += self.KL_baseline_loss()
				return outputs.loss
			else:
				return outputs.loss
		
		if self.use_kl_baseline_loss:
			self.KL_baseline_loss 	= kl_baseline_loss.KLBaselineLoss(
										model 				= self.model, 
										tokenizer 			= self.tokenizer, 
										dataset 			= self.__load_format_dataset(
																dataset_loc = os.path.join(
																	self.original_cwd,
																	self.kl_dataset
																),
																split = 'train'
															),
										scaleby 			= self.kl_scaleby,
										n_examples_per_step	= self.kl_n_examples_per_step,
										masking 			= self.kl_masking,
										model_kwargs 		= self.cfg.model.model_kwargs, 
										tokenizer_kwargs 	= self.cfg.model.tokenizer_kwargs
									)
		
		initialize_added_token_weights()
		
		# store weights pre-training so we can inspect the initial status later
		saved_weights = {'random_seed': self.random_seed, 0: self.added_token_weights}
		
		if not self.tuning_data or not(isinstance(self.unfreezing,float) and np.isnan(self.unfreezing)):
			log.info(f'Saving randomly initialized weights')
			save_weights(saved_weights)
			if not self.tuning_data:	
				return
		
		# collect hyperparameters
		lr  		= self.lr
		epochs 		= self.max_epochs
		min_epochs 	= self.min_epochs
		patience 	= self.patience
		delta 		= self.delta
		optimizer 	= torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0)
		
		# store the old embeddings so we can verify that only the new ones get updated
		self.old_embeddings = self.word_embeddings.clone()
		
		inputs, labels, dev_inputs, dev_labels, masked_inputs, masked_dev_inputs = get_tuner_inputs_labels()
		
		hyperparameters_str =  f'lr={lr}, min_epochs={min_epochs}, max_epochs={epochs}, '
		hyperparameters_str += f'patience={patience}, \u0394={delta}, unfreezing={None if isinstance(self.unfreezing,(int,float)) and np.isnan(self.unfreezing) else self.unfreezing}'
		hyperparameters_str += f'{self.unfreezing_epochs_per_layer}' if self.unfreezing == 'gradual' else ''
		hyperparameters_str += f', mask_args={self.mask_args}' if not np.isnan(self.mask_args) else ''
		
		log.info(f'Training model @ "{os.getcwd().replace(self.original_cwd, "")}" ({hyperparameters_str})')
		
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
					if self.cfg.debug:
						_ = self.__log_debug_predictions(epoch, epochs)
					
					self.model.train()
					
					optimizer.zero_grad(set_to_none=True) # this is supposed to be faster than .zero_grad()
					
					if self.masked_tuning_style == 'roberta':
						inputs 		= self.mixed_tuning_data['inputs']
					
					train_outputs 	= self.model(**inputs, labels=labels)
					train_loss 		= compute_loss(train_outputs)
					train_loss.backward()
					
					tb_loss_dict, tb_metrics_dict = {}, {}
					
					record_epoch_metrics(
						epoch, train_outputs, delta, 
						self.tuning + ' (train)', metrics, 
						tb_loss_dict, tb_metrics_dict,
						best_losses, patience_counters
					)
					
					if not (self.unfreezing == 'complete' or 'mixout' in str(self.unfreezing)):
						zero_grad_for_non_added_tokens()
					
					optimizer.step()
					
					if not (self.unfreezing == 'complete' or 'mixout' in str(self.unfreezing)):
						verify_word_embeddings()
					
					saved_weights[epoch+1] = self.added_token_weights
					
					# evaluate the model on the dev set(s) and log results
					self.model.eval()
					
					with torch.no_grad():
						dev_losses = []
						for dataset in dev_inputs:
							dev_outputs = self.model(**dev_inputs[dataset], labels=dev_labels[dataset])
							dev_loss 	= compute_loss(dev_outputs, eval=True)
							dev_losses 	+= [dev_loss.item()]
							
							if self.cfg.debug and epoch == self.max_epochs - 1 and dataset == list(self.masked_dev_argument_data.keys())[0]:
								test_outputs = {k: dev_outputs[k].clone().detach().cpu() if isinstance(dev_outputs[k], torch.Tensor) else dev_outputs[k] for k in dev_outputs}
								with open('test-outputs.pkl', 'wb') as out_file:
									pkl.dump(test_outputs, out_file)
								
								test_inputs = {k: v.clone().detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in self.masked_dev_argument_data[dataset].items()}
								with open('test-inputs.pkl', 'wb') as out_file:
									pkl.dump(test_inputs, out_file)
							
							record_epoch_metrics(
								epoch, dev_outputs, delta, 
								self.cfg.dev[dataset].name + ' (dev)', metrics, 
								tb_loss_dict, tb_metrics_dict, 
								best_losses, patience_counters,
								self.masked_dev_argument_data[dataset]['inputs'] if self.exp_type == 'newverb' else None
							)
						
						# Compute loss on masked training data without dropout; this is most representative of the testing procedure, so we can use it to determine the best epoch
						no_dropout_train_outputs 	= self.model(**masked_inputs, labels=labels)
						no_dropout_train_loss 		= compute_loss(no_dropout_train_outputs, eval=True)
						
						dev_losses 					+= [no_dropout_train_loss.item()]
						
						record_epoch_metrics(
							epoch, no_dropout_train_outputs, delta, 
							self.tuning + ' (masked, no dropout)', metrics, 
							tb_loss_dict, tb_metrics_dict, 
							best_losses, patience_counters, 
							self.masked_argument_data['inputs'] if self.exp_type == 'newverb' else None
						)
						
					add_tb_epoch_metrics(epoch, writer, tb_loss_dict, dev_losses, tb_metrics_dict)
					
					if np.mean(dev_losses) < best_mean_loss - delta:
						best_mean_loss 		= np.mean(dev_losses)
						patience_counter 	= 0
						best_epoch 			= epoch + 1
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
		
		add_tb_labels(epoch, writer, tb_metrics_dict)
		writer.flush()
		writer.close()
		
		# debug
		if self.cfg.debug:
			_ = self.__log_debug_predictions(epoch, epochs)
			log.info('')
		
		if not self.save_full_model:
			# we do minus two here because we've saved the randomly initialized weights @ 0 and the random seed
			log.info(f'Saving weights for random initializations and each of {len(saved_weights)-2} training epochs')
			save_weights(saved_weights)
		else:
			log.info(f'Saving model state with lowest avg dev loss (epoch={best_epoch}) to disk')
			with open(os.path.join(self.checkpoint_dir, 'model.pt'), 'wb') as f:
				torch.save(best_model_state_dict, f)
		
		metrics 		= pd.DataFrame(metrics)
		metrics 		= self.__add_hyperparameters_to_summary_df(metrics)
		metrics.metric 	= self.__format_strings_with_tokens_for_display(metrics.metric)
		
		metrics 		= metrics.sort_values(
			by=['dataset','metric','epoch'], 
			key=lambda col: sort_metrics(col)
		).reset_index(drop=True)
		
		log.info('Saving metrics')
		metrics.to_csv(os.path.join(self.checkpoint_dir, 'metrics.csv.gz'), index=False, na_rep='NaN')
		
		log.info('Creating fine-tuning metrics plots')
		self.create_metrics_plots(metrics)
		breakpoint()
	
	def set_model_freezing(self) -> None:
		'''Freezes or unfreezes the model in accordance with the config settings'''
		if self.unfreezing == 'complete':
			self.unfreeze_all_params()
		elif 'mixout' in str(self.unfreezing):
			self.set_mixout_layers()
		elif self.unfreezing == 'all_hidden_layers':
			self.freeze_to_layer(-self.model.config.num_hidden_layers)
		elif not isinstance(self.unfreezing,int):
			self.freeze_to_layer(self.model.config.num_hidden_layers)
		elif isinstance(self.unfreezing,int):
			self.freeze_to_layer(self.unfreezing)
	
	def unfreeze_all_params(self) -> None:
		'''Unfreezes all model parameters, ensures full model is saved to disk'''
		self.save_full_model = True
		log.warning(f'You are using {self.unfreezing} unfreezing, which requires saving the full model state instead of just the weights of the new tokens.')
		log.warning('Only the initial model state and the state with the lowest mean dev loss are retained and available for evaluation.')
		
		for name, param in self.model.named_parameters():
			param.requires_grad = True
			assert param.requires_grad, f'{name} is frozen!'
	
	def set_mixout_layers(self) -> None:
		'''
		Replaces all Linear layers in self.model with MixLinear layers. 
		Sets all dropout layers to have probability 0.
		'''
		self.unfreeze_all_params()
		
		mixout_prob = float(re.search(r'(([0-9]+)?\.[0-9]+$)', self.unfreezing)[0])
		assert 0 < mixout_prob < 1, f'Mixout probability must be between 0 and 1, but you specified {mixout_prob}!'
		
		def replace_layer_for_mixout(module: nn.Module) -> nn.Module:
			'''
			Replaces a Linear layer with a Mixout Layer.
			Replaces a Dropout layer with a 0 probability Dropout Layer.
			Returns other Layers/objects unchanged.
			
				params:
					module (nn.Module)	: the module to (potentially) replace
				
				returns:
					module (nn.Module)	: a module for use with Mixout, based on the passed module
			'''
			if isinstance(module, nn.Dropout):
				return nn.Dropout(0)
			elif isinstance(module, nn.Linear):
				target_state_dict   = deepcopy(module.state_dict())
				bias				= True if module.bias is not None else False
				new_module		  	= MixLinear(
										module.in_features,
										module.out_features,
										bias,
										target_state_dict['weight'],
										mixout_prob
									).to(self.device)
				new_module.load_state_dict(target_state_dict)
				return new_module
			else:
				return module
		
		def recursive_setattr(obj: 'any', attr: str, value: 'any') -> None:
			'''
			Recursively sets attributes of child objects
			
				params:
					obj (any)	: an object to set an attribute for
					attr (str)	: a name of a (nested) attribute, where levels are indicated by a period.
								  For instance, 'bert.encoder.layer.0.attention.self.query'
								  In this case, bert has a child object, 'encoder', which has an attribute
								  'layer', and so on. Regular setattr won't work for these cases
								  because bert does not have any attribute 'encoder.layer' itself
					value (any)	: the value to set attr to
			'''
			attr = attr.split('.', 1)
			if len(attr) == 1:
				setattr(obj, attr[0], value)
			else:
				recursive_setattr(getattr(obj, attr[0]), attr[1], value)
		
		# use tuple to avoid ordereddict warning
		for name, module in tuple(self.model.named_modules()):
			if name:
				recursive_setattr(self.model, name, replace_layer_for_mixout(module))
		
		log.info(f'Linear layers have been replaced with MixLinear, mixout_prob={mixout_prob}. Dropout layers have been disabled.')
	
	def freeze_to_layer(self, n: int = None) -> None:
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
				log.warning('Only the initial model state and the state with the lowest mean dev loss are retained and available for evaluation.')
		
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
	
	
	# word embedding analysis
	def get_cossims(
		self, tokens: List[str] = None, 
		targets: Dict[str,str] = {}, topk: int = 50,
		groups: List[str] = [],
		group_types: List[str] = [],
		group_labels: List[Dict[str,str]] = {},
	) -> pd.DataFrame:
		'''
		Returns a dataframe containing information about the k most similar tokens to tokens
		If targets is provided, also includes infomation about the cossim of the tokens to 
		the targets they are mapped to
			
			params:
				tokens (list) 			: list of tokens to get cosine similarities for
				targets (dict)			: for each token in tokens, which tokens to get cosine similarities for
				topk (int)				: how many of the most similar tokens to tokens to record
				groups (list)			: list of column names defined by cossims to use when applying group labels
				group_type (list)		: list of strings for each group_label naming the kinds of group
				group_labels (dict)		: list of dicts mapping the tokens to group labels for each group type
			
			returns:
				cossims (pd.DataFrame)	: dataframe containing information about cosine similarities for each token/target combination + topk most similar tokens
		'''
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
		
		tokens = self.tokens_to_mask if tokens is None else tokens
		targets = self.__format_tokens_for_tokenizer(targets) if targets else {}
		targets = tuner_utils.apply_to_all_of_type(targets, str, lambda token: token if tuner_utils.verify_tokens_exist(self.tokenizer, token) else None) or {}
		
		cos = nn.CosineSimilarity(dim=-1)
		cossims = []
		
		for token in tokens:
			token_id 		= self.tokenizer.convert_tokens_to_ids(token)
			token_embedding = self.word_embeddings[token_id]
			token_cossims 	= cos(token_embedding, self.word_embeddings)
			included_ids 	= torch.topk(token_cossims, k=topk+1).indices.tolist() # add one so we can leave out the identical token
			token_cossims 	= token_cossims.tolist()
			update_cossims(cossims=cossims, values=token_cossims, included_ids=included_ids, excluded_ids=token_id, k=topk, target_group=f'{topk} most similar')
			
			if token in targets:
				target_ids = [token_id for token_id in self.tokenizer.convert_tokens_to_ids(targets[token]) if token_id != self.unk_token_id]
				update_cossims(cossims=cossims, values=token_cossims, included_ids=target_ids, target_group=token)
				
				out_groups = {k: v for k, v in targets.items() if not k == token}
				for out_group_token in out_groups:
					out_group_target_ids = [token_id for token_id in self.tokenizer.convert_tokens_to_ids(targets[out_group_token]) if token_id != self.unk_token_id]
					update_cossims(cossims=cossims, values=token_cossims, included_ids=out_group_target_ids, target_group=out_group_token)
		
		cossims = pd.DataFrame(cossims)
		
		for col in ['predicted_arg', 'target_group']:
			cossims[col]	= self.__format_strings_with_tokens_for_display(cossims[col])
		
		cossims.token 		= tuner_utils.format_roberta_tokens_for_display(cossims.token) if self.model_name == 'roberta' \
							  else self.__format_strings_with_tokens_for_display(cossims.token)
		
		for group, group_type, group_label_map in zip(groups, group_types, group_labels):
			cossims[group_type] = [group_label_map[value] if value in group_label_map else value for value in cossims[group]]
		
		return cossims
	
	def get_tsnes(
		self, n: int = None, 
		targets: Dict[str,List[str]] = {},
		target_group_labels: Dict[str,str] = {},
		**tsne_kwargs
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
			Returns tsne targets filtered according to the model type being used
			
				params:
					n (int) 			: how many of the first n good candidates to include
					targets (dict)		: also include everything manually specified in here
					added_words (list)	: additional words to include
				
				returns:
					targets (dict)		: dict mapping token groups to targets formatted according to model conventions
			'''
			target_values 	= tuner_utils.flatten(list(targets.values()))
			tokenizer_keys 	= list(self.tokenizer.get_vocab().keys())
			
			# we convert to lower and remove roberta stuff for comparison to the dataset words
			# this can be done using dict comprehension, but this way turns out to be a bit faster
			comparison_keys	= [k.replace(chr(288), '').lower() for k in tokenizer_keys]
			
			pos 			= 'verbs' if self.exp_type == 'newverb' else 'nouns'
			
			with open(os.path.join(self.original_cwd, 'conf', pos + '.txt'), 'r') as f:
				candidates 	= [w.lower().strip() for w in f.readlines()]
			
			names_sets_keys = []
			if n is not None:
				names_sets_keys.append((f'first {n}', candidates, comparison_keys))
			
			if targets is not None:
				names_sets_keys.append(('targets', target_values, tokenizer_keys))
			
			targets = {}
			for name, candidate_set, set_keys in names_sets_keys:
				targets[name] 				= {}
				filtered_keys 				= [k for k in set_keys if k in candidate_set]
				selected_keys 				= [tokenizer_keys[set_keys.index(k)] for k in filtered_keys]
				
				 # remove duplicates while preserving order
				 # we get duplicates in roberta when pulling words from the uncased targets files
				targets[name]['words'] 		= list(dict.fromkeys(selected_keys))
				
				if self.model_name == 'roberta':
					# if we are using roberta, filter to tokens that start with a preceding space and are not followed by a capital letter (to avoid duplicates))
					targets[name]['words'] 	= [token for token in targets[name]['words'] if re.search(rf'^{chr(288)}(?![A-Z])', token)]
				
				targets[name]['words']		= [w for w in targets[name]['words'] if not w in added_words]
				if name == f'first {n}':
					targets[name]['words']	= targets[name]['words'][:n]
				
				targets[name]['words']		= targets[name]['words'] + added_words
				targets[name]['tokens']		= {k: self.tokenizer.convert_tokens_to_ids(k) for k in targets[name]['words']}
				targets[name]['embeddings'] = {k: self.word_embeddings[v].reshape(1,-1) for k, v in targets[name]['tokens'].items()}
			
			return targets
		
		formatted_targets 		= self.__format_tokens_for_tokenizer(targets)
		added_words			 	= self.tokens_to_mask
		targets 			 	= get_formatted_tsne_targets(n=n, targets=formatted_targets, added_words=added_words)
		
		if target_group_labels:
			target_group_labels = {self.__format_tokens_for_tokenizer(k): v for k, v in target_group_labels.items()}
		else:
			target_group_labels = {k: k for k in self.tokens_to_mask}
		
		tsne_results 			= []
		
		for group in targets:
			
			# gotta have at least two embeddings to do a tsne
			if len(targets[group]['embeddings']) > 1:
				# we create the TSNE object inside the loop to reset the random state each time for reproducibility
				tsne = TSNE(**tsne_kwargs)
				
				with torch.no_grad():
					# move to CPU since sklearn wants numpy arrays, which have to be an the cpu
					vectors = torch.cat([embedding for embedding in targets[group]['embeddings'].values()]).cpu()
					two_dim = tsne.fit_transform(vectors)
				
				for token, tsne1, tsne2 in zip(targets[group]['words'], two_dim[:,0], two_dim[:,1]):
					
					target_group = 'novel token' \
									if token in added_words \
									else group if group == f'first {n}' \
									else tuner_utils.multiplator([
										target_group 
										for target_group in formatted_targets 
										if token in formatted_targets[target_group]
									])
					
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
		
		tsne_results 				= pd.DataFrame(tsne_results)
		tsne_results.token			= tuner_utils.format_roberta_tokens_for_display(tsne_results.token) if self.model_name == 'roberta' \
									  else self.__format_strings_with_tokens_for_display(tsne_results.token)
		tsne_results.target_group 	= self.__format_strings_with_tokens_for_display(tsne_results.target_group)
		
		return tsne_results
	
	
	# evaluation functions
	def predict_sentences(
		self,
		sentences: List[str] = None,
		info: str = '',
		output_fun: Callable = print
	) -> Dict:
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
		restore_training 		= False
		if self.model.training:
			restore_training 	= True
			log.warning('Cannot predict in training mode. Setting to eval mode temporarily.')
			self.model.eval()
		
		if sentences is None:
			sentences = f'The local {self.mask_token} will step in to help.'
			log.info(f'No sentence was provided. Using default sentence "{sentences}"')
		
		max_len 	= max([len(sentence) for sentence in sentences])
		sentences 	= self.__format_data_for_tokenizer(tuner_utils.listify(sentences))
		model_inputs = self.tokenizer(sentences, return_tensors='pt', padding=True).to(self.device)
		
		with torch.no_grad():
			outputs = self.model(**model_inputs)
		
		logprobs 			= F.log_softmax(outputs.logits, dim=-1)
		predicted_ids 		= torch.argmax(logprobs, dim=-1)
		predicted_sentences = [self.tokenizer.decode(predicted_sentence_ids) for predicted_sentence_ids in predicted_ids]
		
		if output_fun is not None:
			for sentence, predicted_sentence in zip(sentences, predicted_sentences):
				output_fun(f'{info + " " if info else ""}input: {(sentence + ",").ljust(max_len+1)} prediction: {predicted_sentence}')
		
		if restore_training:
			self.model.train()
		
		return {'inputs': sentences, 'model_inputs': model_inputs, 'predictions': predicted_sentences, 'outputs': outputs}
	
	def get_topk_mask_token_predictions(
		self, 
		predictions: Dict,
		eval_cfg: DictConfig = None,
		k: int = 20,
		data: Dict = None,
		targets: Dict[str,List[str]] = None
	) -> pd.DataFrame:
		'''
		Get the topk model predictions for mask token indices in a set of predictions
		of the form {epoch: {(output generated by self.predict_sentences())}}
		
			params:
				predictions (dict)		: a dictionary mapping an epoch label to a set of sentence predictions generated by predict_sentences()
										  the predictions must be for identical sentences, differing only in the epoch at which they were generated
				eval_cfg (dictconfig)	: a dictconfig containing the data used to generate the predictions, as well as the number of topk_mask_token_predictions to evaluate
				k (int)					: if no eval_cfg is passed, the number of topk mask token predictions to consider
				data (dict)				: a dictionary created by self.__get_formatted_datasets for the sentences used to generate the predictions.
										  if no data is passed, it is assumed that the data are from the eval_cfg.data.prediction_sentences
				targets (dict)			: a dict of lists of target labels to targets to compare predictions to. summary statistics saying how consistent
										  predictions are for these targets are added to the dataframe
			
			returns:
				results (pd.DataFrame)	: a dataframe containing the topk predictions for each sentence along with various summary statistics
		'''
		if data is None:
			data = self.__load_eval_predictions_data(eval_cfg=eval_cfg)
		
		if eval_cfg is not None:
			k = eval_cfg.topk_mask_token_predictions
		
		if targets is None:
			if self.exp_type == 'newarg':
				targets = {token: [token] for token in self.tokens_to_mask}
			elif self.exp_type == 'newverb':
				targets = {'[verb]': [token for token in self.tokens_to_mask]}
				targets.update(self.args)
		else:
			targets = {k: self.__format_tokens_for_tokenizer(v) for k, v in targets.items()}
		
		targets_to_labels 			= {token: label for label in targets for token in targets[label]}
		target_indices 				= torch.tensor(self.tokenizer.convert_tokens_to_ids(list(targets_to_labels.keys()))).to(self.device)
		targets 					= self.__format_strings_with_tokens_for_display(self.tokenizer.convert_ids_to_tokens(target_indices))
				
		epochs 						= list(predictions.keys())
		topk_mask_token_predictions = []
		
		for i, (sentence, masked_token_indices) in enumerate(zip(data['sentences'], data['masked_token_indices'])):
			sentence_predictions 	= []
			display_sentence 		= sentence
			for masked_token_type, masked_token_index in masked_token_indices.items():
				# kind of hacky. assumes we're only teaching one new verb
				if self.exp_type == 'newverb' and masked_token_type == '[verb]':
					display_sentence 	= display_sentence.replace(self.mask_token, self.__format_strings_with_tokens_for_display(self.tokens_to_mask[0]), 1)
				else:	
					display_sentence 	= display_sentence.replace(self.mask_token, masked_token_type, 1)
				
				index_predictions 	= []
				intersect 			= None
				intersect_tgts 		= None
				intersect_type_tgts	= None
				
				target_type_indices = torch.tensor([index for index in target_indices if targets_to_labels[self.tokenizer.convert_ids_to_tokens(index.item())] == masked_token_type]).to(self.device)
				
				type_targets 		= self.__format_strings_with_tokens_for_display(self.tokenizer.convert_ids_to_tokens(target_type_indices))
				
				for epoch in epochs:
					
					probs						= F.softmax(predictions[epoch]['outputs'].logits[i], dim=-1)[masked_token_index]
					
					prob_mass_tgts 				= torch.sum(probs.index_select(-1, target_indices))
					prob_mass_type_tgts			= torch.sum(probs.index_select(-1, target_type_indices))
					
					top 						= torch.topk(torch.log(probs), k=k).indices
					top_tgts 					= torch.tensor([token_id for token_id in top if token_id in target_indices])
					top_type_tgts				= torch.tensor([token_id for token_id in top if token_id in target_type_indices])
					
					n_tgts						= len([index for index in top if index in target_indices])
					n_type_tgts 				= len([index for index in top if index in target_type_indices])
					
					percent_tgts 				= n_tgts/k*100
					percent_type_tgts			= n_type_tgts/k*100
					
					percent_tgts_in_top 		= n_tgts/len(target_indices)*100
					percent_type_tgts_in_top 	= n_type_tgts/len(target_type_indices)*100
					
					top 						= self.tokenizer.convert_ids_to_tokens(top)
					top 						= self.__format_strings_with_tokens_for_display(top)
					
					index_predictions.append({
						'sentence_num'					: i,
						'epoch'							: epoch,
						'topk'							: k,
						'masked_token_type'				: masked_token_type,
						'masked_token_index'			: masked_token_index,
						
						'type_targets' 					: ', '.join(type_targets),
						'n_type_targets'				: n_type_tgts,
						'prob_mass_type_targets'		: prob_mass_type_tgts.item(),
						'percent_type_targets'			: percent_type_tgts,
						'percent_type_targets_in_top'	: percent_type_tgts_in_top,
						
						'targets' 						: ', '.join(targets),
						'n_targets' 					: n_tgts,
						'prob_mass_targets'				: prob_mass_tgts.item(),						
						'percent_targets'				: percent_tgts,
						'percent_targets_in_top'		: percent_tgts_in_top,
						
						'top_predictions'				: ', '.join(top),
					})
					
					intersect					= set(top) if intersect is None else intersect.intersection(top)
					intersect_tgts 				= set(top_tgts) if intersect_tgts is None else intersect_tgts.intersection(top_tgts)
					intersect_type_tgts			= set(top_type_tgts) if intersect_type_tgts is None else intersect_type_tgts.intersection(top_type_tgts)
				
				if len(epochs) > 1:
					perc_overlap 					= len(intersect)/k*100
					for i, _ in enumerate(index_predictions):
						common_type_target_tokens 	= [token for token in intersect if token in type_targets]
						k_without_type_targets 		= k - len(common_type_target_tokens)
						
						common_target_tokens 		= [token for token in intersect if token in targets]
						k_without_targets 			= k - len(common_target_tokens)
						
						index_predictions[i].update({
							'common_type_target_tokens'					: ', '.join(intersect_type_tgts),
							'n_common_type_target_tokens'				: len(intersect_type_tgts),
							'percent_common_type_target_tokens'			: len(intersect_type_tgts)/k*100,
							'percent_type_targets_in_common_tokens'		: len(intersect_type_tgts)/len(target_type_indices)*100,
							
							'common_target_tokens' 						: ', '.join(intersect_tgts),
							'n_common_target_tokens'					: len(intersect_tgts),
							'percent_common_target_tokens'				: len(intersect_tgts)/k*100,
							'percent_targets_in_common_tokens'			: len(intersect_tgts)/len(target_indices)*100,
							
							'common_tokens'								: ', '.join(intersect),
							'n_common_tokens'							: len(intersect),
							'percent_common_tokens'						: len(intersect)/k*100,
							'percent_common_tokens_excl_type_targets'	: len(intersect - set(type_targets))/k_without_type_targets*100,
							'percent_common_tokens_excl_targets'		: len(intersect - set(targets))/k_without_targets*100,
						})
					
				sentence_predictions.extend(index_predictions)
			
			for i, _ in enumerate(sentence_predictions):
				sentence_predictions[i].update({'sentence': display_sentence})
			
			topk_mask_token_predictions.extend(sentence_predictions)
		
		topk_mask_token_predictions = pd.DataFrame(topk_mask_token_predictions)
		topk_mask_token_predictions = tuner_utils.move_cols(
										df=topk_mask_token_predictions,
										cols_to_move='sentence',
									)
		
		return topk_mask_token_predictions
	
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
		model_path 		= os.path.join(self.checkpoint_dir, 'model.pt')
		metrics 		= pd.read_csv(os.path.join(self.checkpoint_dir, 'metrics.csv.gz'))
		total_epochs 	= max(metrics.epoch)
		loss_df 		= metrics[(metrics.metric == 'loss') & (~metrics.dataset.str.endswith(' (train)'))]
		if os.path.isfile(model_path) and not epoch == 0:
			# we use the metrics file to determine the epoch at which the full model was saved
			# note that we have not saved the model state at each epoch, unlike with the weights
			# this is a limitation of the gradual unfreezing approach
			epoch = tuner_utils.get_best_epoch(loss_df, method='mean')
			
			log.info(f'Restoring model state from epoch {epoch}/{total_epochs}')
			
			with open(model_path, 'rb') as f:
				self.model.load_state_dict(torch.load(f, map_location=torch.device(self.device)))
			
			self.model.to(self.device)
			
			# set this so we reinitialize the model state if we later want to restore to epoch 0
			# we only need to do this if we're restoring from the full model state, instead of just restoring the weights
			self.load_full_model = True
			
			return epoch, total_epochs
			
		elif os.path.isfile(model_path) and epoch == 0 and self.load_full_model:
			# recreate the model's initial state if we are loading from 0
			# we need to make sure that when we are restoring an unfrozen model to 0, we start at the initial state
			# if we had restored to a later epoch and then tried to go back to an earlier one just
			# by restoring the weights, that would still leave the rest of the model updates intact
			self.tokenizer 	= tuner_utils.create_tokenizer_with_added_tokens(self.string_id, self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs)
			self.model 		= AutoModelForMaskedLM.from_pretrained(self.string_id, **self.cfg.model.model_kwargs)
			self.model.resize_token_embeddings(len(self.tokenizer)).to(self.device)
			
			# if we try to re-load to 0, we don't need to bother reloading unless we've gone to a later epoch in the meantime
			self.load_full_model = False
		
		weights_path = os.path.join(self.checkpoint_dir, 'weights.pkl.gz')
		with gzip.open(weights_path, 'rb') as f:
			weights = pkl.load(f)
		
		if epoch == None or epoch in ['max', 'total', 'highest', 'last', 'final']:
			epoch = total_epochs
		elif 'best' in str(epoch):
			epoch = tuner_utils.get_best_epoch(loss_df, method=epoch)
		
		log.info(f'Restoring saved weights from epoch {epoch}/{total_epochs}')
		
		with torch.no_grad():
			for token in weights[epoch]:
				weights[epoch][token].to(self.device)
				token_id = self.tokenizer.convert_tokens_to_ids(token)
				self.word_embeddings[token_id] = weights[epoch][token]
		
		# return the epoch and total_epochs to help if we didn't specify it
		return epoch, total_epochs
	
	def evaluate(self, eval_cfg: DictConfig) -> None:
		'''
		Calls the appropriate function for model evaluation.
		
			params:
				eval_cfg (dictconfig): a configuration specifying evaluation settings
		'''
		
		# this is just done so we can record it in the results
		self.__restore_original_random_seed()
		
		if eval_cfg.data.exp_type in ['newverb', 'newarg']:
			self.__evaluate_newtoken_experiment(eval_cfg=eval_cfg)
		else:
			self.__eval(eval_cfg=eval_cfg)
	
	def load_eval_file(self, eval_cfg: DictConfig) -> Dict:
		'''
		Loads a file specified in the eval cfg, returning a Dict of 
		sentences, inputs, and masked token indices grouped by 
		sentence type for model evaluation.
			
			params: 
				eval_cfg (DictConfig) : dictconfig containing evaluation configuration options
			
			returns:
				types_sentences (dict): dict with sentences, inputs, and arg_indices 
										for each sentence type in the eval data file
		'''
		resolved_path 			= os.path.join(self.original_cwd, 'data', eval_cfg.data.name)
		
		with open(resolved_path, 'r') as f:
			raw_input 			= [line.strip() for line in f]
		
		sentences 				= [[s.strip() for s in r.split(' , ')] for r in raw_input]
		sentences 				= self.__format_data_for_tokenizer(sentences)
		transposed_sentences 	= list(map(list, zip(*sentences)))
		
		if self.exp_type in ['newverb', 'newarg']:
			sentence_types 		= eval_cfg.data.sentence_types
		else:
			sentence_types 		= list(range(len(transposed_sentences))) # dummy value just in case
		
		assert len(sentence_types) == len(transposed_sentences), 'Number of sentence types does not match in data config and data!'
		
		lens 					= [len(sentence_group) for sentence_group in transposed_sentences]
		
		# way faster to flatten the inputs and then restore instead of looping
		flattened_sentences 	= tuner_utils.flatten(transposed_sentences)
		
		# formatting for __get_formatted_datasets
		flattened_sentences 	= {'eval_data': {'data': flattened_sentences}}
		
		mask_args 				= True if eval_cfg.data.exp_type == 'newverb' else False
		
		inputs_dict 			= self.__get_formatted_datasets(
									mask_args 		= mask_args, 
									masking_style 	= 'eval', 
									datasets 		= flattened_sentences, 
									eval_cfg 		= eval_cfg
								)
		
		# unpack the results to group by sentence type
		flattened_sentences		= inputs_dict['eval_data']['sentences']
		masked_inputs 			= inputs_dict['eval_data']['inputs']
		masked_token_indices 	= inputs_dict['eval_data']['masked_token_indices']
		
		types_sentences 		= {}
		for sentence_type, n_sentences, sentence_group in zip(sentence_types, lens, transposed_sentences):
			types_sentences[sentence_type]								= {}
			types_sentences[sentence_type]['sentences'] 				= sentence_group
			
			types_sentences[sentence_type]['inputs']					= {}
			types_sentences[sentence_type]['inputs']['input_ids']		= masked_inputs['input_ids'][:n_sentences]
			types_sentences[sentence_type]['inputs']['attention_mask']	= masked_inputs['attention_mask'][:n_sentences]
			masked_inputs['input_ids']									= masked_inputs['input_ids'][n_sentences:]
			masked_inputs['attention_mask']								= masked_inputs['attention_mask'][n_sentences:]
			
			types_sentences[sentence_type]['masked_token_indices']		= masked_token_indices[:n_sentences]
			masked_token_indices										= masked_token_indices[n_sentences:]
		
		if (
			not all(sentence_type in types_sentences for sentence_type in sentence_types) or \
			any(masked_token_indices) or torch.any(masked_inputs['input_ids']) or torch.any(masked_inputs['attention_mask'])
		):
			raise ValueError('The number of sentences and inputs does not match!')
		
		# safely flatten the dict (if we have only a single sentence type as in the PTB experiments)
		types_sentences = tuner_utils.unlistify(types_sentences)
		
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
		data: Dict = None
	) -> Union[pd.DataFrame,Dict]:
		'''
		Returns a dataframe containing a summary of odds ratios data
			
			params:
				epoch (int)							: which epoch to evaluate
				eval_cfg (DictConfig)				: a dictconfig containing evaluation config options
				data (dict)							: a dict consisting of sentences, inputs, and arg_indices
													  to evaluate model performance on
			
			returns:
				odds_ratios_summary (pd.DataFrame)	: dataframe containing a summary of the odds ratios data
		'''
		
		# use data if provided so we don't have to reload it, but load it automatically if not
		data 				= self.load_eval_file(eval_cfg) if data is None else data
		epoch, total_epochs = self.restore_weights(epoch)
		
		# get the experiment-type specific evaluation groups
		if eval_cfg.data.exp_type == 'newverb':	
			args = self.args
			if 'added_args' in eval_cfg.data and self.args_group in eval_cfg.data.added_args:
				args		= {arg_type: args[arg_type] + self.__format_tokens_for_tokenizer(eval_cfg.data.added_args[self.args_group][arg_type]) for arg_type in args}
		else:
			args 			= self.tokens_to_mask
			tokens_to_roles = {self.__format_tokens_for_tokenizer(v): k for k, v in eval_cfg.data.eval_groups.items()}
		
		# when we load the eval data, we want to return it grouped by sentence type for general ease of use.
		# however, concatenating everything together for evaluation is faster. For this reason, we join everything together,
		# then evaluate, and then split it apart. this may seem a bit redundant, but that's why we're doing it this way
		inputs = {
			'input_ids'		: torch.cat([data[sentence_type]['inputs']['input_ids'] for sentence_type in data]),
			'attention_mask': torch.cat([data[sentence_type]['inputs']['attention_mask'] for sentence_type in data])
		}
		
		masked_token_indices	= tuner_utils.flatten([data[sentence_type]['masked_token_indices'] for sentence_type in data])
		sentences				= tuner_utils.flatten([data[sentence_type]['sentences'] for sentence_type in data])
		
		log.info(f'Evaluating model on testing data')
		with torch.no_grad():
			outputs 			= self.model(**inputs)
		
		odds_ratios_summary 	= self.__collect_results(outputs=outputs, masked_token_indices=masked_token_indices, sentences=sentences, eval_groups=args)
		odds_ratios_summary 	= pd.DataFrame(odds_ratios_summary)
		
		# here's where we add back the sentence type and sentence number information that we collapsed when pasting everything together to run the model
		
		# get the number of sentences of each type
		num_sentences			= [len(data[sentence_type]['sentences']) for sentence_type in data]
		
		# get the sentence types
		sentence_types			= [sentence_type for sentence_type in data]
		
		# repeat each sentence type the appropriate number of times
		sentence_types			= tuner_utils.flatten([[sentence_type] * num for sentence_type, num in zip(sentence_types, num_sentences)])
		
		# construct from 0 to n sentence numbers corresponding to the number of sentences of each type
		sentence_nums			= tuner_utils.flatten([list(range(num)) for num in num_sentences])
		
		# add these columns back to the data. this works because the sentences are ordered in the same way in the data dict as when we concat them together
		# we do this for each token, since each eval token will be represented for each sentence
		for token in odds_ratios_summary.token.unique():
			odds_ratios_summary.loc[odds_ratios_summary.token == token, 'sentence type'] 	= sentence_types
			odds_ratios_summary.loc[odds_ratios_summary.token == token, 'sentence num']		= sentence_nums
		
		# reorder and rename the columns for display and ease of use
		odds_ratios_summary			= tuner_utils.move_cols(odds_ratios_summary, cols_to_move=['sentence type', 'sentence num'], ref_col='sentence', position='after')
		odds_ratios_summary.columns = [col.replace(' ', '_') for col in odds_ratios_summary.columns]
		
		# add experiment specific information
		if eval_cfg.data.exp_type == 'newverb':
			odds_ratios_summary['token_type'] 		= ['tuning' if token in tuner_utils.flatten(list(self.args.values())) else 'eval only' for token in odds_ratios_summary.token]
			# replace the mask tokens in the sentences with the argument types according to the mask token indices
			for sentence, gf_indices in zip(odds_ratios_summary.sentence.unique().copy(), masked_token_indices):
				for gf in gf_indices:
					current_sentence 	= odds_ratios_summary.loc[odds_ratios_summary.sentence == sentence].sentence.unique()[0]
					current_sentence 	= current_sentence.replace(self.mask_token, gf, 1)
					# we know there is a unique sentence since that is how we are looping
					odds_ratios_summary.loc[odds_ratios_summary.sentence == sentence, 'sentence'] = current_sentence
					sentence 			= current_sentence
		else:
			odds_ratios_summary['role_position'] 	= [tokens_to_roles[token] for token in odds_ratios_summary.token]
		
		# format the strings with tokens for display purposes before returning
		for col in ['ratio_name', 'token', 'arg_type']:
			odds_ratios_summary[col] = self.__format_strings_with_tokens_for_display(odds_ratios_summary[col]).tolist()
		
		# add information about the evaluation parameters
		odds_ratios_summary = odds_ratios_summary.assign(
			eval_data 		= eval_cfg.data.name.split('.')[0],					
			epoch_criteria 	= eval_cfg.epoch if isinstance(eval_cfg.epoch, str) else 'manual',
			eval_epoch 		= epoch,
			total_epochs 	= total_epochs,
		)
		
		# add Tuner hyperparameters to the data
		odds_ratios_summary = self.__add_hyperparameters_to_summary_df(odds_ratios_summary)
		
		# for now, we are not using these columns, so we're dropping them before returning. we can easily change this later if desired
		odds_ratios_summary = odds_ratios_summary.drop(
			['logit', 'probability', 'log_probability', 'surprisal', 'predicted_sentence', 'predicted_ids'], axis=1
		)
		
		return odds_ratios_summary
	
	def compare_model_performance_to_baseline(self, eval_cfg: DictConfig) -> List[float]:
		'''
		Evaluates the model on a dataset that a pre-fine-tuning version of the model has been
		trained on, and compares the probability distributions pre- and post-fine-tuning using
		KL divergence. This is useful when unfreezing options have been used, since it provides
		a measure of how much the model's predictions have changed in sentences where we would
		like them to stay the same.
		
			params:
				eval_cfg (DictConfig)	: a dictconfig specifying a comparison_dataset (via file location)
										  to use as a benchmark for MLM performance
			
			returns:
				kl_divs (pd.DataFrame)	: a dataframe with the kl divergences for each of eval_cfg.comparison_n_exs
										  examples in the dataset for the current model compared to a pre-fine-tuning baseline
		'''
		# doing this lets us use either absolute paths or paths relative to the starting dir
		dataset_loc 		= eval_cfg.comparison_dataset.replace(self.original_cwd, '')
		dataset_loc 		= os.path.join(self.original_cwd, eval_cfg.comparison_dataset)
		dataset_name 		= os.path.split(dataset_loc)[-1]
		
		log.info(f'Loading dataset {dataset_name}')
		dataset 			= self.__load_format_dataset(dataset_loc=dataset_loc, split='test', n_examples=eval_cfg.comparison_n_exs)
		
		if self.model.training:
			log.warning('Model performance will not be compared to baseline in training mode. Model will be set to eval mode.')
			self.model.eval()
		
		KL_baseline_loss 	= kl_baseline_loss.KLBaselineLoss(
								model 				= self.model,
								tokenizer 			= self.tokenizer,
								dataset 			= dataset,
								n_examples_per_step = eval_cfg.comparison_n_exs,
								masking 			= eval_cfg.comparison_masking,
								model_kwargs 		= self.cfg.model.model_kwargs,
								tokenizer_kwargs 	= self.cfg.model.tokenizer_kwargs,
							)
		
		with torch.no_grad():
			mean_kl_div, kl_divs, mask_indices \
							 = KL_baseline_loss(progress_bar=True, return_all=True)
		
		log.info(f'Mean KL divergence from baseline on {dataset_name}: {mean_kl_div:.2f} (\u00b1{tuner_utils.sem(kl_divs):.2f})')
		
		# pandas needs the kl_divs to be on cpu
		kl_divs				= pd.DataFrame(kl_divs.cpu(), columns=['kl_div']).assign(
								dataset_name 	= dataset_name,
								source 			= dataset['source'],
								text 			= dataset['text'],
								sentence_num 	= dataset['original_pos'],
								eval_kl_masking = eval_cfg.comparison_masking,
							)
		
		if eval_cfg.comparison_masking != 'none':
			kl_divs 		= kl_divs.assign(mask_indices=mask_indices)
		
		return kl_divs
	
	
	# wrapper/helper functions for plots/accuracies (implemented in tuner_utils and tuner_plots)
	def create_metrics_plots(self, metrics: pd.DataFrame) -> None:
		'''
		Calculates which metrics to plot using identical y axes and which to plot on the same figure, and plots metrics
		
			params:
				metrics (pd.DataFrame)	: a dataframe containing metrics to plot
		'''
		ignore_for_ylims = deepcopy(self.tokens_to_mask)
		dont_plot_separately = []
		
		if self.exp_type == 'newverb':
			
			ignore_for_ylims += tuner_utils.flatten([[arg_type, f'({arg_type})'] for arg_type in list(self.args.keys())]) + \
								self.__format_strings_with_tokens_for_display(tuner_utils.flatten([self.args[arg_type] for arg_type in self.args]))
			
			args = deepcopy(self.args)
			args = {k: self.__format_strings_with_tokens_for_display(v) for k, v in args.items()}
			
			for arg_type in args:
				for arg in args[arg_type]:
					dont_plot_separately.extend([m for m in metrics.metric.unique() if m.startswith(f'{arg} ({arg_type})')])
		
		tuner_plots.create_metrics_plots(metrics=metrics, ignore_for_ylims=ignore_for_ylims, dont_plot_separately=dont_plot_separately)
	
	def create_cossims_plot(self, *args: Tuple, **kwargs: Dict) -> None:
		'''
		Calls tuner_plots.create_cossims_plot
		
			params:
				*args (tuple)	: passed to tuner_plots.create_cossims_plot
				**kwargs (dict)	: passed to tuner_plots.create_cossims_plot
		'''
		tuner_plots.create_cossims_plot(*args, **kwargs)
	
	def create_tsnes_plots(self, *args: Tuple, **kwargs: Dict) -> None:
		'''
		Calls tuner_plots.create_tsnes_plots
		
			params:
				*args (tuple)	: passed to tuner_plots.create_tsnes_plot
				**kwargs (dict)	: passed to tuner_plots.create_tsnes_plot
		'''
		tuner_plots.create_tsnes_plots(*args, **kwargs)
	
	def create_odds_ratios_plots(self, *args: Tuple, **kwargs: Dict) -> None:
		
		'''
		Calls tuner_plots.create_odds_ratios_plots
		
			params:
				*args (tuple)	: passed to tuner_plots.create_odds_ratios_plots
				**kwargs (dict)	: passed to tuner_plots.create_odds_ratios_plots
		'''
		tuner_plots.create_odds_ratios_plots(*args, **kwargs)
	
	def create_kl_divs_plot(self, *args: Tuple, **kwargs: Dict) -> None:
		'''
		Calls tuner_plots.create_kl_divs_plot
		
			params:
				*args (tuple)	: passed to tuner_plots.create_kl_divs_plot
				**kwargs (dict)	: passed to tuner_plots.create_kl_divs_plot
		'''
		tuner_plots.create_kl_divs_plot(*args, **kwargs)
	
	def get_odds_ratios_accuracies(self, *args: Tuple, **kwargs: Dict) -> pd.DataFrame:
		'''
		Calls tuner_utils.get_odds_ratios_accuracies, returns a dataframe containing accuracy information with tokens formatted for display
		
			params:
				*args (tuple)		: passed to tuner_utils.get_odds_ratios_accuracies
				**kwargs (list)		: passed to tuner_utils.get_odds_ratios_accuracies
			
			returns:
				acc (pd.DataFrame)	: dataframe containing accuracy information for each novel token in each sentence
		'''
		return tuner_utils.get_odds_ratios_accuracies(*args, **kwargs)
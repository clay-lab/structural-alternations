# tuner.py
# 
# Tunes a model on training data and provides functions for evaluation
import os
import re
import sys
import json
import gzip
import hydra
import torch
import spacy
import string
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
from mixout.module import MixLinear
from omegaconf import DictConfig, OmegaConf, open_dict, ListConfig
from contextlib import suppress
from deprecated import deprecated
from collections import Counter
from nltk.corpus import stopwords

import transformers

from transformers import logging as lg
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.manifold import TSNE

import tuner_plots
import tuner_utils
import kl_baseline_loss
import layerwise_baseline_loss
from tuner_utils import none
from mixout.module import MixLinear

lg.set_verbosity_error()

log = logging.getLogger(__name__)

class Tuner:
	
	# START Computed Properties
	
	@property
	def mixed_tuning_data(self) -> Dict:
		'''Returns a dict with roberta-style masked sentences, inputs, and masked token indices.'''
		return self._get_formatted_datasets(masking_style=self.masked_tuning_style)[self.tuning]
	
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
	def _create_inputs(
		self,
		sentences: List[str] = None,
		to_mask: Union[List[int],List[str]] = None,
		masking_style: str = 'always',
		compat_mode: bool = False
	) -> Tuple['BatchEncoding', torch.Tensor, List[Dict]]:
		'''
		Creates masked model inputs from a batch encoding or a list of sentences, with tokens in to_mask masked
				
			params:
				sentences (list) 				: a list of sentences to get inputs for
				to_mask (list)					: list of token_ids or token strings to mask in the inputs
				masking_style (str)				: if 'always', always replace to_mask token_ids with mask tokens
												  if 'none', do nothing
												  if None, return bert/roberta-style masked data
				compat_mode (str)				: we changed how we mask sentences at one point
												  previously, we replaced tokens to mask in the string, and then tokenized
												  now, we tokenize, and then replace the indices of the tokens to mask with the mask token id
												  the new way is better, but it's not the right way to evaluate models we ran the old way
												  this provides backward compatibility
			
			returns:
				masked_inputs (BatchEncoding)	: the inputs with the tokens in to_mask replaced with mask, 
												  original, or random tokens, dependent on masking_style
				labels (tensor)					: a tensor with the target labels
				masked_token_indices (list)		: a list of dictionaries mapping the (display-formatted) tokens in to_mask to their 
												  original position(s) in each sentence (since they are no longer in the masked sentences)
		'''
		if not sentences:
			return None, None, None
		
		if to_mask is None:
			to_mask = self.tokens_to_mask
		
		to_mask = [self.tokenizer.convert_tokens_to_ids(token) if isinstance(token,str) else token for token in tuner_utils.listify(to_mask)]
		assert not any(token_id == self.unk_token_id for token_id in to_mask), 'At least one token to mask is not in the model vocabulary!'
		
		inputs = self._format_data_for_tokenizer(sentences)
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
		
		# if we are in compat mode, we replace the tokens in the string and use those as the inputs to get the masked inputs
		if compat_mode:
			inputs 				= self._format_data_for_tokenizer(sentences)
			to_replace 			= self._format_data_for_tokenizer(self._format_strings_with_tokens_for_display(self.tokenizer.convert_ids_to_tokens(to_mask)))
			for i, _ in enumerate(inputs):
				for t in to_replace:
					inputs[i] = inputs[i].replace(t, self.mask_token)
			
			inputs 				= self.tokenizer(inputs, return_tensors='pt', padding=True).to(self.device)
		
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
	
	def _get_formatted_datasets(
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
									  	  'eval' is equivalent to 'always' for newarg experiments. for new verb experiments, 
									  	  it produces a dataset where each unique masked sentence is used only once
									  	  this speeds up eval since it means we don't need to run the model on identical sentences repeatedl
				datasets (Dict-like)	: which datasets to generated formatted data for
				eval_cfg (DictConfig)	: used during evaluation to add generalization arguments to the newverb experiments
			
			returns:
				formatted_data (dict)	: a dict with, for each dataset, sentences, inputs, (+ masked_token_indices if masking_style != 'none')
		'''
		if (hasattr(self, 'mask_added_tokens') and self.mask_added_tokens) or not hasattr(self, 'mask_added_tokens'):
			to_mask = self.tokenizer.convert_tokens_to_ids(self.tokens_to_mask)
		else:
			to_mask = []
		
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
				args 					= self._format_strings_with_tokens_for_display(self.args)
				if eval_cfg is not None:
					if 'added_args' in eval_cfg.data and self.args_group in eval_cfg.data.added_args:
						args 			= {arg_type: args[arg_type] + self._format_strings_with_tokens_for_display(eval_cfg.data.added_args[self.args_group][arg_type]) for arg_type in args}
					elif 'eval_args' in eval_cfg.data:
						args 			= {arg_type: args[arg_type] + self._format_strings_with_tokens_for_display(eval_cfg.data[eval_cfg.data.eval_args][arg_type]) for arg_type in eval_cfg.data[eval_cfg.data.eval_args]}
				
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
			inputs, labels, masked_token_indices = self._create_inputs(sentences=datasets[dataset]['data'], to_mask=to_mask, masking_style=masking_style if masking_style != 'eval' else 'always', compat_mode=eval_cfg.compat_mode if eval_cfg is not None and 'compat_mode' in eval_cfg.keys() else None)
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
	
	def _generate_filled_verb_data(self, sentences: List[str], to_replace: Dict[str,List[str]]) -> List[str]:
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
	def _format_strings_with_tokens_for_display(
		self, 
		data: 'any', 
		additional_tokens: List[str] = None
	) -> 'any':
		'''
		Formats strings containing tokenizer-formatted tokens for display
		
			params:
				data (any)						: a data structure (possibly infinitely nested) containing some string(s)
				additional_tokens (list[str])	: a list of strings containing additional tokens to format for display
												  used when testing on unseen tokens
			
			returns:
				data structured in the same way as the input data, with added tokens formatted for display
		'''
		additional_tokens = additional_tokens if additional_tokens is not None else []
		
		tokens_to_format = self.tokens_to_mask + additional_tokens
		if self.exp_type == 'newverb':
			# need to use deepcopy so we don't add newverb arguments to the tokens to mask attribute here
			tokens_to_format = deepcopy(tokens_to_format)
			tokens_to_format += list(self.args.values())
			tokens_to_format = tuner_utils.flatten(tokens_to_format)
		
		# in case any additional tokens are multiply specified
		tokens_to_format = list(set(tokens_to_format))
		
		# in newverb experiments, we only want to uppercase the added tokens, not the argument tokens
		tokens_to_uppercase = self.tokens_to_mask
		
		return tuner_utils.format_strings_with_tokens_for_display(
			data=data, 
			tokenizer_tokens=tokens_to_format, 
			tokens_to_uppercase=tokens_to_uppercase, 
			model_name=self.model_name, 
			string_id=self.string_id
		)
	
	def _format_data_for_tokenizer(self, data: str) -> str:
		'''
		Formats a data structure with strings (including mask tokens) in a way that makes it possible to use with self's tokenizer
		
			params:
				data (str) : a (possibly nested) data structure containing strings
			
			returns:
				output in the same structure as data, with strings formatted according to tokenizer requirements
		'''
		return tuner_utils.format_data_for_tokenizer(data=data, mask_token=self.mask_token, string_id=self.string_id, remove_punct=self.strip_punct)
		
	def _format_tokens_for_tokenizer(self, tokens: 'any') -> 'any':
		'''Pipelines formatting tokens for models'''
		formatted_tokens	 = self._format_data_for_tokenizer(tokens)
		if self.model_name == 'roberta':
			formatted_tokens = tuner_utils.format_roberta_tokens_for_tokenizer(formatted_tokens)
		else:
			formatted_tokens = tuner_utils.apply_to_all_of_type(formatted_tokens, str, lambda x: x if not x.startswith('^') else None)
		
		return formatted_tokens
	
	def _add_hyperparameters_to_summary_df(self, df: pd.DataFrame) -> pd.DataFrame:
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
			'use_kl_baseline_loss', 'use_layerwise_baseline_loss', 'original_cwd', 
			'kl_batch_size', 'layerwise_batch_size',
		]
		
		included_vars = [var for var in vars(self) if not var in exclude]
		included_vars = [var for var in included_vars if isinstance(vars(self)[var],(str,int,float,bool))]
		sorted_vars = sorted([var for var in included_vars], key=lambda item: re.sub(r'^(model)', '0\\1', item))
		
		for var in sorted_vars:
			df[var] = vars(self)[var]
		
		return df
	
	# evaluation
	def _log_debug_predictions(
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
				f'The local {self.mask_token} has stepped in to help.',
				f'The {self.mask_token} has {self._format_strings_with_tokens_for_display(self.tokens_to_mask[0])} the {self.mask_token}.',
				f'The {self._format_strings_with_tokens_for_display(self.args["[subj]"][0])} has {self.mask_token} the {self._format_strings_with_tokens_for_display(self.args["[obj]"][0])}.',
				f'The {self.mask_token} has {self.mask_token} the {self.mask_token}.',
			]
		else:
			sentences = [f'The local {self.mask_token} has stepped in to help.'] + self.tuning_data['sentences'][:2] + self.tuning_data['sentences'][-2:]
			for i, sentence in enumerate(sentences):
				for token in self._format_strings_with_tokens_for_display(self.tokens_to_mask):
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
	
	def _load_eval_predictions_data(self, eval_cfg: DictConfig) -> Dict:
		'''
		Loads the predictions data included in the eval cfg according to the settings
		
			params:
				eval_cfg (dictconfig)	: a dict config containing a list of sentences 
										  to get predictions for under eval_cfg.data.prediction_sentences
			
			returns:
				data (dict)				: a dict containing the formatted predictions sentences 
										  from the eval cfg, ready to put into the model
		'''
		if 'prediction_sentences' in eval_cfg.data:
			sentences 			= OmegaConf.to_container(eval_cfg.data.prediction_sentences)
		else:
			sentences 			= []
		
		if 'args_prediction_sentences' in eval_cfg.data:
			name = self.args_group
			
			if name in eval_cfg.data.args_prediction_sentences:
				sentences 		+= [
					s
					for s in tuner_utils.flatten(
						list(
							eval_cfg.data.args_prediction_sentences[name].values()
						)
					)
				]
			# else:
			#	log.warning(f'Prediction sentences for {name} were requested but do not exist!')
		
		if eval_cfg.debug:
			if self.exp_type == 'newverb':
				gfs 			= list(self.args.keys())
				debug_sentences = [
									f'The local {gfs[0]} has stepped in to help.',
									f'The {gfs[0]} has {self._format_strings_with_tokens_for_display(self.tokens_to_mask[0])} the {gfs[1]}.',
									f'The {self._format_strings_with_tokens_for_display(self.args["[subj]"][0])} has [verb] the {self._format_strings_with_tokens_for_display(self.args["[obj]"][0])}.',
									f'The {gfs[0]} has [verb] the {gfs[1]}.',
								]
			else:
				debug_sentences	= [f'The local {self._format_strings_with_tokens_for_display(self.tokens_to_mask[0])} will step in to help.'] + self.tuning_data['sentences'][:2] + self.tuning_data['sentences'][-2:]
		
			sentences 			= debug_sentences + sentences
		
		# remove duplicates while preserving order
		# this way we don't evaluate duplicates twice	
		sentences 		= list(dict.fromkeys(sentences))
		
		if sentences:
			data		= tuner_utils.unlistify(self._get_formatted_datasets(
							mask_args=True, 
							masking_style='eval', 
							datasets={'prediction_sentences': {'data': sentences}},
							eval_cfg=eval_cfg,
						))
		else:
			data 		= {}
		
		return data
	
	def _get_eval_predictions(
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
		prediction_data = self._load_eval_predictions_data(eval_cfg=eval_cfg)
		
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
	
	def _collect_results(
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
				if self.exp_type == 'newverb' and token in eval_groups.keys():
					tokens = eval_groups[token]
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
						if	not other_token == token 
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
							# for newverb experiments, we compare a token to itself in the other position
							# this addresses overall probability biases within the tokens that already exist in the model
							# for newarg experiments, this is not a concern, since all eval tokens are novel,
							# so we instead compare the relative probabilities of the different tokens in the same position
							if self.exp_type == 'newverb':
								logprob = logprobs[other_token_index,token_id]
							else: 
								logprob = logprobs[token_indices[token],self.tokenizer.convert_tokens_to_ids(other_token)]
							
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
								'other arg type' 		: other_token,
								'other log probability'	: logprob,
								'position ratio name'	: f'position {positions[token]}/position {positions[other_token]}',
								**common_args
							})
					else:
						results.append({**common_args})
		
		return results
	
	def _restore_original_random_seed(self) -> None:
		'''Restores the original random seed used to generate weights for the novel tokens to an attribute'''
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
			
	def _load_format_dataset(
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
	
	@deprecated(reason='__eval has not been updated to work with the latest version of Tuner. Use at your own risk.')
	def _eval(self, eval_cfg: DictConfig) -> None:
		'''
		Evaluates a model without any fine-tuning
			
			params: eval_cfg (DictConfig): evaluation config options
		'''
		self.model.eval()
		
		inputs = self.load_eval_file(eval_cfg)['inputs']
		
		with torch.no_grad():	
			log.info('Evaluating model on testing data')
			outputs = self.model(**inputs)
		
		results = self._collect_results(inputs, eval_cfg.data.eval_groups, outputs)
		summary = self.summarize_results(results)
		
		log.info('Creating aconf and entropy plots')
		tuner_plots.graph_results(summary, eval_cfg)
	
	def _get_sentences_summary_from_odds_ratios_summary(self, summary: pd.DataFrame) -> pd.DataFrame:
		'''
		Computes a summary for each sentence from the summary by 
		tokens for newverb experiments. To do this, we compute the ratio 
		
		p(t1|t1 expected position) * p(t2|t2 expected position) ...
		———————————————————————————————————————————————————————————
		p(t2|t1 expected position) * p(t1|t2 expected position) ...
		
		For each combination of t1, t2, .... This gives us the 
		probability of the sentence ... t1 ... t2 ... compared 
		to the probability of the sentence ... t2 ... t1 ...
		since the other probabilities cancel out.
		
			params: 
				summary (pd.DataFrame): a dataframe containing an 
										odds ratios summary 
										(returned by get_odds_ratios_summary)
		
			returns:
				summary (pd.DataFrame): a dataframe containing the 
										odds ratios for each sentence 
										constructed as described above
		'''
		# get all the tuples of tokens in the summary
		tuples = summary[['arg_type', 'token']].drop_duplicates(ignore_index=True)
		tuples = tuples.groupby('arg_type', sort=False)['token'].apply(list).to_dict()
		tuples = [[{k: v} for v in tuples[k]] for k in tuples]
		tuples = itertools.product(*tuples)
		tuples = [{k: v for d in t for k, v in d.items()} for t in tuples]
		
		sentences_summary = []
		log.info(f'Getting summary for sentences @ epoch {tuner_utils.multiplator(summary.eval_epoch.unique())}')
		for sentence in tqdm(summary.sentence.unique()):
			for t in tuples:
				logprob_correct 	= torch.sum(
										torch.tensor([
											summary.loc[
												(summary.sentence == sentence) & 
												(summary.token == token) & 
												(summary.arg_type == arg_type)
											].log_probability.iloc[0]
											for arg_type, token in t.items()
										])
									)
				
				other_permutations 	= itertools.permutations(t.values())
				other_permutations 	= [o_p for o_p in other_permutations if not o_p == tuple(t.values())]
				other_permutations 	= [dict(zip(t.keys(), o_p)) for o_p in other_permutations]
				
				for i, p in enumerate(other_permutations):
					logprob_wrong 	= torch.sum(
										torch.tensor([
											summary.loc[
												(summary.sentence == sentence) &
												(summary.token == token) & 
												(summary.other_arg_type == other_arg_type)
											].other_log_probability.iloc[0]
											for other_arg_type, token in p.items()
										])
									)
					
					ratio_name = 'correct/incorrect' if len(other_permutations) == 1 else f'correct/incorrect{i+1:02d}'
					
					sentences_summary.append({
						'odds_ratio'		: logprob_correct - logprob_wrong,
						'ratio_name'		: ratio_name,
						'full_ratio_name'	: f'(p({")*p(".join(["|".join([v, k]) for k, v in t.items()])}))/(p({")*p(".join(["|".join([v, k]) for k, v in p.items()])}))',
						'sentence'			: sentence,
						'sentence_type'		: summary[summary.sentence == sentence].sentence_type.unique()[0],
						'sentence_num' 		: summary[summary.sentence == sentence].sentence_num.unique()[0],
					})
		
		sentences_summary = pd.DataFrame(sentences_summary)
		sentences_summary = tuner_utils.transfer_hyperparameters_to_df(summary, sentences_summary)
		
		return sentences_summary
	
	def _evaluate_newtoken_experiment(self, eval_cfg: DictConfig) -> None:
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
		
		def get_cossims_for_current_epoch(epoch: Union[int,str], targets: Dict[str,List[str]] = None) -> pd.DataFrame:
			'''Get cosine similarities for the current epoch for the given targets.'''
			cossims_args 		= dict(topk=eval_cfg.k)
			if eval_cfg.data.exp_type == 'newarg' and targets is None:
				cossims_args.update(dict(targets=eval_cfg.data.masked_token_targets))
			elif eval_cfg.data.exp_type == 'newverb' and targets is None:
				cossims_args.update(dict(targets=self._get_newverb_cossim_targets(return_counts=False)))
			elif targets is not None:
				cossims_args.update(dict(targets=targets))
			
			predicted_roles 	= {v: k for k, v in eval_cfg.data.eval_groups.items()}
			target_group_labels = {k: v for k, v in eval_cfg.data.masked_token_target_labels.items()} if 'masked_token_target_labels' in eval_cfg.data else {}
			
			groups 				= ['predicted_arg', 'target_group']
			group_types 		= ['predicted_role', 'target_group_label']
			group_labels 		= [predicted_roles, target_group_labels]
			cossims_args.update(dict(groups=groups, group_types=group_types, group_labels=group_labels))
			
			cossims = pd.DataFrame()
			for correction in eval_cfg.cossims_corrections:
				correction_kwargs 	= eval_cfg.cossims_corrections_kwargs[correction] if correction in eval_cfg.cossims_corrections_kwargs else {}
				cossims 			= pd.concat([cossims, self.get_cossims(**cossims_args, correction=correction, correction_kwargs=correction_kwargs)], ignore_index=True)
			
			cossims = cossims.assign(eval_epoch=epoch)
			
			return cossims
		
		def get_tsnes_for_current_epoch(epoch: Union[int,str], targets: Dict[str,List[str]] = None) -> pd.DataFrame:
			'''Get t-SNEs for the current epoch.'''
			tsne_args = dict(
							n=eval_cfg.num_tsne_words, 
							n_components=2, 
							random_state=0, 
							learning_rate='auto', 
							init='pca',
							targets=targets if targets is not None else {},
						)
			
			if 'masked_token_targets' in eval_cfg.data and targets is None:
				tsne_args.update(dict(targets=eval_cfg.data.masked_token_targets))
				
			if self.exp_type == 'newverb' and targets is None:
				tsne_args.update(dict(targets={**self._get_newverb_cossim_targets(return_counts=False), **tsne_args['targets']}))
			
			if 'masked_token_target_labels' in eval_cfg.data:
				tsne_args.update(dict(target_group_labels=target_group_labels))
			
			if not tsne_args['targets']:
				tsne_args = {k: v for k, v in tsne_args.items() if not k == 'targets'}
			
			tsnes = self.get_tsnes(**tsne_args)
			tsnes = tsnes.assign(eval_epoch=epoch)
			return tsnes	
		
		self.model.eval()
		
		data 				= self.load_eval_file(eval_cfg)
		
		if eval_cfg.data.exp_type == 'newverb':
			summary_zero 		= self.get_odds_ratios_summary(epoch=0, eval_cfg=eval_cfg, data=data)
			newverb_cossim_targets, target_counts = self._get_newverb_cossim_targets()
			log.info(f'Found similarity token targets: {target_counts}')
			predictions_zero 	= {0: self._get_eval_predictions(summary=summary_zero, eval_cfg=eval_cfg, output_fun=log.info)}
			cossims 			= get_cossims_for_current_epoch(epoch=0, targets=newverb_cossim_targets)
			tsnes 				= get_cossims_for_current_epoch(epoch=0, targets=newverb_cossim_targets)
		else:
			cossims 			= get_cossims_for_current_epoch(epoch=0)
			tsnes 				= get_tsnes_for_current_epoch(epoch=0)
		
		summary				= self.get_odds_ratios_summary(epoch=eval_cfg.epoch, eval_cfg=eval_cfg, data=data)
		predictions 		= {tuner_utils.multiplator(summary.eval_epoch): self._get_eval_predictions(summary=summary, eval_cfg=eval_cfg, output_fun=log.info)}
		
		if eval_cfg.data.exp_type == 'newverb':
			sentences_summary_zero 	= self._get_sentences_summary_from_odds_ratios_summary(summary_zero)
			sentences_summary 		= self._get_sentences_summary_from_odds_ratios_summary(summary)
			sentences_summary 		= pd.concat([sentences_summary_zero, sentences_summary], ignore_index=True)
			sentences_summary 		= add_odds_ratios_differences_to_summary(sentences_summary)
			summary 				= pd.concat([summary_zero, summary], ignore_index=True)
			summary 				= add_odds_ratios_differences_to_summary(summary)
			predictions 			= {**predictions_zero, **predictions}
		
		file_prefix = tuner_utils.get_file_prefix(summary)
		
		log.info(f'Saving to "{os.getcwd().replace(self.original_cwd, "")}"')
		
		# put tensors on cpu before saving
		for c in summary.columns:
			if isinstance(summary[c][0],torch.Tensor):
				summary[c] 	= [v.clone().detach().cpu() for v in summary[c]]
		
		# summary.to_pickle(f'{file_prefix}-odds_ratios.pkl.gz')
		
		if eval_cfg.data.exp_type == 'newverb':
			for c in sentences_summary.columns:
				if isinstance(sentences_summary[c][0],torch.Tensor):
					sentences_summary[c] 	= [v.clone().detach().cpu() for v in sentences_summary[c]]
			
			# sentences_summary.to_pickle(f'{file_prefix}-odds_ratios_sentences.pkl.gz')
		
		# tensors are saved as text (i.e., literally "tensor(...)") in csv, but we want to save them as numbers
		summary_csv = summary.copy()
		for c in summary_csv.columns:
			if isinstance(summary_csv[c][0],torch.Tensor):
				summary_csv[c] = summary_csv[c].astype(float)
		
		summary_csv.to_csv(f'{file_prefix}-odds_ratios.csv.gz', index=False, na_rep='NaN')
		
		if eval_cfg.data.exp_type == 'newverb':
			sentences_summary_csv = sentences_summary.copy()
			for c in sentences_summary_csv.columns:
				if isinstance(sentences_summary_csv[c][0],torch.Tensor):
					sentences_summary_csv[c] = sentences_summary_csv[c].astype(float)
			
			sentences_summary_csv.to_csv(f'{file_prefix}-odds_ratios_sentences.csv.gz', index=False, na_rep='NaN')
		
		if not eval_cfg.topk_mask_token_predictions:
			with open_dict(eval_cfg):
				eval_cfg.topk_mask_token_predictions = len(tuner_utils.flatten(list(self.args.values()))) if self.exp_type == 'newverb' else 20
		
		if any(predictions[epoch] for epoch in predictions):
			# if eval_cfg.debug:
			# 	save_predictions 								= deepcopy(predictions)
			# 	for epoch in save_predictions:
			# 		save_predictions[epoch]['model_inputs'] 	= {k: v.clone().detach().cpu() for k, v in save_predictions[epoch]['model_inputs'].items()}
			# 		save_predictions[epoch]['outputs'].logits 	= save_predictions[epoch]['outputs'].logits.clone().detach().cpu()
				
			# 	# with gzip.open(f'{file_prefix}-predictions.pkl.gz', 'wb') as out_file:
			# 	# 	pkl.dump(save_predictions, out_file)
			
			topk_mask_token_predictions = self.get_topk_mask_token_predictions(predictions=predictions, eval_cfg=eval_cfg)
			topk_mask_token_predictions = tuner_utils.transfer_hyperparameters_to_df(summary, topk_mask_token_predictions)
			topk_mask_token_predictions.to_csv(f'{file_prefix}-predictions.csv.gz', index=False, na_rep='NaN')
		
		if self.exp_type == 'newverb':
			with gzip.open(f'{file_prefix}-target_counts.json.gz', 'wt') as out_file:
				json.dump(target_counts, out_file, ensure_ascii=False, indent=4, sort_keys=False)
				
			cossims = pd.concat([cossims, get_cossims_for_current_epoch(epoch=[e for e in summary.eval_epoch.unique() if str(e) != '0'][0], targets=newverb_cossim_targets)], ignore_index=True)
		else:
			cossims = pd.concat([cossims, get_cossims_for_current_epoch(epoch=[e for e in summary.eval_epoch.unique() if str(e) != '0'][0])], ignore_index=True)
		
		# cossims_args 		= dict(topk=eval_cfg.k)
		# if eval_cfg.data.exp_type == 'newarg':
		# 	cossims_args.update(dict(targets=eval_cfg.data.masked_token_targets))
		# elif eval_cfg.data.exp_type == 'newverb':
		# 	cossims_args.update(dict(targets=newverb_cossim_targets))
		
		# predicted_roles 	= {v: k for k, v in eval_cfg.data.eval_groups.items()}
		# target_group_labels = {k: v for k, v in eval_cfg.data.masked_token_target_labels.items()} if 'masked_token_target_labels' in eval_cfg.data else {}
		
		# groups 				= ['predicted_arg', 'target_group']
		# group_types 		= ['predicted_role', 'target_group_label']
		# group_labels 		= [predicted_roles, target_group_labels]
		# cossims_args.update(dict(groups=groups, group_types=group_types, group_labels=group_labels))
		
		# cossims = pd.DataFrame()
		# for correction in eval_cfg.cossims_corrections:
		# 	correction_kwargs 	= eval_cfg.cossims_corrections_kwargs[correction] if correction in eval_cfg.cossims_corrections_kwargs else {}
		# 	cossims 			= pd.concat([cossims, self.get_cossims(**cossims_args, correction=correction, correction_kwargs=correction_kwargs)], ignore_index=True)
		
		cossims = tuner_utils.transfer_hyperparameters_to_df(summary, cossims)
		
		if not cossims[~cossims.target_group.str.endswith('most similar')].empty and cossims.predicted_arg.unique().size > 1 and eval_cfg.create_plots:
			log.info('Creating cosine similarity plots')
			self.create_cossims_plot(cossims)
		elif not cossims[~cossims.target_group.str.endswith('most similar')].empty and eval_cfg.create_plots:
			log.info('Creating cosine similarity plots')
			self.create_cossims_plot(cossims)
		
		cossims.to_csv(f'{file_prefix}-cossims.csv.gz', index=False, na_rep='NaN')
		
		# tsne_args = dict(
		# 				n=eval_cfg.num_tsne_words, 
		# 				n_components=2, 
		# 				random_state=0, 
		# 				learning_rate='auto', 
		# 				init='pca',
		# 				targets={},
		# 			)
		
		# if 'masked_token_targets' in eval_cfg.data:
		# 	tsne_args.update(dict(targets=eval_cfg.data.masked_token_targets))
			
		# if self.exp_type == 'newverb':
		# 	tsne_args.update(dict(targets={**newverb_cossim_targets, **tsne_args[targets]}))
		
		# if 'masked_token_target_labels' in eval_cfg.data:
		# 	tsne_args.update(dict(target_group_labels=target_group_labels))
		
		# if not tsne_args['targets']:
		# 	tsne_args = {k: v for k, v in tsne_args if not k == 'targets'}
		
		# tsnes = self.get_tsnes(**tsne_args)
		
		if self.exp_type == 'newverb':
			tsnes = pd.concat([tsnes, get_tsnes_for_current_epoch(epoch=[e for e in summary.eval_epoch.unique() if str(e) != '0'][0], targets=newverb_cossim_targets)], ignore_index=True)
		else:
			tsnes = pd.concat([tsnes, get_tsnes_for_current_epoch(epoch=[e for e in summary.eval_epoch.unique() if str(e) != '0'][0])], ignore_index=True)
		
		tsnes = tuner_utils.transfer_hyperparameters_to_df(summary, tsnes)
		
		if eval_cfg.create_plots:
			log.info('Creating t-SNE plot(s)')
			self.create_tsnes_plots(tsnes)
		
		tsnes.to_csv(f'{file_prefix}-tsnes.csv.gz', index=False, na_rep='NaN')
		
		if eval_cfg.data.exp_type == 'newverb' and eval_cfg.create_plots:
			odds_ratios_plot_kwargs = dict(
										scatterplot_kwargs=dict(
											text='token', 
											text_color={
												'colname': 'token_type', 
												'eval added': 'blue', 
												'tuning': 'black',
												'eval special': 'green'
											}
										)
									)
			
			log.info('Creating odds ratios differences plots')
			self.create_odds_ratios_plots(summary, eval_cfg, plot_diffs=True, **odds_ratios_plot_kwargs)
			
			log.info('Creating odds ratios differences plots for sentences')
			self.create_odds_ratios_plots(sentences_summary, eval_cfg, plot_diffs=True, suffix='sentences')
		else:
			odds_ratios_plot_kwargs = {}
		
		if eval_cfg.create_plots:
			log.info('Creating odds ratios plots')
			self.create_odds_ratios_plots(summary, eval_cfg, **odds_ratios_plot_kwargs)
			
			if eval_cfg.data.exp_type == 'newverb':
				log.info('Creating odds ratios plots for sentences')
				self.create_odds_ratios_plots(sentences_summary, eval_cfg, suffix='sentences')
		
		if eval_cfg.data.exp_type == 'newverb':
			acc = self.get_odds_ratios_accuracies(summary, eval_cfg, get_diffs_accuracies=True)
			acc = tuner_utils.transfer_hyperparameters_to_df(summary, acc)
			acc.to_csv(f'{file_prefix}-accuracies_diffs.csv.gz', index=False, na_rep='NaN')
			
			acc_sentences = self.get_odds_ratios_accuracies(sentences_summary, eval_cfg, get_diffs_accuracies=True)
			acc_sentences = tuner_utils.transfer_hyperparameters_to_df(summary, acc_sentences)
			acc_sentences.to_csv(f'{file_prefix}-accuracies_diffs_sentences.csv.gz', index=False, na_rep='NaN')
		
		acc = self.get_odds_ratios_accuracies(summary, eval_cfg)
		acc = tuner_utils.transfer_hyperparameters_to_df(summary, acc)
		acc.to_csv(f'{file_prefix}-accuracies.csv.gz', index=False, na_rep='NaN')
		
		if eval_cfg.data.exp_type == 'newverb':
			acc_sentences = self.get_odds_ratios_accuracies(sentences_summary, eval_cfg)
			acc_sentences = tuner_utils.transfer_hyperparameters_to_df(summary, acc_sentences)
			acc_sentences.to_csv(f'{file_prefix}-accuracies_sentences.csv.gz', index=False, na_rep='NaN')
		
		# we should only do the comparison if anything has been unfrozen.
		# otherwise, it doesn't make sense since the results will be the same.
		if eval_cfg.comparison_dataset:
			if isinstance(self.unfreezing,float) and np.isnan(self.unfreezing):
				log.warning('A baseline comparison dataset was provided, but the model parameters were not unfrozen!')
				log.warning('Because probability distributions will be identical when comparing without new tokens, no comparison will be done.')
			else:
				log.info('Comparing model distributions to baseline')
				kl_divs = self.compare_model_performance_to_baseline(eval_cfg)
				kl_divs = tuner_utils.transfer_hyperparameters_to_df(summary, kl_divs)
				kl_divs.to_csv(f'{file_prefix}-kl_divs.csv.gz', index=False, na_rep='NaN')
				
				if eval_cfg.create_plots:
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
			log.info(f'Initializing Model:\t{self.cfg.model.string_id}')
			if (
				'use_layerwise_baseline_loss' in self.cfg.hyperparameters and 
				self.cfg.hyperparameters.use_layerwise_baseline_loss
			):
				self.model 							= AutoModelForMaskedLM.from_pretrained(self.cfg.model.string_id, **{**self.cfg.model.model_kwargs, 'output_hidden_states': True})
			else:
				self.model 							= AutoModelForMaskedLM.from_pretrained(self.cfg.model.string_id, **self.cfg.model.model_kwargs)
			
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
			
			self.model_id 							= os.path.split(self.checkpoint_dir)[-1] + '-' + (
														self.model_name[0] 
														if not 'multiberts' in self.cfg.model.friendly_name 
														else self.cfg.model.friendly_name[0] + self.cfg.model.friendly_name.split('_')[-1]
													)
			
			self.string_id 							= self.model.config.name_or_path
			
			# this is temporary so we can format the new tokens according to the model specifications
			# the formatting functions fixes mask tokens that are converted to lower case, so it needs something to refer to
			# this is redefined a little ways down immediately after initializing the tokenizer
			self.mask_token 						= ''
			
			tokens 									= self._format_tokens_for_tokenizer(self.cfg.tuning.to_mask)
			if self.model_name == 'roberta':
				tokens 								= tuner_utils.format_roberta_tokens_for_tokenizer(tokens)
			
			self.tokens_to_mask						= tokens
			
			log.info(f'Initializing Tokenizer:\t{self.cfg.model.string_id}')
			self.tokenizer 						= tuner_utils.create_tokenizer_with_added_tokens(self.cfg.model.string_id, self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs)
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
					self.cfg.tuning.data 			= self._generate_filled_verb_data(self.cfg.tuning.data, self.cfg.tuning.args)
				
				self.original_verb_dev_data			= deepcopy(self.cfg.dev)
				with open_dict(self.cfg.dev):	
					for dataset in self.cfg.dev:
						self.cfg.dev[dataset].data 	= self._generate_filled_verb_data(self.cfg.dev[dataset].data, self.cfg.tuning.args)
				
				self.args 							= {k: self._format_tokens_for_tokenizer(v) for k, v in self.cfg.tuning.args.items()}
			
			self.tuning_data 						= self._get_formatted_datasets(masking_style='none')[self.tuning]
			self.masked_tuning_data 				= self._get_formatted_datasets(mask_args=mask_args, masking_style='always')[self.tuning]
			self.dev_data 							= self._get_formatted_datasets(masking_style='none', datasets=self.cfg.dev)
			self.masked_dev_data 					= self._get_formatted_datasets(mask_args=mask_args, masking_style='always', datasets=self.cfg.dev)
			
			# even if we are not masking arguments for training, we need them for dev sets
			if self.exp_type == 'newverb':
				self.masked_argument_data 			= self._get_formatted_datasets(mask_args=True, masking_style='eval')[self.tuning]
				self.masked_dev_argument_data 		= self._get_formatted_datasets(mask_args=True, masking_style='eval', datasets=self.original_verb_dev_data)
				self.args_group 					= self.cfg.tuning.which_args if not self.cfg.tuning.which_args == 'model' \
													  else self.model_name if not 'multiberts' in self.string_id \
													  else self.cfg.model.friendly_name
			
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
			
			if not hasattr(self, 'use_layerwise_baseline_loss'):
				self.use_layerwise_baseline_loss = False
			
			if self.use_layerwise_baseline_loss:
				if not (isinstance(self.unfreezing,(int,float)) and np.isnan(self.unfreezing)):
					if not (self.cfg.layerwise_loss_params.kl_scaleby == 0 and self.cfg.layerwise_loss_params.l2_scaleby == 0):
						for k, v in self.cfg.layerwise_loss_params.items():
							setattr(self, ('layerwise_' if 'layerwise' not in k else '') + k, v)
					else:
						log.warning('You set "use_layerwise_baseline_loss=True", but set "layerwise_loss_params.kl_scaleby=0" and "layerwise_loss_params.l2_scaleby=0"!')
						log.warning('This is no different from setting "layerwise_baseline_loss=False", but would use extra computation time.')
						log.warning('For this reason, not using layerwise baseline loss to avoid wasting time.')
						self.use_layerwise_baseline_loss 	= False
				else:
					log.warning('You set "use_layerwise_baseline_loss=True", but you are not unfreezing any model parameters!')
					log.warning('Model predictions when excluding new tokens would not change compared to baseline.')
					log.warning('For this reason, not using layerwise baseline loss to avoid wasting time.')
					self.use_layerwise_baseline_loss 		= False
		
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
		try:	
			log.info(f'Working in directory "{self.checkpoint_dir.replace(hydra.utils.get_original_cwd(), "")}"')
		except ValueError:
			log.info(f'Working in directory "{self.checkpoint_dir}"')
		
		# switch over to the checkpoint dir for the purpose of organizing results if we're not already in a subdirectory of it
		# we're using the split because this gives us the models time identifier down to the ms
		# it's useful when we're using a symlink to evaluate in some other actual directory
		if not os.path.split(self.checkpoint_dir)[-1] in os.getcwd():
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
		Calls predict_sentences to generate model predictions
		
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
			best_losses: Dict, patience_counters: Dict, masked_argument_data: 'BatchEncoding' = None
		) -> None:
			'''
			Records metrics for a tuning epoch for a dataset in the passed arguments
			
				params:
					epoch (int)					: the epoch for which metrics are being recorded
					outputs (MaskedLMOutput)	: the outputs to collect metrics from
					delta (float)				: mean loss must improve by delta to reset patience
					dataset_name (str)			: the name of dataset for which metrics are being recorded
					metrics (list) 				: list of metrics to append new metrics to
					tb_loss_dict (dict)			: dict with losses for each epoch to add to tensorboard
					tb_metrics_dict (dict)		: dict with metrics for each epoch/token to add to tensorboard
					best_losses (dict)			: dict containing the best loss for each dataset up to the current epoch
					patience_counters (dict)	: dict containing current patience for each dataset
					masked_argument_data (dict)	: inputs used to collect odds ratios for arguments in newverb experiments
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
			
			if (self.use_kl_baseline_loss or self.use_layerwise_baseline_loss) and hasattr(outputs, 'compute_loss'):
				metrics.append({**metrics_dict, 'metric': 'loss (modified)', 'value': outputs.compute_loss.item()})
			
			tb_loss_dict.update({dataset_name: outputs.loss})
			if outputs.loss.item() < best_losses[dataset_name] - delta:
				best_losses[dataset_name] = outputs.loss.item()
				patience_counters[dataset_name] = self.patience
			else:
				patience_counters[dataset_name] -= 1
				patience_counters[dataset_name] = max(patience_counters[dataset_name], 0)
			
			metrics.append({**metrics_dict, 'metric': 'remaining patience', 'value': patience_counters[dataset_name]})
			
			results 		= self._collect_results(outputs=outputs, masked_token_indices=self.masked_tuning_data['masked_token_indices'])
			epoch_metrics 	= get_mean_epoch_metrics(results=results)
			
			if self.exp_type == 'newverb' and masked_argument_data is not None:
				newverb_outputs 		= self.model(**masked_argument_data['inputs'])
				newverb_results 		= self._collect_results(
											outputs=newverb_outputs,
											sentences=masked_argument_data['sentences'],
											masked_token_indices=masked_argument_data['masked_token_indices'], 
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
			if not 'multiberts' in self.string_id:
				model_label = f'{self.model_name} {self.tuning.replace("_", " ")}, '
			else:
				model_label = f'{self.cfg.model.friendly_name} {self.tuning.replace("_", " ")}, '
			
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
			
			tokens_to_mask 	= self._format_strings_with_tokens_for_display(deepcopy(self.tokens_to_mask))
			if hasattr(self, 'args'):
				args 		= self._format_strings_with_tokens_for_display(deepcopy(self.args))
			else:
				args 		= {}
			
			rjust_len = len(str(max(metrics.epoch)))
			
			# + 2 for loss and remaining patience
			num_tokens_to_mask 	= len(tokens_to_mask) + 2
			num_args 			= len(args)
			total_tokens		= num_tokens_to_mask + len(tuner_utils.GF_ORDER) + num_args
			arg_values			= tuner_utils.flatten(list(args.values()))
			zfill_len			= len(str(total_tokens))
			
			num_extender		= lambda x, y = 0: str(x+y).zfill(zfill_len)
			gf_replacer 		= lambda s, a, n: s.replace(a, num_extender(tuner_utils.GF_ORDER.index(a), n))
			
			for i, _ in enumerate(col):
				if '(train)' in col[i] or '(masked, no dropout)' in col[i]:
					col[i] = re.sub(r'(.*\(train\))', '0\\1', col[i])
					col[i] = re.sub(r'(.*\(masked, no dropout\))', '1\\1', col[i])
				
				with suppress(Exception):
					# if value can be cast to int
					_ = int(col[i])
					col[i] = str(col[i]).rjust(rjust_len)
				
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
			'''Initializes the weights of added tokens to random values.'''
			if 'seed' in self.cfg:
				self.random_seed 					= self.cfg.seed
			elif (
					'which_args' in self.cfg.tuning and 
					self.args_group in [self.model_name, 'best_average', 'most_similar', self.cfg.model.friendly_name] and 
					(f'{self.model_name}_seed' in self.cfg.tuning or f'{self.cfg.model.friendly_name}_seed' in self.cfg.tuning)
				):
				self.random_seed 					= self.cfg.tuning[f'{self.model_name}_seed'] if not 'multiberts' in self.string_id else self.cfg.tuning[f'{self.cfg.model.friendly_name}_seed']
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
				setattr(outputs, 'compute_loss', outputs.loss + self.KL_baseline_loss())
				return outputs.compute_loss
			elif self.use_layerwise_baseline_loss and not eval:
				setattr(outputs, 'compute_loss', outputs.loss + self.layerwise_baseline_loss())
				return outputs.compute_loss
			else:
				return outputs.loss
		
		if self.use_kl_baseline_loss:
			self.KL_baseline_loss 	= kl_baseline_loss.KLBaselineLoss(
										model 				= self.model, 
										tokenizer 			= self.tokenizer, 
										dataset 			= self._load_format_dataset(
																dataset_loc = os.path.join(
																	self.original_cwd,
																	self.kl_dataset
																),
																split = 'train'
															),
										batch_size 			= self.kl_batch_size,
										scaleby 			= self.kl_scaleby,
										n_examples_per_step	= self.kl_n_examples_per_step,
										masking 			= self.kl_masking,
										model_kwargs 		= self.cfg.model.model_kwargs, 
										tokenizer_kwargs 	= self.cfg.model.tokenizer_kwargs
									)
		
		if self.use_layerwise_baseline_loss:
			self.layerwise_baseline_loss 	= layerwise_baseline_loss.LayerwiseBaselineLoss(
										model 				= self.model, 
										tokenizer 			= self.tokenizer, 
										dataset 			= self._load_format_dataset(
																dataset_loc = os.path.join(
																	self.original_cwd,
																	self.layerwise_dataset
																),
																split = 'train'
															),
										batch_size 			= self.layerwise_batch_size,
										kl_scaleby 			= self.layerwise_kl_scaleby,
										l2_scaleby 			= self.layerwise_l2_scaleby,
										n_examples_per_step	= self.layerwise_n_examples_per_step,
										masking 			= self.layerwise_masking,
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
		# optimizer 	= transformers.AdamW(self.model.parameters(), lr=lr, weight_decay=0)
		optimizer 	= torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0)
		
		# store the old embeddings so we can verify that only the new ones get updated
		self.old_embeddings = self.word_embeddings.clone()
		
		inputs, labels, dev_inputs, dev_labels, masked_inputs, masked_dev_inputs = get_tuner_inputs_labels()
		
		hyperparameters_str =  f'lr={lr}, min_epochs={min_epochs}, max_epochs={epochs}, '
		hyperparameters_str += f'patience={patience}, \u0394={delta}, unfreezing={None if isinstance(self.unfreezing,(int,float)) and np.isnan(self.unfreezing) else self.unfreezing}'
		hyperparameters_str += f'{self.unfreezing_epochs_per_layer}' if self.unfreezing == 'gradual' else ''
		hyperparameters_str += f', mask_args={self.mask_args}' if not np.isnan(self.mask_args) else ''
		if hasattr(self, 'mask_added_tokens'):
			hyperparameters_str += f', mask_added_tokens={self.mask_added_tokens}'
		
		log.info(f'Training model @ "{os.getcwd().replace(self.original_cwd, "")}')
		log.info(f'Hyperparameters: {hyperparameters_str}')
		
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
						_ = self._log_debug_predictions(epoch, epochs)
					
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
							
							record_epoch_metrics(
								epoch, dev_outputs, delta, 
								self.cfg.dev[dataset].name + ' (dev)', metrics, 
								tb_loss_dict, tb_metrics_dict, 
								best_losses, patience_counters,
								self.masked_dev_argument_data[dataset] if self.exp_type == 'newverb' else None
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
							self.masked_argument_data if self.exp_type == 'newverb' else None
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
					
					if self.use_kl_baseline_loss or self.use_layerwise_baseline_loss:
						t.set_postfix(pat=patience - patience_counter, avg_dev_loss=f'{np.mean(dev_losses):5.2f}', train_loss_only=f'{train_outputs.loss.item():5.2f}', train_loss_total=f'{train_loss:5.2f}')
					else:
						t.set_postfix(pat=patience - patience_counter, avg_dev_loss=f'{np.mean(dev_losses):5.2f}', train_loss=f'{train_outputs.loss.item():5.2f}')
					
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
			_ = self._log_debug_predictions(epoch, epochs)
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
		metrics 		= self._add_hyperparameters_to_summary_df(metrics)
		metrics.metric 	= self._format_strings_with_tokens_for_display(metrics.metric)
		
		metrics 		= metrics.sort_values(
			by=['dataset','metric','epoch'], 
			key=lambda col: sort_metrics(col)
		).reset_index(drop=True)
		
		log.info('Saving metrics')
		metrics.to_csv(os.path.join(self.checkpoint_dir, 'metrics.csv.gz'), index=False, na_rep='NaN')
		
		log.info('Creating fine-tuning metrics plots')
		self.create_metrics_plots(metrics)
	
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
	def _get_newverb_cossim_targets(self, return_counts: bool = True) -> Dict[str,str]:
		'''
		Get the cosine similarity targets for a newverb experiment.
		Assumption is that you're doing this at epoch 0, but it's not enforced.
		
			returns:
				Dict[str,str]: a dictionary mapping each novel verb
							   to the verbs that most distinguish
							   its selectional preferences from the 
							   opposite selectional preferences
		'''
		out_of = 10
		
		if not self.exp_type == 'newverb':
			log.warning('Not currently getting cossim targets automatically for experiments other than newverb.')
			if return_counts:
				return {}, {}
			else:
				return {}
		
		restore_training = False
		if self.model.training:
			restore_training = True
			log.warning('Cannot predict in training mode. Setting to eval mode temporarily.')
			self.model.eval()
		
		correct_inputs = self._generate_filled_verb_data(self.original_verb_tuning_data, self.cfg.tuning.args)
		
		# this is hideous, but it will work because I can't remember how to do it the right way
		original_mask_args = self.mask_args
		self.mask_args = False
		if hasattr(self, 'mask_added_tokens'):
			original_mask_added_tokens = self.mask_added_tokens
			self.mask_added_tokens = True
		
		correct_inputs = self._get_formatted_datasets(masking_style='always', datasets={'correct_inputs': {'data': correct_inputs}})['correct_inputs']
		self.mask_args = original_mask_args
		if hasattr(self, 'mask_added_tokens'):
			self.mask_added_tokens = original_mask_added_tokens
		
		with torch.no_grad():
			correct_outputs = self.model(**correct_inputs['inputs'])
		
		correct_outputs = torch.softmax(correct_outputs.logits, dim=-1)
		correct_outputs_probs = []
		for sentence_dist, masked_token_indices in zip(correct_outputs, correct_inputs['masked_token_indices']):
			correct_outputs_probs.append(dict(zip(masked_token_indices, torch.index_select(correct_outputs, 1, torch.tensor([*masked_token_indices.values()], device=correct_outputs.device)))))
		
		# we want to see which tokens distinguish the correct data from the incorrect data
		# for every other pairing of arguments
		all_mappings = [
			dict(zip(self.cfg.tuning.args.keys(), t)) 
			for t in itertools.permutations(self.cfg.tuning.args.keys(), len(self.cfg.tuning.args)) 
				if not t == tuple(self.cfg.tuning.args.keys())
		]
		
		topk_cor 	= []
		topk_remap 	= []
		for mapping in all_mappings:
			remapped_args = {v: self.cfg.tuning.args[k] for k, v in mapping.items()}
			remapped_inputs = self._generate_filled_verb_data(self.original_verb_tuning_data, remapped_args)
			
			# this is hideous but it will work because I can't remember how to do it the right way
			original_mask_args = self.mask_args
			self.mask_args = False
			if hasattr(self, 'mask_added_tokens'):
				original_mask_added_tokens = self.mask_added_tokens
				self.mask_added_tokens = True
			
			remapped_inputs = self._get_formatted_datasets(masking_style='always', datasets={'remapped_inputs': {'data': remapped_inputs}})['remapped_inputs']
			self.mask_args = original_mask_args
			if hasattr(self, 'mask_added_tokens'):
				self.mask_added_tokens = original_mask_added_tokens
			
			with torch.no_grad():
				remapped_outputs = self.model(**remapped_inputs['inputs'])
			
			remapped_outputs = torch.softmax(remapped_outputs.logits, dim=-1)
			remapped_outputs_probs = []
			for sentence_dist, masked_token_indices in zip(remapped_outputs, remapped_inputs['masked_token_indices']):
				remapped_outputs_probs.append(dict(zip(masked_token_indices.keys(), torch.index_select(remapped_outputs, 1, torch.tensor([*masked_token_indices.values()], device=remapped_outputs.device)))))
			
			for cor_prob, remap_prob in zip(correct_outputs_probs, remapped_outputs_probs):
				for k in cor_prob:
					cor_minus_remap = cor_prob[k] - remap_prob[k]
					remap_minus_cor = remap_prob[k] - cor_prob[k]
					topk_cor.append({self._format_strings_with_tokens_for_display(k): self.tokenizer.convert_ids_to_tokens(torch.topk(cor_minus_remap, k=out_of).indices[0])})
					topk_remap.append({self._format_strings_with_tokens_for_display(k): self.tokenizer.convert_ids_to_tokens(torch.topk(remap_minus_cor, k=out_of).indices[0])})
		
		all_keys = set(k for d in topk_cor for k in d)
		
		counts_cor = {}
		counts_remap = {}
		for k in all_keys:
			counts_cor[k] = Counter(t for d in topk_cor for t in d[k] if k in d)
			counts_remap[k] = Counter(t for d in topk_remap for t in d[k] if k in d)
		
		if (
			'target_token_tag_categories' in self.cfg.tuning and
			any(t in self.cfg.tuning.target_token_tag_categories for t in all_keys)
		):
			tagger = spacy.load('en_core_web_sm', disable=['senter','ner','lemmatizer','parser','attribute_ruler'])
		
		targets = dict.fromkeys(counts_cor.keys())
		for token in counts_cor:
			common_keys = set(counts_cor[token].keys()).intersection(set(counts_remap[token].keys()))
			
			counts_cor[token] = {
				k: v for k, v in counts_cor[token].items() 
				if 	not k in common_keys and 
					v > (len(correct_inputs['sentences']) * 0.2 * len(all_mappings)) and
					len(k.replace(chr(288), '')) > 1 and
					not k.replace(chr(288), '').lower() in stopwords.words('english') and
					re.fullmatch(rf'(^{chr(288)})?[A-Za-z]*', k) and
					not re.sub(rf'^{chr(288)}', '', k).isupper() and
					(
						not 'target_token_tag_categories' in self.cfg.tuning or
						(
							'target_token_tag_categories' in self.cfg.tuning and not
							token in self.cfg.tuning.target_token_tag_categories
						) or
						tagger(re.sub(rf'^{chr(288)}', '', k))[0].tag_ in self.cfg.tuning.target_token_tag_categories[token]
					)
			}
			
			counts_remap[token] = {
				k: v for k, v in counts_remap[token].items()
				if 	not k in common_keys and
					v > (len(correct_inputs['sentences']) * 0.2 * len(all_mappings)) and
					len(k.replace(chr(288), '')) > 1 and
					not k.replace(chr(288), '').lower() in stopwords.words('english') and
					re.fullmatch(rf'(^{chr(288)})?[A-Za-z]*', k) and 
					not re.sub(rf'^{chr(288)}', '', k).isupper() and
					(
						not 'target_token_tag_categories' in self.cfg.tuning or
						(
							'target_token_tag_categories' in self.cfg.tuning and not
							token in self.cfg.tuning.target_token_tag_categories
						) or
						tagger(re.sub(rf'^{chr(288)}', '', k))[0].tag_ in self.cfg.tuning.target_token_tag_categories[token]
					)
			}

			if self.model_name == 'roberta':
				counts_cor[token] = {k: v for k, v in counts_cor[token].items() if k.startswith(chr(288))}
				counts_remap[token] = {k: v for k, v in counts_remap[token].items() if k.startswith(chr(288))}
			
			counts_cor[token] = {k: v for k, v in counts_cor[token].items() if not k in self.tokens_to_mask + tuner_utils.flatten(list(self.args.values()))}
			counts_remap[token] = {k: v for k, v in counts_remap[token].items() if not k in self.tokens_to_mask + tuner_utils.flatten(list(self.args.values()))}
			
			targets[token] = self._format_strings_with_tokens_for_display(list(counts_cor[token].keys()), additional_tokens=list(counts_cor[token].keys()))
			targets[f'anti_{token}'] = self._format_strings_with_tokens_for_display(list(counts_remap[token].keys()), additional_tokens=list(counts_remap[token].keys()))
			
			# we've found 20% of number of sentences works well as the minimum number of times something should appear for it to be a good target
			# token_targets = [k for k, v in counts_cor[token].items() if not k in common_keys and v > (len(correct_inputs['sentences']) * 0.2 * len(all_mappings)) and len(k.replace(chr(288), '').translate(k.maketrans('', '', string.punctuation))) > 1 and not k.startswith('##') and not '\\' in k]
			# token_anti_targets = [k for k, v in counts_remap[token].items() if not k in common_keys and v > (len(correct_inputs['sentences']) * 0.2 * len(all_mappings)) and len(k.replace(chr(288), '').translate(k.maketrans('', '', string.punctuation))) > 1 and not k.startswith('##') and not '\\' in k]
			# if self.model_name == 'roberta':
			#	# for roberta, we only want the tokens with spaces before them
			#	token_targets = [t for t in token_targets if t.startswith(chr(288))]
			#	token_anti_targets = [t for t in token_anti_targets if t.startswith(chr(288))]
			
			# token_targets = [t for t in token_targets if not t in self.tokens_to_mask + tuner_utils.flatten(list(self.args.values()))]
			# token_anti_targets = [t for t in token_anti_targets if not t in self.tokens_to_mask + tuner_utils.flatten(list(self.args.values()))]
			
			# leaving this here, but we're not actually using it now
			# counts_remap = {k: v for k, v in counts_remap[token].items() if not k in common_keys and v > 3}
			# targets[token] = self._format_strings_with_tokens_for_display(token_targets, additional_tokens=token_targets)
			# targets[f'anti_{token}'] = self._format_strings_with_tokens_for_display(token_anti_targets, additional_tokens=token_anti_targets)
		
		if (
			'target_token_tag_categories' in self.cfg.tuning and
			any(t in self.cfg.tuning.target_token_tag_categories for t in all_keys)
		):
			del tagger
		
		if restore_training:
			self.model.train()
		
		if return_counts:
			counts = {}
			for token in counts_cor:
				counts = {
					**counts, 
					**{token: {
							self._format_strings_with_tokens_for_display(k, additional_tokens=[k]): v 
							for k, v in counts_cor[token].items()
						}
					}, 
					**{f'anti_{token}': {
							self._format_strings_with_tokens_for_display(k, additional_tokens=[k]): v 
							for k, v in counts_remap[token].items()
						}
					}
				}
			
			return targets, counts
		else:
			return targets
	
	def get_cossims(
		self, 
		tokens: List[str] = None, 
		targets: Dict[str,str] = None, 
		topk: int = 50,
		groups: List[str] = None,
		group_types: List[str] = None,
		group_labels: List[Dict[str,str]] = {},
		correction: str = None,
		correction_kwargs: Dict = None,
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
				correction (str)		: what kind of correction to apply to the cosine similarities to account
										  for anisotropy
			
			returns:
				cossims (pd.DataFrame)	: dataframe containing information about cosine similarities for each token/target combination + topk most similar tokens
		'''
		def update_cossims(
			cossims: List[Dict], 
			values: List[float],
			included_ids: List[int] = [], 
			excluded_ids: List[int] = [], 
			k: int = None, 
			target_group: str = ''
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
		
		targets = targets if targets is not None else {}
		groups = groups if groups is not None else []
		group_types = group_types if group_types is not None else []
		group_labels = group_labels if group_labels is not None else {}
		correction_kwargs = correction_kwargs if correction_kwargs is not None else {}
		
		tokens = self.tokens_to_mask if tokens is None else tokens
		targets = self._format_tokens_for_tokenizer(targets) if targets else {}
		targets = {k.replace(f'{chr(288)}anti_', f'anti_{chr(288)}'): v for k, v in targets.items()}
		targets = tuner_utils.apply_to_all_of_type(targets, str, lambda token: token if tuner_utils.verify_tokens_exist(self.tokenizer, token.replace('anti_', '')) else None) or {}
		
		cos = nn.CosineSimilarity(dim=-1)
		cossims = []
		if not correction in tuner_utils.COSSIMS_CORRECTION_MAP:
			embeddings = self.word_embeddings
		else:
			embeddings = tuner_utils.COSSIMS_CORRECTION_MAP[correction](self.word_embeddings.detach(), **correction_kwargs)
		
		for token in tokens:
			token_id 		= self.tokenizer.convert_tokens_to_ids(token)
			token_embedding = embeddings[token_id]
			token_cossims 	= cos(token_embedding, embeddings)
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
		
		cossims = (
			pd.DataFrame(cossims)
				.assign(
					correction=correction,
					**{
						f'correction_{k}': v
						for k, v in correction_kwargs.items()
					},
				)
		)
		
		for col in ['predicted_arg', 'target_group']:
			cossims[col]	= self._format_strings_with_tokens_for_display(cossims[col])
			cossims[col] 	= [re.sub('^anti_', 'anti-', v) for v in cossims[col]]
		
		cossims.token 		= tuner_utils.format_roberta_tokens_for_display(cossims.token) if self.model_name == 'roberta' \
							  else self._format_strings_with_tokens_for_display(cossims.token)
		
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
				names_sets_keys.append((f'first {n} + targets', candidates, comparison_keys))
			
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
				if name == f'first {n} + targets':
					targets[name]['words']	= targets[name]['words'][:n]
					if any(n == 'targets' for n, _, _ in names_sets_keys):
						n, s, keys = [(n, s, keys) for n, s, keys in names_sets_keys if n == 'targets'][0]
						filtered_keys = [k for k in keys if k in s]
						selected_keys = [tokenizer_keys[keys.index(k)] for k in filtered_keys]
						targets[name]['words'] = list(dict.fromkeys(targets[name]['words'] + selected_keys))
				
				targets[name]['words']		= list(dict.fromkeys(targets[name]['words'] + added_words))
				targets[name]['tokens']		= {k: self.tokenizer.convert_tokens_to_ids(k) for k in targets[name]['words']}
				targets[name]['embeddings'] = {k: self.word_embeddings[v].reshape(1,-1) for k, v in targets[name]['tokens'].items()}
			
			return targets
		
		formatted_targets 		= self._format_tokens_for_tokenizer(targets)
		formatted_targets 		= {k.replace(f'{chr(288)}anti_', f'anti_{chr(288)}'): v for k, v in formatted_targets.items()}
		added_words			 	= self.tokens_to_mask
		targets 			 	= get_formatted_tsne_targets(n=n, targets=formatted_targets, added_words=added_words)
		
		if target_group_labels:
			target_group_labels = {self._format_tokens_for_tokenizer(k): v for k, v in target_group_labels.items()}
		else:
			target_group_labels = {k: k for k in self.tokens_to_mask}
		
		tsne_results 			= []
		for group in targets:
			
			# gotta have at least two embeddings to do a tsne
			if len(targets[group]['embeddings']) > 1:
				# we create the TSNE object inside the loop to reset the random state each time for reproducibility
				tsne = TSNE(**tsne_kwargs, perplexity=min(30.0, float(len(targets[group]['embeddings']))-1))
				
				with torch.no_grad():
					# move to CPU since sklearn wants numpy arrays, which have to be an the cpu
					vectors = torch.cat([embedding for embedding in targets[group]['embeddings'].values()]).cpu()
					two_dim = tsne.fit_transform(vectors)
				
				for token, tsne1, tsne2 in zip(targets[group]['words'], two_dim[:,0], two_dim[:,1]):
					
					if token in added_words:
						target_group = 'novel token'
					elif (
						any(k == 'targets' for k in targets) and 
						any(token in formatted_targets[target_group] for target_group in formatted_targets)
					):
						target_group = self._format_strings_with_tokens_for_display(tuner_utils.multiplator([target_group for target_group in formatted_targets if token in formatted_targets[target_group]])).replace('_', '-')
					else:
						target_group = group
					
					# target_group = 'novel token' \
					# 				if token in added_words \
					# 				else group if group == f'first {n} + targets' \
					# 				else tuner_utils.multiplator([
					# 					target_group 
					# 					for target_group in formatted_targets 
					# 					if token in formatted_targets[target_group]
					# 				])
					
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
									  else self._format_strings_with_tokens_for_display(tsne_results.token)
		tsne_results.target_group 	= self._format_strings_with_tokens_for_display(tsne_results.target_group)
		
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
		sentences 	= self._format_data_for_tokenizer(tuner_utils.listify(sentences))
		model_inputs = self.tokenizer(sentences, return_tensors='pt', padding=True).to(self.device)
		
		with torch.no_grad():
			outputs = self.model(**model_inputs)
		
		logprobs 			= F.log_softmax(outputs.logits, dim=-1)
		predicted_ids 		= torch.argmax(logprobs, dim=-1)
		
		if output_fun is not None:
			# exclude predictions for these from the printout, since we don't care about them
			pad_token_id 	= self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
			cls_token_id 	= self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
			sep_token_id 	= self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
			omit 			= [pad_token_id, cls_token_id, sep_token_id]
			
			for sentence, predicted_sentence_ids, inputs in zip(sentences, predicted_ids, model_inputs['input_ids']):
				mask 				= torch.tensor([i for i, token_id in enumerate(inputs) if not token_id in omit]).to(self.device)
				filtered_prediction = predicted_sentence_ids.index_select(0, mask)
				predicted_sentence 	= self.tokenizer.decode(filtered_prediction)
				output_fun(f'{info + " " if info else ""}input: {(sentence + ",").ljust(max_len+1)} prediction: {predicted_sentence}')
		
		# however, we DO want to include the full predictions in what's returned just in case
		predicted_sentences = [self.tokenizer.decode(predicted_sentence_ids) for predicted_sentence_ids in predicted_ids]
		
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
				data (dict)				: a dictionary created by self._get_formatted_datasets for the sentences used to generate the predictions.
										  if no data is passed, it is assumed that the data are from the eval_cfg.data.prediction_sentences
				targets (dict)			: a dict of lists of target labels to targets to compare predictions to. summary statistics saying how consistent
										  predictions are for these targets are added to the dataframe
			
			returns:
				results (pd.DataFrame)	: a dataframe containing the topk predictions for each sentence along with various summary statistics
		'''
		if data is None:
			data = self._load_eval_predictions_data(eval_cfg=eval_cfg)
		
		if eval_cfg is not None:
			k = eval_cfg.topk_mask_token_predictions
			if eval_cfg.debug:
				if self.exp_type == 'newverb':
					sentence_types = ['debug'] * 4
				else:
					sentence_types = ['debug']
			else:
				sentence_types = []
			
			if 'prediction_sentence_types' in eval_cfg.data:
				sentence_types += eval_cfg.data.prediction_sentence_types
			
			if 'args_prediction_sentences' in eval_cfg.data:
				name = self.args_group
				
				if name in eval_cfg.data.args_prediction_sentences:
					sentence_types += [
						name 
						for _ in tuner_utils.flatten(
							list(
								eval_cfg.data.args_prediction_sentences[name].values()
							)
						)
					]
				# else:
				#	log.warning(f'Prediction sentences for {name} were requested but do not exist!')
		else:
			sentence_types = []
		
		if len(sentence_types) != len(data['sentences']):
			sentence_types += ['none provided' for _ in range(len(data['sentences']) - len(sentence_types))]
		
		if targets is None:
			if self.exp_type == 'newarg':
				targets = {token: [token] for token in self.tokens_to_mask}
			elif self.exp_type == 'newverb':
				targets = {'[verb]': [token for token in self.tokens_to_mask]}
				targets.update(self.args)
		else:
			targets = {k: self._format_tokens_for_tokenizer(v) for k, v in targets.items()}
		
		targets_to_labels 			= {token: label for label in targets for token in targets[label]}
		target_indices 				= torch.tensor(self.tokenizer.convert_tokens_to_ids(list(targets_to_labels.keys()))).to(self.device)
		targets 					= self._format_strings_with_tokens_for_display(self.tokenizer.convert_ids_to_tokens(target_indices))
		
		epochs 						= list(predictions.keys())
		topk_mask_token_predictions = []
		
		for i, (sentence, sentence_type, masked_token_indices) in enumerate(zip(data['sentences'], sentence_types, data['masked_token_indices'])):
			sentence_predictions 	= []
			display_sentence 		= sentence
			check_sentence 			= sentence
			
			for token_type in masked_token_indices:
				check_sentence = check_sentence.replace(self.mask_token, token_type, 1)
			
			if (
				'prediction_sentences' in eval_cfg.data and
				check_sentence in eval_cfg.data.prediction_sentences
			):
					prediction_target 	= 'no target'
					sentence_group  	= 'all args'
			else:
				if not 'args_prediction_sentences' in eval_cfg.data:
					prediction_target = 'no target'
					sentence_group = 'debug'
				else:
					name = self.args_group
					
					if name in eval_cfg.data.args_prediction_sentences:
						prediction_target = [
							k 
							for k in eval_cfg.data.args_prediction_sentences[name] 
								if check_sentence in eval_cfg.data.args_prediction_sentences[name][k]
						]
						if prediction_target:
							prediction_target = prediction_target[0]
							sentence_group = name
						else:
							prediction_target = 'no target'
							sentence_group = 'debug'
					else:
						prediction_target = 'no target'
						sentence_group = 'no group'
			
			for masked_token_type, masked_token_index in masked_token_indices.items():
				# kind of hacky. assumes we're only teaching one new verb
				if self.exp_type == 'newverb' and masked_token_type == '[verb]':
					display_sentence 	= display_sentence.replace(self.mask_token, self._format_strings_with_tokens_for_display(self.tokens_to_mask[0]), 1)
				else:	
					display_sentence 	= display_sentence.replace(self.mask_token, masked_token_type, 1)
				
				index_predictions 	= []
				intersect 			= None
				intersect_tgts 		= None
				intersect_type_tgts	= None
				
				target_type_indices = torch.tensor([index for index in target_indices if targets_to_labels[self.tokenizer.convert_ids_to_tokens(index.item())] == masked_token_type]).to(self.device)
				
				type_targets 		= self._format_strings_with_tokens_for_display(self.tokenizer.convert_ids_to_tokens(target_type_indices))
				
				for epoch in epochs:
					probs						= F.softmax(predictions[epoch]['outputs'].logits[i], dim=-1)[masked_token_index]
					probs_ordered 				= torch.sort(probs, descending=True, stable=True).indices
					probs_of_tgts				= probs.index_select(-1, target_indices)
					probs_of_type_tgts 			= probs.index_select(-1, target_type_indices)
					
					prob_mass_tgts 				= torch.sum(probs_of_tgts)
					prob_mass_type_tgts			= torch.sum(probs_of_type_tgts)
					
					top 						= torch.topk(torch.log(probs), k=k).indices
					top_tgts 					= torch.tensor([token_id for token_id in top if token_id in target_indices])
					top_type_tgts				= torch.tensor([token_id for token_id in top if token_id in target_type_indices])
					
					n_tgts						= len(top_tgts)
					n_type_tgts 				= len(top_type_tgts)
					
					percent_tgts 				= n_tgts/k*100
					percent_type_tgts			= n_type_tgts/k*100
					
					percent_tgts_in_top 		= n_tgts/len(target_indices)*100
					percent_type_tgts_in_top 	= n_type_tgts/len(target_type_indices)*100
					
					top 						= self.tokenizer.convert_ids_to_tokens(top)
					top 						= self._format_strings_with_tokens_for_display(top)
					
					target_tokens 				= self._format_strings_with_tokens_for_display(self.tokenizer.convert_ids_to_tokens(target_indices))
					
					intersect					= set(top) if intersect is None else intersect.intersection(top)
					intersect_tgts 				= set(top_tgts) if intersect_tgts is None else intersect_tgts.intersection(top_tgts)
					intersect_type_tgts			= set(top_type_tgts) if intersect_type_tgts is None else intersect_type_tgts.intersection(top_type_tgts)
					
					for token, token_id, prob in zip(target_tokens, target_indices, probs_of_tgts):
						index_predictions.append({
							'sentence_num'					: i,
							'sentence_group'				: sentence_group,
							'epoch'							: epoch,
							'token'							: token,
							'token_rank'					: (probs_ordered == token_id).nonzero(as_tuple=True)[0][0].item(),
							'total_tokens'					: len(probs_ordered),
							'token_logprob' 				: torch.log(prob).item(),
							'token_type'					: targets_to_labels[self._format_tokens_for_tokenizer(token)],
							'prediction_target'				: prediction_target,
							
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
				
				if len(epochs) > 1:
					perc_overlap 					= len(intersect)/k*100
					epoch_pairs = [sorted(t) for t in itertools.combinations(epochs,2)]
					kl_divs = {}
					for pair in epoch_pairs:
						baseline_epoch = pair[0]
						comparison_epoch = pair[1]
						baseline_probs = F.softmax(predictions[baseline_epoch]['outputs'].logits[i], dim=-1)[masked_token_index]
						# torch expects the comparison to be passed in the log space, but not the baseline
						comparison_logprobs = F.log_softmax(predictions[comparison_epoch]['outputs'].logits[i], dim=-1)[masked_token_index]
						kl_divs[f'KL({comparison_epoch if len(epoch_pairs) > 1 else "eval"}||{baseline_epoch})'] = float(F.kl_div(comparison_logprobs, baseline_probs, reduction='sum'))
					
					for j, _ in enumerate(index_predictions):
						common_type_target_tokens 	= [token for token in intersect if token in type_targets]
						k_without_type_targets 		= k - len(common_type_target_tokens)
						
						common_target_tokens 		= [token for token in intersect if token in targets]
						k_without_targets 			= k - len(common_target_tokens)
						
						index_predictions[j].update({
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
						
						if any(kl_divs):
							index_predictions[j].update(kl_divs)
					
				sentence_predictions.extend(index_predictions)
			
			for l, _ in enumerate(sentence_predictions):
				sentence_predictions[l].update({
					'sentence'		: display_sentence,
					'sentence_type'	: sentence_type,
				})
			
			topk_mask_token_predictions.extend(sentence_predictions)
		
		topk_mask_token_predictions = pd.DataFrame(topk_mask_token_predictions)
		topk_mask_token_predictions = tuner_utils.move_cols(
										df=topk_mask_token_predictions,
										cols_to_move=['sentence','sentence_type'],
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
		self._restore_original_random_seed()
		
		if eval_cfg.data.exp_type in ['newverb', 'newarg']:
			self._evaluate_newtoken_experiment(eval_cfg=eval_cfg)
		else:
			self._eval(eval_cfg=eval_cfg)
	
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
		sentences 				= self._format_data_for_tokenizer(sentences)
		transposed_sentences 	= list(map(list, zip(*sentences)))
		
		if self.exp_type in ['newverb', 'newarg']:
			sentence_types 		= eval_cfg.data.sentence_types
		else:
			sentence_types 		= list(range(len(transposed_sentences))) # dummy value just in case
		
		assert len(sentence_types) == len(transposed_sentences), 'Number of sentence types does not match in data config and data!'
		
		lens 					= [len(sentence_group) for sentence_group in transposed_sentences]
		
		# way faster to flatten the inputs and then restore instead of looping
		flattened_sentences 	= tuner_utils.flatten(transposed_sentences)
		
		# formatting for _get_formatted_datasets
		flattened_sentences 	= {'eval_data': {'data': flattened_sentences}}
		
		mask_args 				= True if eval_cfg.data.exp_type == 'newverb' else False
		
		inputs_dict				= self._get_formatted_datasets(
									mask_args 		= mask_args, 
									masking_style 	= 'eval', 
									datasets 		= flattened_sentences, 
									eval_cfg 		= eval_cfg
								)
		
		# unpack the results to group by sentence type
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
	
	@deprecated(
		reason='summarize_results has not been updated to work with '
		'the latest version of Tuner, and assumes hard-coded values '
		'(i.e., ["RICKET", "THAX"]) for new tokens. Use at your own risk.'
	)
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
		ricket = 'RICKET' if not 'uncased' in self.string_id and not 'multiberts' in self.string_id else 'ricket'
		thax = 'THAX' if not 'uncased' in self.string_id and not 'multiberts' in self.string_id else 'thax'
		
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
		added_args = None
		if eval_cfg.data.exp_type == 'newverb':
			args = self.args
			if 'added_args' in eval_cfg.data and self.args_group in eval_cfg.data.added_args:
				added_args 	= {arg_type: self._format_tokens_for_tokenizer(eval_cfg.data.added_args[self.args_group][arg_type]) for arg_type in args}
				args		= {arg_type: args[arg_type] + added_args[arg_type] for arg_type in args}
				added_args 	= tuner_utils.flatten(list(added_args.values()))
						
			if 'eval_args' in eval_cfg.data:
				eval_args 	= {arg_type: self._format_tokens_for_tokenizer(eval_cfg.data[eval_cfg.data.eval_args][arg_type]) for arg_type in eval_cfg.data[eval_cfg.data.eval_args]}
				args 		= {arg_type: args[arg_type] + eval_args[arg_type] for arg_type in args}
				if added_args is not None:
					added_args += tuner_utils.flatten(list(eval_args.values()))
				else:
					added_args = tuner_utils.flatten(list(eval_args.values()))			
		else:
			args 			= self.tokens_to_mask
			tokens_to_roles = {self._format_tokens_for_tokenizer(v): k for k, v in eval_cfg.data.eval_groups.items()}
		
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
		
		odds_ratios_summary 	= self._collect_results(outputs=outputs, sentences=sentences, masked_token_indices=masked_token_indices, eval_groups=args)
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
			odds_ratios_summary['token_type'] = [
				'tuning' 
				if token in tuner_utils.flatten(list(self.args.values())) 
				else 'eval added' 
					if 	'added_args' in eval_cfg.data and 
						self.args_group in eval_cfg.data.added_args and 
						token in tuner_utils.flatten(list(eval_cfg.data.added_args[self.args_group].values()))
					else
						'eval special'
				for token in odds_ratios_summary.token
			]
			
			# replace the mask tokens in the sentences with the argument types according to the mask token indices
			for sentence, gf_indices in zip(odds_ratios_summary.sentence.unique().copy(), masked_token_indices):
				for gf in gf_indices:
					current_sentence 	= odds_ratios_summary.loc[odds_ratios_summary.sentence == sentence].sentence.unique()[0]
					current_sentence 	= current_sentence.replace(self.mask_token, gf, 1)
					# we know there is a unique sentence since that is how we are looping
					odds_ratios_summary.loc[odds_ratios_summary.sentence == sentence, 'sentence'] = current_sentence
					sentence 			= current_sentence
		else:
			odds_ratios_summary['role_position'] = [tokens_to_roles[token] for token in odds_ratios_summary.token]
		
		# format the strings with tokens for display purposes before returning
		for col in ['ratio_name', 'token', 'arg_type']:
			odds_ratios_summary[col] = self._format_strings_with_tokens_for_display(odds_ratios_summary[col], added_args).tolist()
		
		# add information about the evaluation parameters
		odds_ratios_summary = odds_ratios_summary.assign(
			eval_data 		= eval_cfg.data.name.split('.')[0],					
			epoch_criteria 	= eval_cfg.epoch if isinstance(eval_cfg.epoch, str) else 'manual',
			eval_epoch 		= epoch,
			total_epochs 	= total_epochs
		)
		
		if 'eval_args' in eval_cfg.data:
			odds_ratios_summary = odds_ratios_summary.assign(
				eval_args 	= eval_cfg.data.eval_args	
			)
		
		# add Tuner hyperparameters to the data
		odds_ratios_summary = self._add_hyperparameters_to_summary_df(odds_ratios_summary)
		
		# for now, we are not using these columns, so we're dropping them before returning. we can easily change this later if desired
		odds_ratios_summary = odds_ratios_summary.drop(
			[
				'logit', 'probability', # 'log_probability', 'other_log_probability',
				'surprisal', 'predicted_sentence', 'predicted_ids'
			], axis=1
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
		dataset 			= self._load_format_dataset(dataset_loc=dataset_loc, split='test', n_examples=eval_cfg.comparison_n_exs)
		
		if self.model.training:
			log.warning('Model performance will not be compared to baseline in training mode. Model will be set to eval mode.')
			self.model.eval()
		
		KL_baseline_loss 	= kl_baseline_loss.KLBaselineLoss(
								model 				= self.model,
								tokenizer 			= self.tokenizer,
								dataset 			= dataset,
								batch_size 			= eval_cfg.comparison_batch_size,
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
								self._format_strings_with_tokens_for_display(tuner_utils.flatten([self.args[arg_type] for arg_type in self.args]))
			
			args = deepcopy(self.args)
			args = {k: self._format_strings_with_tokens_for_display(v) for k, v in args.items()}
			
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

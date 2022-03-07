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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import torch.nn as nn

from tqdm import trange, tqdm
from math import ceil, floor, sqrt
from typing import Dict, List, Tuple, Union, Type
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import logging as lg
from transformers import BertForMaskedLM, BertTokenizer
from transformers import RobertaForMaskedLM, RobertaTokenizer
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from sklearn.manifold import TSNE

from core.tuner_utils import *

lg.set_verbosity_error()

log = logging.getLogger(__name__)

GF_ORDER = ['[subj]', '[obj]', '[2obj]', '[iobj]', '[obl]', '[pobj]', '[adj]']

class Tuner:

	# START Computed Properties
	
	@property
	def exp_type(self) -> str:
		return self.cfg.tuning.exp_type
	
	@property
	def model_class(self) -> Type['PreTrainedModel']:
		return eval(self.cfg.model.base_class) if isinstance(eval(self.cfg.model.base_class), type) else None
	
	@property
	def tokenizer_class(self) -> Type['PreTrainedTokenizer']:
		return eval(self.cfg.model.tokenizer) if isinstance(eval(self.cfg.model.tokenizer), type) else None
	
	@property
	def model_bert_name(self) -> str:
		return self.cfg.model.base_class.lower().replace('formaskedlm', '') if self.cfg.model.base_class != 'multi' else None
	
	@property
	def mask_tok(self) -> str:
		return self.tokenizer.mask_token
	
	@property
	def mask_tok_id(self) -> int:
		return self.tokenizer.get_vocab()[self.mask_tok]
	
	@property
	def string_id(self) -> str:
		return self.cfg.model.string_id
	
	@property
	def reference_sentence_type(self) -> str:
		return self.cfg.tuning.reference_sentence_type
	
	@property
	def dev_reference_sentence_type(self) -> str:
		return self.cfg.dev.reference_sentence_type
	
	@property
	def masked_tuning_style(self) -> str:
		return self.cfg.hyperparameters.masked_tuning_style.lower()
	
	@property
	def masked(self) -> bool:
		return self.masked_tuning_style != 'none'
	
	@property
	def tuning_data(self) -> List[str]:
		data = [strip_punct(s) for s in self.cfg.tuning.data] if self.cfg.hyperparameters.strip_punct else list(self.cfg.tuning.data)
		data = [d.lower() for d in data] if 'uncased' in self.string_id else data
		# warning related to roberta: it treats tokens with preceding spaces as different from tokens without
		# this means that if we use a token at the beginning of a sentence and in the middle, it won't be typical
		# here we check for this, and warn the user to avoid this situation
		if self.model_bert_name == 'roberta':
			for token in self.tokens_to_mask:
				at_beginning = any([bool(re.search('^' + token, d)) for d in data])
				in_middle = any([bool(re.search(' ' + token, d)) for d in data])
				if at_beginning * in_middle > 0:
					log.warning('RoBERTa treats tokens with preceding spaces differently, but you have used the same token for both cases! This may complicate results.')
		
		return data
	
	@property
	def mixed_tuning_data(self) -> List[str]:
		to_mix = self.verb_tuning_data if self.exp_type == 'newverb' else self.tuning_data
		
		data = []
		for s in to_mix:
			if self.cfg.hyperparameters.strip_punct:
				s = strip_punct(s)
			
			for tok in self.tokens_to_mask:
				r = np.random.random()
				# Bert tuning regimen
				# Masked tokens are masked 80% of the time, 
				# original 10% of the time,
				# and random word 10% of the time
				if r < 0.8:
					s = s.replace(tok, self.mask_tok)
				elif 0.8 <= r < 0.9:
					pass
				elif 0.9 <= r:
					while True:
						# we do this to ensure that the random word is tokenized as one word so that it doesn't throw off the lengths and halt tuning
						random_word = np.random.choice(list(self.tokenizer.get_vocab().keys()))
						random_word = random_word.replace(chr(288), '')
						# if the sentence doesn't begin with our target to replace, 
						# we need to add a space before it since that can throw off tokenization for some models
						# then we run the check, and remove the space for replacement into the string
						if not s.lower().startswith(tok.lower()):
							random_word = ' ' + random_word
					
						if len(self.tokenizer.tokenize(random_word)) == 1:
							random_word = random_word.strip()
							break			
					
					s = s.replace(tok, random_word)
			
			data.append(s)
		
		return data
	
	@property
	def masked_tuning_data(self) -> List[str]:
		to_mask = self.verb_tuning_data if self.exp_type == 'newverb' else self.tuning_data
		
		data = []
		for s in to_mask:
			if self.cfg.hyperparameters.strip_punct:
				s = strip_punct(s)
			for tok in self.tokens_to_mask:
				s = s.replace(tok, self.mask_tok)
			
			data.append(s)
		
		return data
	
	@property
	def verb_tuning_data(self) -> Dict[str,List[str]]:
		if not 'args' in self.cfg.tuning.keys():
			log.warning("You're trying to get new verb data for the wrong kind of experiment!")
			return self.tuning_data
		
		to_replace = self.cfg.tuning.args
		
		args, values = zip(*to_replace.items())
		replacement_combinations = itertools.product(*list(to_replace.values()))
		to_replace_dicts = [dict(zip(args, t)) for t in replacement_combinations]
		
		data = []
		for d in to_replace_dicts:
			for sentence in self.tuning_data:
				if self.cfg.hyperparameters.strip_punct:
					s = strip_punct(s)
				
				for arg, value in d.items():
					sentence = sentence.replace(arg, value)
				
				data.append(sentence)
		
		sentences = [d.lower() for d in data] if 'uncased' in self.string_id else data
		
		# Return the args as well as the sentences, 
		# since we need to save them in order to 
		# access them directly when evaluating
		# return {
		# 	'args' : to_replace,
		# 	'data' : sentences
		# }
		return sentences
	
	@property
	def masked_argument_data(self) -> Dict:
		if self.exp_type != 'newverb':
			log.warn(f"You're trying to get data for the wrong kind of experiment! {self.exp_type}")
			return {}
		
		# get the tuning data with the arguments replaced with mask tokens so we can get the current predictions about them
		sentences = self.tuning_data
		for i, s in enumerate(sentences):
			for arg_type in self.cfg.tuning.args:
				s = s.replace(arg_type, self.tokenizer.mask_token)
			
			sentences[i] = s
		
		inputs = self.tokenizer(sentences, return_tensors='pt', padding=True)
		
		# get the order of the arguments so that we know which mask token corresponds to which argument type
		args_in_order = [[word for word in strip_punct(sentence).split(' ') if word in self.cfg.tuning.args] for sentence in self.tuning_data]
		masked_token_indices = [[index for index, token_id in enumerate(i) if token_id == self.mask_tok_id] for i in inputs['input_ids']]
		sentence_arg_indices = [dict(zip(arg, index)) for arg, index in zip(args_in_order, masked_token_indices)]
		
		return {'sentences' : sentences, 'inputs' : inputs, 'sentence_arg_indices' : sentence_arg_indices}
	
	@property
	def tokens_to_mask(self) -> List[str]:
		# convert things to lowercase for uncased models
		tokens = [t.lower() for t in self.cfg.tuning.to_mask] if 'uncased' in self.string_id else list(self.cfg.tuning.to_mask)
		# add the versions of the tokens with preceding spaces to our targets for roberta
		if self.model_bert_name == 'roberta':
			tokens += [chr(288) + t for t in tokens]
		return tokens
	
	@property
	def dev_data(self) -> List[str]:
		dev_data = {}
		for dataset in self.cfg.dev:
			data = [strip_punct(s) for s in self.cfg.dev[dataset].data] if self.cfg.hyperparameters.strip_punct else list(self.cfg.dev[dataset].data)
			data = [d.lower() for d in data] if 'uncased' in self.string_id else data
			# warning related to roberta: it treats tokens with preceding spaces as different from tokens without
			# this means that if we use a token at the beginning of a sentence and in the middle, it won't be typical
			# here we check for this, and warn the user to avoid this situation
			if self.model_bert_name == 'roberta':
				for token in self.tokens_to_mask:
					at_beginning = any([bool(re.search('^' + token, d)) for d in data])
					in_middle = any([bool(re.search(' ' + token, d)) for d in data])
					if at_beginning * in_middle > 0:
						log.warning('RoBERTa treats tokens with preceding spaces differently, but you have used the same token for both cases! This may complicate results.')
			
			dev_data.update({dataset: data})
		
		return dev_data
	
	@property
	def mixed_dev_data(self) -> List[str]:
		if not self.cfg.dev:
			return {}
		
		to_mix = {dataset: self.verb_dev_data[dataset]['data'] if self.cfg.dev[dataset].exp_type == 'newverb' else self.dev_data[dataset] for dataset in self.cfg.dev}
		mixed_dev_data = {}
		for dataset in self.cfg.dev:
			data = []
			for s in to_mix[dataset]:
				if self.cfg.hyperparameters.strip_punct:
					s = strip_punct(s)
				
				for tok in self.tokens_to_mask:
					r = np.random.random()
					# Bert tuning regimen
					# Masked tokens are masked 80% of the time, 
					# original 10% of the time,
					# and random word 10% of the time
					if r < 0.8:
						s = s.replace(tok, self.mask_tok)
					elif 0.8 <= r < 0.9:
						pass
					elif 0.9 <= r:
						while True:
							# we do this to ensure that the random word is tokenized as one word so that it doesn't throw off the lengths and halt tuning
							random_word = np.random.choice(list(self.tokenizer.get_vocab().keys()))
							random_word = random_word.replace(chr(288), '')
							# if the sentence doesn't begin with our target to replace, 
							# we need to add a space before it since that can throw off tokenization for some models
							# then we run the check, and remove the space for replacement into the string
							if not s.lower().startswith(tok.lower()):
								random_word = ' ' + random_word
						
							if len(self.tokenizer.tokenize(random_word)) == 1:
								random_word = random_word.strip()
								break			
						
						s = s.replace(tok, random_word)
				
				data.append(s)
			
			mixed_dev_data.update({dataset: data})
		
		return mixed_dev_data
	
	@property
	def masked_dev_data(self) -> Dict[str,List[str]]:
		if not self.cfg.dev:
			return {}
		
		to_mask = {dataset: self.verb_dev_data[dataset]['data'] if self.cfg.dev[dataset].exp_type == 'newverb' else self.dev_data[dataset] for dataset in self.cfg.dev}
		
		masked_dev_data = {}
		for dataset in self.cfg.dev:
			data = []
			for s in to_mask[dataset]:
				if self.cfg.hyperparameters.strip_punct:
					s = strip_punct(s)
				for tok in self.tokens_to_mask:
					s = s.replace(tok, self.mask_tok)
				
				data.append(s)
			
			masked_dev_data.update({dataset:data})
		
		return masked_dev_data
	
	@property
	def masked_dev_argument_data(self) -> Dict[str,List[str]]:
		if self.exp_type != 'newverb':
			log.warn(f"You're trying to get data for the wrong kind of experiment! {self.exp_type}")
			return {}
		
		if not self.cfg.dev:
			return {}
		
		dev_argument_data = {}
		for dataset in self.dev_data:
			sentences = self.dev_data[dataset]
			for i, s in enumerate(sentences):
				for arg_type in self.cfg.tuning.args:
					s = s.replace(arg_type, self.tokenizer.mask_token)			
				
				sentences[i] = s
			
			inputs = self.tokenizer(sentences, return_tensors='pt', padding=True)
			
			# get the order of the arguments so that we know which mask token corresponds to which argument type
			args_in_order = [[word for word in strip_punct(sentence).split(' ') if word in self.cfg.tuning.args] for sentence in self.dev_data[dataset]]
			masked_token_indices = [[index for index, token_id in enumerate(i) if token_id == self.mask_tok_id] for i in inputs['input_ids']]
			sentence_arg_indices = [dict(zip(arg, index)) for arg, index in zip(args_in_order, masked_token_indices)]
			
			dev_argument_data[dataset] = {'sentences' : sentences, 'inputs' : inputs, 'sentence_arg_indices' : sentence_arg_indices}
		
		return dev_argument_data
	
	@property
	def verb_dev_data(self) -> Dict[str,List[str]]:
		if not self.cfg.dev:
			return {}
		
		verb_dev_data = {}
		for dataset in self.cfg.dev:
			if not 'args' in self.cfg.dev[dataset].keys():
				log.warning("You're trying to get new verb data for the wrong kind of experiment!")
				verb_dev_data.update({dataset: self.dev_data[dataset]})
			else:
				to_replace = self.cfg.dev[dataset].args
				
				args, values = zip(*to_replace.items())
				replacement_combinations = itertools.product(*list(to_replace.values()))
				to_replace_dicts = [dict(zip(args, t)) for t in replacement_combinations]
				
				data = []
				for d in to_replace_dicts:
					for sentence in self.dev_data[dataset]:
						if self.cfg.hyperparameters.strip_punct:
							s = strip_punct(s)
						
						for arg, value in d.items():
							sentence = sentence.replace(arg, value)
						
						data.append(sentence)
				
				sentences = [d.lower() for d in data] if 'uncased' in self.string_id else data
				
				# Return the args as well as the sentences, 
				# since we need to save them in order to 
				# access them directly when evaluating
				verb_dev_data.update({'args' : to_replace, 'data' : sentences})
		
		return verb_dev_data
	
	# END Computed Properties
	
	def __init__(self, cfg: DictConfig) -> None:
		self.cfg = cfg
		
		if self.string_id != 'multi':
			log.info(f"Initializing Tokenizer:\t{self.cfg.model.tokenizer}")
			
			# we do this with the self.cfg.tuning.to_mask data so that 
			# the versions with preceding spaces can be automatically 
			# added to roberta correctly and returned from self.tokens_to_mask
			self.tokenizer = create_tokenizer_with_added_tokens(self.string_id, self.tokenizer_class, self.cfg.tuning.to_mask, **self.cfg.model.tokenizer_kwargs)
			
			log.info(f"Initializing Model:\t{self.cfg.model.base_class}")
			self.model = self.model_class.from_pretrained(self.string_id, **self.cfg.model.model_kwargs)
			self.model.resize_token_embeddings(len(self.tokenizer))
	
	def tune(self) -> None:
		"""
		Fine-tunes the model on the provided tuning data. Saves updated weights to disk.
		"""
		
		# function to return the weight updates so we can save them every epoch
		def get_updated_weights() -> Dict[str,torch.Tensor]:
			updated_weights = {}
			for token in self.tokens_to_mask:
				tok_id = self.tokenizer.get_vocab()[token]
				updated_weights[token] = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight[tok_id,:].clone()
			
			return updated_weights
		
		# verify that tokens_to_mask matches in test and dev sets (i.e., dev sets is a subset of the training sets)
		devs = self.cfg.dev.copy()
		for dataset in self.cfg.dev:
			if not all(token in self.cfg.tuning.to_mask for token in self.cfg.dev[dataset].to_mask):
				with open_dict(devs):
					log.warn(f'Not all dev tokens to mask from {dataset} are in the training set! This is probably not what you intended. Removing this dataset from the dev data.')
					del devs[dataset]
		
		with open_dict(self.cfg):
			self.cfg.dev = devs
		
		with torch.no_grad():
			# This reinitializes the token weights to random values to provide variability in model tuning
			model_embedding_weights = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight
			model_embedding_dim = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.embedding_dim
			num_new_tokens = len(self.tokens_to_mask)
			new_embeds = nn.Embedding(num_new_tokens, model_embedding_dim)
			
			std, mean = torch.std_mean(model_embedding_weights)
			log.info(f"Initializing new token(s) with random data drawn from N({mean:.2f}, {std:.2f})")
			
			# we do this here manually because otherwise running multiple models
			# using multirun was giving identical results
			seed = int(torch.randint(2**32-1, (1,))) if not 'seed' in self.cfg else self.cfg.seed
			set_seed(seed)
			log.info(f"Seed set to {seed}")
			
			nn.init.normal_(new_embeds.weight, mean=mean, std=std)
			for i, tok in enumerate(self.tokens_to_mask):
				tok_id = self.tokenizer.convert_tokens_to_ids(tok)
				getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight[tok_id] = new_embeds.weight[i]
		
		if not self.tuning_data:
			log.info(f'Saving randomly initialized weights')
			with gzip.open('weights.pkl.gz', 'wb') as f:
				pkl.dump({0: get_updated_weights(), 'random_seed': seed}, f)
			return
		
		# Collect Hyperparameters
		lr = self.cfg.hyperparameters.lr
		epochs = self.cfg.hyperparameters.max_epochs
		min_epochs = self.cfg.hyperparameters.min_epochs
		optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0)
		
		# Store the old embeddings so we can verify that only the new ones get updated
		self.old_embeddings = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.clone()
		
		# Freeze parameters
		log.info(f"Freezing model parameters")
		for name, param in self.model.named_parameters():
			if 'word_embeddings' not in name:
				param.requires_grad = False
			
			if param.requires_grad:
				assert 'word_embeddings' in name, f"{name} is not frozen!"
		
		# Determine which data to use based on the experiment
		# Do this once ahead of time if we are not changing it
		# but do it in the loop if we are using the roberta-style randomized tuning data per epoch
		# if self.exp_type == 'newverb':
		# 	args = self.verb_tuning_data['args']
		# 	with open('args.yaml', 'w') as outfile:
		# 		outfile.write(OmegaConf.to_yaml(args))
		
		if self.exp_type == 'newverb' and not self.masked:
			inputs_data = self.verb_tuning_data
			# dev_inputs_data = {dataset: self.verb_dev_data[dataset]['data'] for dataset in self.verb_dev_data}
		elif self.masked and self.masked_tuning_style == 'always':
			inputs_data = self.masked_tuning_data
			# dev_inputs_data = self.masked_dev_data
		elif self.masked and self.masked_tuning_style in ['bert', 'roberta']: # when using bert tuning or roberta tuning. For roberta tuning, this is done later on
			inputs_data = self.mixed_tuning_data
			# dev_inputs_data = self.mixed_dev_data
		elif not self.masked:
			inputs_data = self.tuning_data
			# dev_inputs_data = self.dev_data
		
		dev_inputs_data = self.masked_dev_data
		
		labels_data = self.verb_tuning_data if self.exp_type == 'newverb' else self.tuning_data
		dev_labels_data = {dataset: self.verb_dev_data[dataset]['data'] for dataset in self.verb_dev_data} if self.exp_type == 'newverb' else self.dev_data
		
		# make sure that adding the new tokens doesn't mess up the tokenization of the rest of the sequence
		if not (verify_tokenization_of_sentences(self.tokenizer, inputs_data, self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs) and \
			    verify_tokenization_of_sentences(self.tokenizer, labels_data, self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs)):
			log.error('The new tokens added affected the tokenization of other elements in the inputs! Try using different strings.')
			return
		
		if self.exp_type == 'newverb' and not verify_tokenization_of_sentences(self.tokenizer, self.masked_argument_data['sentences'], self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs):
			log.error('The new tokens added affected the tokenization of other elements in the masked argument inputs! Try using different strings.')
			return
		
		for dataset in dev_inputs_data:
			if not (verify_tokenization_of_sentences(self.tokenizer, dev_inputs_data[dataset], self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs) and \
				    verify_tokenization_of_sentences(self.tokenizer, dev_labels_data[dataset], self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs)):
				log.error(f'The new tokens added affected the tokenization of other elements in the dev inputs for dataset {dataset}! Try using different strings.')
				return
		
		for dataset in self.masked_dev_argument_data:
			if not verify_tokenization_of_sentences(self.tokenizer, self.masked_dev_argument_data[dataset]['sentences'], self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs):
				log.error(f'The new tokens added affected the tokenization of other elements in the masked dev argument inputs for dataset {dataset}! Try using different strings.')
				return
		
		inputs = self.tokenizer(inputs_data, return_tensors="pt", padding=True)
		labels = self.tokenizer(labels_data, return_tensors="pt", padding=True)["input_ids"]
		
		dev_inputs = {dataset: self.tokenizer(dev_inputs_data[dataset], return_tensors='pt', padding=True) for dataset in dev_inputs_data}
		dev_labels = {dataset: self.tokenizer(dev_labels_data[dataset], return_tensors='pt', padding=True)['input_ids'] for dataset in dev_labels_data}
		
		# used to calculate metrics during training
		masked_inputs = self.tokenizer(self.masked_tuning_data, return_tensors="pt", padding=True)
		masked_dev_inputs = {dataset: self.tokenizer(self.masked_dev_data[dataset], return_tensors='pt', padding=True) for dataset in self.masked_dev_data}
		
		log.info(f"Training model @ '{os.getcwd().replace(hydra.utils.get_original_cwd(), '')}' (min_epochs={min_epochs}, max_epochs={epochs}, patience={self.cfg.hyperparameters.patience}, \u0394={self.cfg.hyperparameters.delta})")
		
		# Store weights pre-training so we can inspect the initial status later
		saved_weights = {}
		saved_weights[0] = get_updated_weights()
		
		datasets = [self.cfg.tuning.name + ' (train)', self.cfg.tuning.name + ' (masked, no dropout)'] + [dataset + ' (dev)' for dataset in self.cfg.dev]
		
		metrics = pd.DataFrame(data = {
			'epoch'   : list(range(1,epochs+1)) * len(datasets),
			'dataset' : np.repeat(datasets, [epochs] * len(datasets))
		})
		metrics['loss'] = np.nan
		
		writer = SummaryWriter()
		
		with trange(epochs) as t:
			patience_counter = 0
			patience_counters = {d.replace('_', ' ') : self.cfg.hyperparameters.patience for d in datasets}
			best_mean_loss = np.inf
			best_losses = {d.replace('_', ' ') : np.inf for d in datasets}
			for epoch in t:
				self.model.train()
				optimizer.zero_grad(set_to_none=True) # this is supposed to be faster than .zero_grad()
				
				# If we are using roberta-style masking, get new randomly changed inputs each epoch
				if self.masked_tuning_style == 'roberta':
					inputs_data = self.mixed_tuning_data
					# dev_inputs_data = self.mixed_dev_data
					# we only need to do this for the inputs; the labels were checked before and remain the same
					count = 0
					while not verify_tokenization_of_sentences(self.tokenizer, inputs_data, self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs):
						count += 1
						log.warning('The new tokens added affected the tokenization of sentences generated using roberta-style tuning!')
						log.warning(f'Affected: {inputs_data}')
						log.warning('Rerolling to try again.')
						inputs_data = self.mixed_tuning_data
						# don't do this too many times if it consistently fails; just break
						if count > 10:
							log.error('Unable to find roberta-style masked tuning data that was tokenized correctly after 10 tries. Exiting.')
							return
					
					# for dataset in dev_inputs_data:
					# 	count = 0
					# 	while not verify_tokenization_of_sentences(self.tokenizer, dev_inputs_data[dataset], self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs):
					# 		count += 1
					# 		log.warning('The new tokens added affected the tokenization of dev sentences generated using roberta-style tuning!')
					# 		log.warning(f'Affected: {dataset}, {dev_inputs_data}')
					# 		log.warning('Rerolling to try again.')
					# 		dev_inputs_data[dataset] = self.mixed_dev_data[dataset]
					# 		if count > 10:
					# 			log.error(f'Unable to find roberta-style masked dev data for {dataset} that was tokenized correctly after 10 tries. Exiting.')
					# 			return
					
					inputs = self.tokenizer(inputs_data, return_tensors="pt", padding=True)
					# dev_inputs = {dataset: self.tokenizer(dev_inputs_data[dataset], return_tensors='pt', padding=True) for dataset in dev_inputs_data}
				
				# Compute loss
				train_outputs = self.model(**inputs, labels=labels)
				train_loss = train_outputs.loss
				train_loss.backward()
				
				# Log result
				metrics.loc[(metrics.epoch == epoch + 1) & (metrics.dataset == self.cfg.tuning.name + ' (train)'), 'loss'] = train_loss.item()
				tb_loss_dict = {f'{self.cfg.tuning.name.replace("_", " ") + " (train)"}': train_loss}
				if train_loss.item() < best_losses[self.cfg.tuning.name.replace('_', ' ') + ' (train)'] - self.cfg.hyperparameters.delta:
					best_losses[self.cfg.tuning.name.replace('_', ' ') + ' (train)'] = train_loss.item()
					patience_counters[self.cfg.tuning.name.replace('_', ' ') + ' (train)'] = self.cfg.hyperparameters.patience
				else:
					patience_counters[self.cfg.tuning.name.replace('_', ' ') + ' (train)'] -= 1
					patience_counters[self.cfg.tuning.name.replace('_', ' ') + ' (train)'] = max(patience_counters[self.cfg.tuning.name.replace('_', ' ') + ' (train)'], 0)
				
				metrics.loc[(metrics.epoch == epoch + 1) & (metrics.dataset == self.cfg.tuning.name + ' (train)'), 'remaining patience'] = patience_counters[self.cfg.tuning.name.replace('_', ' ') + ' (train)']
				
				train_results = self.collect_results(masked_inputs, labels, self.tokens_to_mask, train_outputs)
				
				# get metrics for plotting
				epoch_metrics = self.get_epoch_metrics(train_results)
				
				tb_metrics_dict = {}
				for metric in epoch_metrics:
					tb_metrics_dict[metric] = {}
					for token in epoch_metrics[metric]:
						tb_metrics_dict[metric][token] = {}
						metrics.loc[(metrics.epoch == epoch + 1) & (metrics.dataset == self.cfg.tuning.name + ' (train)'), f'{token} mean {metric} in expected position'] = epoch_metrics[metric][token]
						tb_metrics_dict[metric][token].update({f'{self.cfg.tuning.name.replace("_", " ") + " (train)"}': epoch_metrics[metric][token]})
				
				# store weights of the relevant tokens so we can save them
				saved_weights[epoch + 1] = get_updated_weights()
				
				# GRADIENT ADJUSTMENT
				# 
				# The word embeddings remain unfrozen, but we only want to update
				# the embeddings of the novel tokens. To do this, we zero-out
				# all gradients except for those at these token indices.
				nz_grad = {}
				for token in self.tokens_to_mask:
					token_id = self.tokenizer.get_vocab()[token]
					nz_grad[token_id] = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.grad[token_id].clone()
				
				# Zero out all gradients of word_embeddings in-place
				getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.grad.data.fill_(0) # note that fill_(None) doesn't work here
				
				# Replace the original gradients at the relevant token indices
				for token_to_mask in nz_grad:
					getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.grad[token_to_mask] = nz_grad[token_to_mask]
				
				optimizer.step()
				
				# Check that we changed the correct number of parameters
				new_embeddings = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.clone()
				num_changed_params = torch.round(torch.sum(torch.mean(torch.ne(self.old_embeddings, new_embeddings) * 1., dim = -1))) # use torch.round to attempt to fix rare floating point rounding error
				num_expected_to_change = len(self.tokens_to_mask)
				assert num_changed_params == num_expected_to_change, f"Exactly {num_expected_to_change} embeddings should have been updated, but {num_changed_params} were!"
				
				# evaluate the model on the dev set(s)
				self.model.eval()
				with torch.no_grad():
					dev_losses = []
					for dataset in dev_inputs:
						dev_outputs = self.model(**dev_inputs[dataset], labels=dev_labels[dataset])
						dev_loss = dev_outputs.loss
						dev_losses += [dev_loss.item()]
						
						metrics.loc[(metrics.epoch == epoch + 1) & (metrics.dataset == self.cfg.dev[dataset].name + ' (dev)'),'loss'] = dev_loss.item()
						tb_loss_dict.update({f'{self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"}': dev_loss})
						
						if dev_loss.item() < best_losses[self.cfg.dev[dataset].name.replace('_', ' ') + ' (dev)'] - self.cfg.hyperparameters.delta:
							best_losses[self.cfg.dev[dataset].name.replace('_', ' ') + ' (dev)'] = dev_loss.item()
							patience_counters[self.cfg.dev[dataset].name.replace('_', ' ') + ' (dev)'] = self.cfg.hyperparameters.patience
						else:
							patience_counters[self.cfg.dev[dataset].name.replace('_', ' ') + ' (dev)'] -= 1
							patience_counters[self.cfg.dev[dataset].name.replace('_', ' ') + ' (dev)'] = max(patience_counters[self.cfg.dev[dataset].name.replace('_', ' ') + ' (dev)'], 0)
							
						metrics.loc[(metrics.epoch == epoch + 1) & (metrics.dataset == self.cfg.dev[dataset].name + ' (dev)'), 'remaining patience'] = patience_counters[self.cfg.dev[dataset].name.replace('_', ' ') + ' (dev)']
						
						dev_results = self.collect_results(masked_dev_inputs[dataset], dev_labels[dataset], self.tokens_to_mask, dev_outputs)
						
						dev_epoch_metrics = self.get_epoch_metrics(dev_results)
						
						if self.exp_type == 'newverb':
							masked_dev_argument_inputs = self.masked_dev_argument_data[dateset]['inputs']
							newverb_dev_outputs = self.model(**self.masked_dev_argument_data[dataset]['inputs'])
							newverb_dev_results = self.collect_newverb_results(newverb_dev_outputs)
							newverb_dev_epoch_metrics = self.get_newverb_epoch_metrics(newverb_dev_results)
							dev_epoch_metrics = {metric: {**dev_epoch_metrics[metric], **newverb_dev_epoch_metrics[metric]} for metric in list(dict.fromkeys([*list(dev_epoch_metrics.keys()), *list(dev_newverb_epoch_metrics.keys())]))}
						
						for metric in dev_epoch_metrics:
							for token in dev_epoch_metrics[metric]:
								metrics.loc[(metrics['epoch'] == epoch + 1) & (metrics.dataset == self.cfg.dev[dataset].name + ' (dev)'), f'{token} mean {metric} in expected position'] = dev_epoch_metrics[metric][token]
								tb_metrics_dict[metric][token].update({f'{self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"}': dev_epoch_metrics[metric][token]})
					
					# Compute loss on masked training data without dropout, 
					# as this is most representative of the testing procedure
					# so we can use it to determine the best epoch
					no_dropout_train_outputs = self.model(**masked_inputs, labels=labels)
					no_dropout_train_loss = no_dropout_train_outputs.loss
					
					dev_losses += [no_dropout_train_loss.item()]
					
					# Log result
					metrics.loc[(metrics.epoch == epoch + 1) & (metrics.dataset == self.cfg.tuning.name + ' (masked, no dropout)'), 'loss'] = no_dropout_train_loss.item()
					
					if no_dropout_train_loss.item() < best_losses[self.cfg.tuning.name.replace('_', ' ') + ' (masked, no dropout)'] - self.cfg.hyperparameters.delta:
						best_losses[self.cfg.tuning.name.replace('_', ' ') + ' (masked, no dropout)'] = no_dropout_train_loss.item()
						patience_counters[self.cfg.tuning.name.replace('_', ' ') + ' (masked, no dropout)'] = self.cfg.hyperparameters.patience
					else:
						patience_counters[self.cfg.tuning.name.replace('_', ' ') + ' (masked, no dropout)'] -= 1
						patience_counters[self.cfg.tuning.name.replace('_', ' ') + ' (masked, no dropout)'] = max(patience_counters[self.cfg.tuning.name.replace('_', ' ') + ' (masked, no dropout)'], 0)
						
					metrics.loc[(metrics.epoch == epoch + 1) & (metrics.dataset == self.cfg.tuning.name + ' (masked, no dropout)'), 'remaining patience'] = patience_counters[self.cfg.tuning.name.replace('_', ' ') + ' (masked, no dropout)']
					
					tb_loss_dict.update({f'{self.cfg.tuning.name.replace("_", " ") + " (masked, no dropout)"}': no_dropout_train_loss})
					
					no_dropout_train_results = self.collect_results(masked_inputs, labels, self.tokens_to_mask, no_dropout_train_outputs)
					
					# get metrics for plotting
					no_dropout_epoch_metrics = self.get_epoch_metrics(no_dropout_train_results)
					
					if self.exp_type == 'newverb':
						# we only do this without dropout for the masked argument data, since we don't want to update the weights based on these results
						no_dropout_newverb_outputs = self.model(**self.masked_argument_data['inputs'])
						no_dropout_newverb_results = self.collect_newverb_results(no_dropout_newverb_outputs)
						no_dropout_newverb_epoch_metrics = self.get_newverb_epoch_metrics(no_dropout_newverb_results)
						no_dropout_epoch_metrics = {metric: {**no_dropout_epoch_metrics[metric], **no_dropout_newverb_epoch_metrics[metric]} for metric in list(dict.fromkeys([*list(no_dropout_epoch_metrics.keys()), *list(no_dropout_newverb_epoch_metrics.keys())]))}
					
					for metric in no_dropout_epoch_metrics:
						for token in no_dropout_epoch_metrics[metric]:
							tb_metrics_dict[metric][token] = {}
							metrics.loc[(metrics.epoch == epoch + 1) & (metrics.dataset == self.cfg.tuning.name + ' (masked, no dropout)'), f'{token} mean {metric} in expected position'] = no_dropout_epoch_metrics[metric][token]
							tb_metrics_dict[metric][token].update({f'{self.cfg.tuning.name.replace("_", " ") + " (masked, no dropout)"}': no_dropout_epoch_metrics[metric][token]})
				
				writer.add_scalars('loss', tb_loss_dict, epoch)
				
				# for dataset in tb_loss_dict:
				# 	we do these replacements because tensorboard doesn't like to aggregate tags containing parentheses
				#	writer.add_scalar(f'loss/{dataset.replace("(", "<").replace(")", ">")}', tb_loss_dict[dataset], epoch)
				
				writer.add_scalar('loss/mean dev', np.mean(dev_losses), epoch)
				writer.add_scalar('loss/mean dev lower ci', np.mean(dev_losses) - np.std(dev_losses), epoch)
				writer.add_scalar('loss/mean dev upper ci', np.mean(dev_losses) + np.std(dev_losses), epoch)
				
				for metric in tb_metrics_dict:
					for token in tb_metrics_dict[metric]:
						writer.add_scalars(f'{token} mean {metric} in expected position', tb_metrics_dict[metric][token], epoch)
						
						# for dataset in tb_metrics_dict[metric][token]:
						#	writer.add_scalar(f'{token} mean {metric} in expected position/{dataset.replace("(", "<").replace(")", ">")}', tb_metrics_dict[metric][token][dataset], epoch)
						
						dev_only_token_metric = [tb_metrics_dict[metric][token][dataset] for dataset in tb_metrics_dict[metric][token] if not dataset.endswith('(train)')]
						writer.add_scalar(f'{token} mean {metric} in expected position/mean dev', np.mean(dev_only_token_metric), epoch)
						writer.add_scalar(f'{token} mean {metric} in expected position/mean dev lower ci', np.mean(dev_only_token_metric) - np.std(dev_only_token_metric), epoch)
						writer.add_scalar(f'{token} mean {metric} in expected position/mean dev upper ci', np.mean(dev_only_token_metric) + np.std(dev_only_token_metric), epoch)
				
				if np.mean(dev_losses) < best_mean_loss - self.cfg.hyperparameters.delta:
					best_mean_loss = np.mean(dev_losses)
					patience_counter = 0
				else:
					patience_counter += 1
					patience_counter = min(self.cfg.hyperparameters.patience, patience_counter)
					if patience_counter >= self.cfg.hyperparameters.patience and epoch + 1 >= min_epochs:
							metrics.loc[(metrics.epoch == epoch + 1), 'remaining patience overall'] = self.cfg.hyperparameters.patience - patience_counter
							writer.add_scalars('remaining patience', {**patience_counters, 'overall': self.cfg.hyperparameters.patience-patience_counter}, epoch)
							# for dataset in patience_counters:
							# 	writer.add_scalar(f'remaining patience/{dataset.replace("(", "<").replace(")", ">")}', patience_counters[dataset], epoch)
							# writer.add_scalar('remaining patience/overall', self.cfg.hyperparameters.patience-patience_counter, epoch)
							t.set_postfix(pat=self.cfg.hyperparameters.patience-patience_counter, avg_dev_loss='{0:5.2f}'.format(np.mean(dev_losses)), train_loss='{0:5.2f}'.format(train_loss.item()))
							break
				
				writer.add_scalars('remaining patience', {**patience_counters, 'overall': self.cfg.hyperparameters.patience-patience_counter}, epoch)
				# for dataset in patience_counters:
				#	writer.add_scalar(f'remaining patience/{dataset.replace("(", "<").replace(")", ">")}', patience_counters[dataset], epoch)
				#	writer.add_scalar('remaining patience/overall', self.cfg.hyperparameters.patience-patience_counter, epoch)
				
				metrics.loc[(metrics.epoch == epoch + 1), 'remaining patience overall'] = self.cfg.hyperparameters.patience - patience_counter
				
				# if self.cfg.dev:
				t.set_postfix(pat=self.cfg.hyperparameters.patience-patience_counter, avg_dev_loss='{0:5.2f}'.format(np.mean(dev_losses)), train_loss='{0:5.2f}'.format(train_loss.item()))
				# else:
				# 	t.set_postfix(train_loss='{0:5.2f}'.format(train_loss.item()))
		
		# note that we do not plot means in the pdfs if using only the no dropout training set as a dev set
		# but we DO include them in the tensorboard plots. this is because that allows us to include the 
		# hyperparameters info in the tensorboard log in SOME way without requiring us to create a directory
		# name that contains all of it (which results in names that are too long for the filesystem)
		model_label = f'{self.model_bert_name} {self.cfg.tuning.name.replace("_", " ")}, '
		model_label += f'masking: {self.cfg.hyperparameters.masked_tuning_style}, ' if self.masked else 'unmasked, '
		model_label += f'{"no punctuation" if self.cfg.hyperparameters.strip_punct else "punctuation"}, '
		model_label += f'epochs={epoch+1} (min={min_epochs}, max={epochs}), '
		model_label += f'pat={self.cfg.hyperparameters.patience} (\u0394={self.cfg.hyperparameters.delta})'
		
		# Aggregate the plots and add a helpful label
		# note that tensorboard does not support plotting means and CIs automatically even when aggregating
		# Thus, we only do this manually for the average dev loss, since plots of other means are in the PDF
		# and are less likely to be useful given the effort it would take to manually construct them
		# metrics_labels = {'loss' : ['Multiline', [f'loss_{dataset.replace("(", "-").replace(")", "-")}' for dataset in tb_loss_dict]]}
		metrics_labels = {'mean dev loss' : ['Margin', ['loss/mean dev', 'loss/mean dev lower ci', 'loss/mean dev upper ci']]}
		for metric in tb_metrics_dict:
			for token in tb_metrics_dict[metric]:
				# metrics_labels[f'{token} mean {metric} in expected position'] = ['Multiline', [f'{token} mean {metric} in expected position_{dataset.replace("(", "-").replace(")", "-")}' for dataset in tb_metrics_dict[metric][token]]]
				metrics_labels[f'mean dev {token} mean {metric} in expected position'] = ['Margin', [f'{token} mean {metric} in expected position/mean dev', f'{token} mean {metric} in expected position/mean dev lower ci', f'{token} mean {metric} in expected position/mean dev upper ci']]
		
		# metrics_labels['remaining patience'] = ['Multiline', [f'remaining patience_{dataset.replace("(", "-").replace(")", "-")}' for dataset in patience_counters] + ['remaining patience/overall']]
		
		layout = {model_label : metrics_labels}
		
		writer.add_custom_scalars(layout)
		
		# log this here so the progress bar doesn't get printed twice (which happens if we do the log in the loop)
		if patience_counter >= self.cfg.hyperparameters.patience:
			log.info(f'Mean dev loss has not improved by {self.cfg.hyperparameters.delta} in {patience_counter} epochs (min_epochs={min_epochs}). Halting training at epoch {epoch}.')
		
		# we do minus one here because we've also saved the randomly initialized weights @ 0
		log.info(f"Saving weights for random initializations and each of {len(saved_weights)-1} training epochs")
		with gzip.open('weights.pkl.gz', 'wb') as f:
			pkl.dump({**saved_weights, 'random_seed': seed}, f)
		
		metrics['dataset_type'] = ['dev' if dataset.endswith('(dev)') else 'train' for dataset in metrics.dataset]
		# metrics = metrics.dropna().reset_index(drop=True) this causes problems for the new verb experiments, since we have metrics for the dev sets that don't exist for the training set
		metrics = metrics[~metrics.loss.isnull()].reset_index(drop=True)
		
		metrics = metrics.assign(max_epochs=epochs, min_epochs=min_epochs)
		
		log.info(f'Plotting metrics')
		self.plot_metrics(metrics)
		
		metrics = pd.melt(
			metrics, 
			id_vars = ['epoch', 'dataset', 'dataset_type', 'min_epochs', 'max_epochs'], 
			value_vars = [c for c in metrics.columns if not c in ['epoch', 'dataset', 'dataset_type', 'min_epochs', 'max_epochs']], 
			var_name = 'metric'
		).assign(
			model_id = os.path.split(os.getcwd())[1],
			model_name = self.model_bert_name,
			tuning = self.cfg.tuning.name,
			masked = self.masked,
			masked_tuning_style = self.masked_tuning_style,
			strip_punct = self.cfg.hyperparameters.strip_punct,
			dataset = lambda df: [d.replace('_', ' ') for d in df.dataset],
			patience = self.cfg.hyperparameters.patience,
			delta = self.cfg.hyperparameters.delta
		)
		
		metrics.loc[metrics.metric == 'remaining patience overall', 'dataset'] = 'overall'
		metrics.loc[metrics.metric == 'remaining patience overall', 'dataset_type'] = 'overall'
		metrics = metrics.drop_duplicates().reset_index(drop=True)
		metrics['random_seed'] = seed
		
		log.info(f"Saving metrics")
		metrics.to_csv("metrics.csv.gz", index = False, na_rep = 'NaN')
		
		writer.flush()
		writer.close()
	
	def collect_results(self, masked_inputs: Dict[str,torch.Tensor], labels: torch.Tensor, eval_groups: Union[List[str],Dict[str,List[str]]], outputs: 'MaskedLMOutput') -> Dict:
		results = {}
		
		logits = outputs.logits
		probabilities = nn.functional.softmax(logits, dim=2)
		log_probabilities = nn.functional.log_softmax(logits, dim=2)
		surprisals = -(1/torch.log(torch.tensor(2.))) * nn.functional.log_softmax(logits, dim=2)
		predicted_ids = torch.argmax(log_probabilities, dim=2)
		
		for sentence_num, _ in enumerate(predicted_ids):
			sentence_results = {}
			
			# Foci = indices where input sentences have a [mask] token
			foci = torch.nonzero(masked_inputs["input_ids"][sentence_num] == self.mask_tok_id, as_tuple=True)[0]
			
			for focus in foci:
				focus_results = {}
				# this will need to be changed to deal with the ptb experiments
				# because in that case the eval_groups needs to be the values of the dict
				# rather than the keys
				for token in eval_groups:
					token_id = self.tokenizer.convert_tokens_to_ids(token)
					if self.exp_type in ['entail', 'newverb'] and labels[sentence_num,focus] == token_id:
						focus_results[token] = {}
						focus_results[token]['log probability'] = log_probabilities[sentence_num,focus,token_id].item()
						focus_results[token]['surprisal'] = surprisals[sentence_num,focus,token_id].item()
					elif not self.exp_type in ['entail', 'newverb']:
						focus_results[token] = {}
						logprob_means = []
						surprisal_means = []
						for word in eval_groups[token]:
							token_id = self.tokenizer.get_vocab()[word]
							logprob_means.append(log_probabilities[sentence_num,focus,token_id].item())
							surprisal_means.append(surprisals[sentence_num,focus,token_id].item())
						
						focus_results[token]['mean grouped log probability'] = np.mean(logprob_means)
						focus_results[token]['mean grouped surprisal'] = np.mean(surprisal_means)
				
				sentence_results[focus.item()] = {
					'focus metrics' : focus_results,
					'log probabilities' : log_probabilities[sentence_num,focus],
					'probabilities' : probabilities[sentence_num,focus],
					'logits': logits[sentence_num,focus]
				}	
			
			results[sentence_num] = sentence_results
		
		return results
	
	def collect_newverb_results(self, outputs: 'MaskedLMOutput') -> Dict:
		results = []
		
		logits = outputs.logits
		probabilities = nn.functional.softmax(logits, dim=2)
		log_probabilities = nn.functional.log_softmax(logits, dim=2)
		surprisals = -(1/torch.log(torch.tensor(2.))) * nn.functional.log_softmax(logits, dim=2)
		predicted_ids = torch.argmax(log_probabilities, dim=2)
		
		sentence_arg_indices = self.masked_argument_data['sentence_arg_indices']
		
		newverb_metrics = tuple(zip(sentence_arg_indices, self.tuning_data, logits, probabilities, log_probabilities, surprisals, predicted_ids))
		
		for arg_type in self.cfg.tuning.args:
			for arg in self.cfg.tuning.args[arg_type]:
				predictions_token_arg_sentence = []
				for sentence_num, (arg_indices, sentence, logit, prob, logprob, surprisal, predicted_id) in enumerate(newverb_metrics):
					if self.model_bert_name == 'roberta' and not sentence.startswith(arg_type):
						arg_token_id = self.tokenizer.convert_tokens_to_ids(chr(288) + arg)
					else:
						arg_token_id = self.tokenizer.convert_tokens_to_ids(arg)
					
					if arg_token_id == self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token):
						raise ValueError(f'Argument "{arg}" was not tokenized correctly! Try using a different one instead.')
					
					for arg_position, arg_index in [(arg_position, arg_index) for arg_position, arg_index in arg_indices.items() if not arg_position == arg_type]:
						log_odds = logprob[arg_index,arg_token_id]
						exp_log_odds = logprob[arg_indices[arg_type],arg_token_id]
						odds_ratio = exp_log_odds - log_odds
						
						prediction_row = {
							'arg type' : arg_type,
							'odds ratio' : odds_ratio,
							'ratio name' : arg_type + '/' + arg_position,
							'token id' : arg_token_id,
							'token' : arg,
							'sentence' : sentence,
							'sentence num' : sentence_num,
							'logit' : logit[arg_indices[arg_type],arg_token_id],
							'probability' : prob[arg_indices[arg_type],arg_token_id],
							'log probability' : exp_log_odds,
							'surprisal' : surprisal[arg_indices[arg_type],arg_token_id],
							'predicted ids' : ' '.join([str(i.item()) for i in predicted_id]),
							'predicted sequence' : ' '.join(self.tokenizer.convert_ids_to_tokens(predicted_id))
						}
						
						predictions_token_arg_sentence.append(prediction_row)
					
					results.append(predictions_token_arg_sentence)
		
		results = [d for arg_type in results for d in arg_type]
		
		return results
	
	def get_epoch_metrics(self, results: Dict) -> Dict:
		# calculate the mean of the metrics across all the sentences in the results
		epoch_metrics = {}
		
		for sentence in results:
			for focus in results[sentence]:
				for token in results[sentence][focus]['focus metrics']:
					for metric in results[sentence][focus]['focus metrics'][token]:
						epoch_metrics[metric] = {} if not metric in epoch_metrics.keys() else epoch_metrics[metric]
						epoch_metrics[metric][token] = [] if not token in epoch_metrics[metric].keys() else epoch_metrics[metric][token]
						epoch_metrics[metric][token].append(results[sentence][focus]['focus metrics'][token][metric])
		
		for metric in epoch_metrics:
			for token in epoch_metrics[metric]:
				epoch_metrics[metric][token] = np.mean(epoch_metrics[metric][token])
		
		return epoch_metrics
	
	def get_newverb_epoch_metrics(self, results: Dict, metrics: List[str] = ['log probability', 'surprisal']) -> Dict:
		newverb_epoch_metrics = {
			metric : {
				arg_type : 
					float(torch.mean(
						torch.tensor(
							[r[metric] for r in results if r['arg type'] == arg_type]
						)
					))
				for arg_type in self.cfg.tuning.args
			} 
			for metric in metrics
		}
		
		for metric in newverb_epoch_metrics:
			newverb_epoch_metrics[metric].update({
				token + ' ' + f'({arg_type})' : 
					float(torch.mean(
						torch.tensor(
							[r[metric] for r in results if r['token'] == token]
						)
					))
				for arg_type in self.cfg.tuning.args for token in self.cfg.tuning.args[arg_type]
			})
		
		return newverb_epoch_metrics		
	
	def restore_weights(self, checkpoint_dir: str, epoch: Union[int,str] = 'best_mean') -> Tuple[int,int]:
		weights_path = os.path.join(checkpoint_dir, 'weights.pkl.gz')
		
		with gzip.open(weights_path, 'rb') as f:
			weights = pkl.load(f)
			
		total_epochs = max([e for e in list(weights.keys()) if not isinstance(e, str)])
		
		if epoch == None or epoch in ['max', 'total', 'highest', 'last', 'final']:
			epoch = total_epochs
		elif 'best' in str(epoch):
			metrics = pd.read_csv(os.path.join(checkpoint_dir, 'metrics.csv.gz'))
			loss_df = metrics[(metrics.metric == 'loss') & (~metrics.dataset.str.endswith(' (train)'))]
			epoch = get_best_epoch(loss_df, method = 'mean' if 'mean' in epoch else 'sumsq' if 'sumsq' in epoch else '')
		
		log.info(f'Restoring saved weights from epoch {epoch}/{total_epochs}')
		
		with torch.no_grad():
			for token in weights[epoch]:
				tok_id = self.tokenizer.convert_tokens_to_ids(token)
				getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight[tok_id] = weights[epoch][token]
		
		# return the epoch and total_epochs to help if we didn't specify it
		return epoch, total_epochs
	
	def plot_metrics(self, metrics: pd.DataFrame) -> None:
		# do this to avoid messing up the passed dataframe
		metrics = metrics.copy()
		
		def determine_int_axticks(series: pd.Series, target_num_ticks: int = 10) -> List[int]:
			lowest = series.min()
			highest = series.max()
			
			if (highest - lowest) == target_num_ticks or (highest - lowest) < target_num_ticks:
				return [i for i in range(lowest, highest + 1)]
			
			new_min = target_num_ticks - 1
			new_max = target_num_ticks + 1
			while not highest % target_num_ticks == 0:
				if highest % new_min == 1:
					target_num_ticks = new_min
					break
				else:
					new_min -= 1
					# if we get here, it metrics highest is a prime and there's no good solution,
					# so we'll just brute force something later
					if new_min == 1:
						break
				
				if highest % new_max == 0:
					target_num_ticks = new_max
					break
				elif not new_max >= highest/2:
					new_max += 1
			
			int_axticks = [int(i) for i in list(range(lowest - 1, highest + 1, int(ceil(highest/target_num_ticks))))]
			int_axticks = [i for i in int_axticks if i in range(lowest, highest)]
			
			if not int_axticks:
				int_axticks = list(set([i for i in series.values]))
			
			return int_axticks
		
		all_metrics = [
			m for m in metrics.columns if not m in ['epoch', 'max_epochs', 'min_epochs', 'dataset', 'dataset_type', 'remaining patience overall'] and 
			not any([arg in m for arg_type in self.cfg.tuning.args for arg in self.cfg.tuning.args[arg_type]])
		]
		
		xticks = determine_int_axticks(metrics.epoch)
		
		with PdfPages('metrics.pdf') as pdf:
			for metric in all_metrics:
				# Get the other metrics which are like this one but for different tokens so that we can
				# set the axis limits to a common value. This is so we can compare the metrics visually
				# for each token more easily
				like_metrics = []
				for m in [m for m in metrics.columns if not m in ['epoch', 'max_epochs', 'min_epochs', 'dataset', 'dataset_type', 'remaining patience overall']]:
					if not m == metric:
						m1 = metric
						m2 = m
						for token in self.tokens_to_mask:
							m1 = m1.replace(token.upper(), '').replace(token.lower(), '') # do this to deal with both cased and uncased models
							m2 = m2.replace(token.upper(), '').replace(token.lower(), '') # do this to deal with both cased and uncased models
						
						if self.exp_type == 'newverb':
							for arg_type in self.cfg.tuning.args:
								m1 = m1.replace(arg_type, '')
								m2 = m2.replace(arg_type, '')
								m1 = m1.replace(' ()', '')
								m2 = m2.replace(' ()', '')
								for arg in self.cfg.tuning.args[arg_type]:
									m1 = m1.replace(arg, '')
									m2 = m2.replace(arg, '')
						
						if m1 == m2:
							like_metrics.append(m)
				
				ulim = np.max([*metrics[metric].dropna().values])
				llim = np.min([*metrics[metric].dropna().values])
				
				for m in like_metrics:
					ulim = np.max([ulim, *metrics[m].dropna().values])
					llim = np.min([llim, *metrics[m].dropna().values])
				
				adj = max(np.abs(ulim - llim)/40, 0.05)
				
				fig, ax = plt.subplots(1)
				fig.set_size_inches(9, 7)
				ax.set_ylim(llim - adj, ulim + adj)
				metrics.dataset = [dataset.replace('_', ' ') for dataset in metrics.dataset] # for legend titles
				
				# do this manually so we don't recycle colors
				num_datasets = len(metrics.dataset.unique())+1 # add one for mean
				palette = sns.color_palette(n_colors=num_datasets) if num_datasets <= 10 else sns.color_palette('hls', num_datasets) # if we have more than 10 dev sets, don't repeat colors
				sns.set_palette(palette)
				
				if len(metrics[metric].index) > 1:
					if metric == 'remaining patience':
						if len(metrics[~metrics.dataset.str.endswith('(train)')].dataset.unique()) > 1:
							global_patience = metrics[['epoch', 'remaining patience overall']].drop_duplicates().reset_index(drop=True).rename({'remaining patience overall' : 'remaining patience'}, axis = 1).assign(dataset = 'overall', dataset_type = 'global')
							sns.lineplot(data = pd.concat([metrics, global_patience], ignore_index=True), x = 'epoch', y = metric, ax=ax, hue='dataset', style='dataset_type', legend ='full')
							plt.yticks(determine_int_axticks(pd.concat([pd.concat([metrics['remaining patience'], metrics['remaining patience overall']], ignore_index=True).astype(int), pd.Series(0)], ignore_index=True)))
							handles, labels = ax.get_legend_handles_labels()
						else:
							sns.lineplot(data = metrics[['epoch', metric, 'dataset', 'dataset_type']].dropna(), x = 'epoch', y = metric, ax=ax, hue='dataset', style='dataset_type', legend='full')
							plt.yticks(determine_int_axticks(pd.concat([metrics['remaining patience'].astype(int), pd.Series(0)], ignore_index=True)))
							handles, labels = ax.get_legend_handles_labels()
					elif self.exp_type == 'newverb' and any([re.search(arg_type, metric) for arg_type in self.cfg.tuning.args]):
						# this occurs when we're doing a newverb exp and we want to plot the individual tokens in addition to the mean
						like_metrics = [m for m in like_metrics if re.sub(r'\[(.*)\].*', '[\\1]', metric) in m]
						token_metrics = metrics[['epoch'] + like_metrics + ['dataset', 'dataset_type']]
						token_metrics = token_metrics.melt(id_vars=['epoch', 'dataset', 'dataset_type'])
						token_metrics['token'] = [re.sub(r'^([^\s]+).*', '\\1', variable) for variable in token_metrics.variable]
						
						v_adjust = (ax.get_ylim()[1] - ax.get_ylim()[0])/100
						
						sns.lineplot(data = metrics[['epoch', metric, 'dataset', 'dataset_type']].dropna(), x = 'epoch', y = metric, ax = ax, hue='dataset', style='dataset_type', legend='full')
						if not token_metrics.empty:
							# for dataset, dataset_metrics in metrics.groupby('dataset'):
							# 	dataset_metrics = dataset_metrics.dropna()
							# 	if not dataset_metrics.empty:
							# 		ax.text(floor(max(dataset_metrics.epoch)*.8), dataset_metrics[dataset_metrics.epoch == floor(max(dataset_token_data.epoch)*.8)][metric]-v_adjust, dataset + ' mean', size=8, horizontalalignment='center', verticalalignment='top', color='black', zorder=15)
							
							for t, token_data in token_metrics.groupby('token'):
								token_data = token_data.dropna().reset_index(drop=True)
								sns.lineplot(data = token_data.dropna(), x='epoch', y='value', ax=ax, hue='dataset', style='dataset_type', linewidth=0.5, legend=False, alpha=0.3)
								for dataset, dataset_token_data in token_data.groupby('dataset'):
									ax.text(floor(max(dataset_token_data.epoch)*.8), dataset_token_data[dataset_token_data.epoch == floor(max(dataset_token_data.epoch)*.8)].value-v_adjust, t, size=6, horizontalalignment='center', verticalalignment='top', color='black', zorder=15, alpha=0.3)
						
						if len(metrics[~metrics.dataset.str.endswith('(train)')].dataset.unique()) > 1:
							sns.lineplot(data = metrics[(~metrics.dataset.str.endswith('(train)')) & (metrics.dataset != 'overall')], x = 'epoch', y = metric, ax = ax, color = palette[-1], ci = 68)
							handles, labels = ax.get_legend_handles_labels()
							handles += [ax.lines[-1]]
							labels += ['mean dev']
							ax.legend(handles=handles,labels=labels)
							ax.lines[-1].set_linestyle(':')
					else:
						sns.lineplot(data = metrics[['epoch', metric, 'dataset', 'dataset_type']].dropna(), x = 'epoch', y = metric, ax = ax, hue='dataset', style='dataset_type', legend='full')
						if len(metrics[~metrics.dataset.str.endswith('(train)')].dataset.unique()) > 1:
							sns.lineplot(data = metrics[(~metrics.dataset.str.endswith('(train)')) & (metrics.dataset != 'overall')], x = 'epoch', y = metric, ax = ax, color = palette[-1], ci = 68)
							handles, labels = ax.get_legend_handles_labels()
							handles += [ax.lines[-1]]
							labels += ['mean dev']
							ax.legend(handles=handles,labels=labels)
							ax.lines[-1].set_linestyle(':')
					
					# remove redundant information from the legend
					handles = ax.get_legend().legendHandles
					labels = [text.get_text() for text in ax.get_legend().texts]
					handles_labels = tuple(zip(handles, labels))
					handles_labels = [handle_label for handle_label in handles_labels if not handle_label[1] in ['dataset', 'dataset_type', 'train', 'dev', 'global']]
					handles = [handle for handle, _ in handles_labels]
					labels = [label for _, label in handles_labels]
					ax.legend(handles=handles, labels=labels, fontsize=9)
				else:
					log.warning(f'Not enough data to create line plots for metrics. Try fine-tuning for >1 epoch.')
					return
					# sns.scatterplot(data = metrics[['epoch', metric, 'dataset', 'dataset_type']].dropna(), x = 'epoch', y = metric, ax = ax, hue='dataset', style='dataset_type', legend='full')
					
					# if self.exp_type == 'newverb' and any([re.search(arg_type, metric) for arg_type in self.cfg.tuning.args]):
					# 	# this occurs when we're doing a newverb exp and we want to plot the individual tokens in addition to the mean
					# 	# first, gather the metrics we want on this plot
					# 	like_metrics = [m for m in like_metrics if re.sub(r'\[(.*)\].*', '[\\1]', metric) in m]
					# 	token_metrics = metrics[['epoch'] + like_metrics + ['dataset', 'dataset_type']]
					# 	token_metrics = token_metrics.melt(id_vars=['epoch', 'dataset', 'dataset_type'])
					# 	token_metrics['token'] = [re.sub(r'^([^\s]+).*', '\\1', variable) for variable in token_metrics.variable]
						
						
					# 	v_adjust = (ax.get_ylim()[1] - ax.get_ylim()[0])/100
						
					# 	for t, token_data in token_metrics.groupby('token'):
					# 		token_data = token_data.dropna().reset_index(drop=True)
					# 		sns.scatterplot(data = token_data.dropna(), x='epoch', y='value', ax=ax, hue='dataset', style='dataset_type', legend='none')
					# 		ax.text(max(token_data.epoch)-1, token_data[token_data.epoch == max(token_data.epoch)-1].value-v_adjust, token, size=8, horizontalalignment='center', verticalalignment='top', color='black', zorder=15)
						
					# 	if len(metrics[~metrics.dataset.str.endswith('(train)')].dataset.unique()) > 1:
					# 		sns.lineplot(data = metrics[(~metrics.dataset.str.endswith('(train)')) & (metrics.dataset != 'overall')], x = 'epoch', y = metric, ax = ax, color = palette[-1], ci = 68)
					# 		handles, labels = ax.get_legend_handles_labels()
					# 		handles += [ax.lines[-1]]
					# 		labels += ['mean dev']
					# 		ax.legend(handles=handles,labels=labels)
					# 		ax.lines[-1].set_linestyle(':')
					
					# if len(metrics[~metrics.dataset.str.endswith('(train)')].dataset.unique()) > 1:
					# 	sns.scatterplot(data = metrics, x = 'epoch', y = metric, color = palette[-1], ax = ax, ci = 68)
					# 	handles, labels = ax.get_legend_handles_labels()
					# 	handles += [ax._children[-1]]
					# 	labels += ['mean dev']
					# 	ax.legend(handles=handles,labels=labels)
				
				plt.xticks(xticks)
				
				title = f'{self.model_bert_name} {metric}\n'
				title += f'tuning: {self.cfg.tuning.name.replace("_", " ")}, '
				title += ((f'masking: ' + self.masked_tuning_style) if self.masked else "unmasked") + ', '
				title += f'{"with punctuation" if not self.cfg.hyperparameters.strip_punct else "no punctuation"}\n'
				title += f'epochs: {metrics.epoch.max()} (min: {metrics.min_epochs.unique()[0]}, max: {metrics.max_epochs.unique()[0]}), patience: {self.cfg.hyperparameters.patience} (\u0394={self.cfg.hyperparameters.delta})\n\n'
				
				if metric == 'remaining patience':
					if len(metrics[~metrics.dataset.str.endswith('(train)')].dataset.unique()) > 1:
						# we don't see to say the max for patience, since it is already given and constant for every dataset
						title += f'overall: min @ {global_patience.sort_values(by=metric).reset_index(drop=True)["epoch"][0]}: {int(global_patience.sort_values(by=metric).reset_index(drop=True)[metric][0])}\n'
					
					title += f'{self.cfg.tuning.name.replace("_", " ")} (train): min @ {metrics[metrics.dataset == self.cfg.tuning.name.replace("_"," ") + " (train)"].sort_values(by = metric).reset_index(drop = True)["epoch"][0]}: {int(metrics[metrics.dataset == self.cfg.tuning.name.replace("_", " ") + " (train)"].sort_values(by = metric).reset_index(drop = True)[metric][0])}'
					title += f'\n{self.cfg.tuning.name.replace("_", " ")} (masked, no dropout): min @ {metrics[metrics.dataset == self.cfg.tuning.name.replace("_"," ") + " (masked, no dropout)"].sort_values(by = metric).reset_index(drop = True)["epoch"][0]}: {int(metrics[metrics.dataset == self.cfg.tuning.name.replace("_", " ") + " (masked, no dropout)"].sort_values(by = metric).reset_index(drop = True)[metric][0])}'
					
					for dataset in self.cfg.dev:
						title += f'\n{dataset.replace("_", " ")} (dev): min @ {metrics[metrics.dataset == self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"].sort_values(by = metric).reset_index(drop = True)["epoch"][0]}: {int(metrics[metrics.dataset == self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"].sort_values(by = metric).reset_index(drop = True)[metric][0])}'
				else:
					if len(metrics[~metrics.dataset.str.endswith('(train)')].dataset.unique()) > 1:
						mean = metrics[(~metrics.dataset.str.endswith('(train)')) & (metrics.dataset != 'overall')][['epoch',metric]].groupby('epoch')[metric].agg('mean')
						title += f'mean dev: max @ {int(mean.idxmax())}: {round(mean.max(), 2)}, '
						title += f'min @ {int(mean.idxmin())}: {round(mean.min(), 2)}\n'
					
					# this conditional is added because we do not have metrics for the new argument data in the new verb experiments from the training set with dropout
					if not metrics[metrics.dataset == self.cfg.tuning.name.replace("_", " ") + " (train)"][[metric]].dropna().empty:
						title += f'{self.cfg.tuning.name.replace("_", " ")} (train): max @ {metrics[metrics.dataset == self.cfg.tuning.name.replace("_"," ") + " (train)"].sort_values(by = metric, ascending = False).reset_index(drop = True)["epoch"][0]}: {round(metrics[metrics.dataset == self.cfg.tuning.name.replace("_", " ") + " (train)"].sort_values(by = metric, ascending = False).reset_index(drop = True)[metric][0],2)}, '
						title += f'min @ {metrics[metrics.dataset == self.cfg.tuning.name.replace("_"," ") + " (train)"].sort_values(by = metric).reset_index(drop = True)["epoch"][0]}: {round(metrics[metrics.dataset == self.cfg.tuning.name.replace("_", " ") + " (train)"].sort_values(by = metric).reset_index(drop = True)[metric][0],2)}'
					
					title += f'\n{self.cfg.tuning.name.replace("_", " ")} (masked, no dropout): max @ {metrics[metrics.dataset == self.cfg.tuning.name.replace("_"," ") + " (masked, no dropout)"].sort_values(by = metric, ascending = False).reset_index(drop = True)["epoch"][0]}: {round(metrics[metrics.dataset == self.cfg.tuning.name.replace("_", " ") + " (masked, no dropout)"].sort_values(by = metric, ascending = False).reset_index(drop = True)[metric][0],2)}, '
					title += f'min @ {metrics[metrics.dataset == self.cfg.tuning.name.replace("_"," ") + " (masked, no dropout)"].sort_values(by = metric).reset_index(drop = True)["epoch"][0]}: {round(metrics[metrics.dataset == self.cfg.tuning.name.replace("_", " ") + " (masked, no dropout)"].sort_values(by = metric).reset_index(drop = True)[metric][0],2)}'
					
					for dataset in self.cfg.dev:
						title += f'\n{dataset.replace("_", " ")} (dev): max @ {metrics[metrics.dataset == self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"].sort_values(by = metric, ascending = False).reset_index(drop = True)["epoch"][0]}: {round(metrics[metrics.dataset == self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"].sort_values(by = metric, ascending = False).reset_index(drop = True)[metric][0],2)}, '
						title += f'min @ {metrics[metrics.dataset == self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"].sort_values(by = metric).reset_index(drop = True)["epoch"][0]}: {round(metrics[metrics.dataset == self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"].sort_values(by = metric).reset_index(drop = True)[metric][0],2)}'
				
				title = ax.set_title(title)
				fig.tight_layout()
				fig.subplots_adjust(top=0.7)
				pdf.savefig()
				plt.close('all')
				del fig
	
	
	def most_similar_tokens(self, tokens: List[str] = [], targets: Dict[str,str] = {}, k: int = 50) -> pd.DataFrame:
		"""
		Returns a datafarame containing information about the k most similar tokens to tokens
		or if targets is provided, infomation about the cossim of the tokens to the targets they are mapped to in targets
		"""
		word_embeddings = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight
		
		if not tokens:
			tokens = self.tokens_to_mask
		
		tokens = [t.lower() for t in tokens] if 'uncased' in self.string_id else tokens
		if targets:
			targets = {k.lower() : v for k, v in targets.items()} if 'uncased' in self.string_id else targets
		
		# if we are training roberta, we only currently care about the cases with spaces in front for masked tokens
		# otherwise, try to do something sensible with other tokens
		# if they exist, use them
		# if they have a space in front, replace it with a chr(288)
		# if they don't exist, but a version with a space in front does, use that
		if self.model_bert_name == 'roberta':
			tokens = [t for t in tokens if (t.startswith(chr(288)) and t in self.tokens_to_mask) or not t in self.tokens_to_mask]
			tokens = [t if len(self.tokenizer.tokenize(re.sub(chr(288), ' ', t))) == 1 and not self.tokenizer.tokenize(re.sub(chr(288), ' ', t)) == self.tokenizer.unk_token else ' ' + t if len(self.tokenizer.tokenize(' ' + t)) == 1 and not self.tokenizer.tokenize(' ' + t) == self.tokenizer.unk_token else None for t in tokens]
			tokens = [t for t in tokens if t is not None]
			tokens = [re.sub('^ ', chr(288), t) for t in tokens]
			
			# format the keys in targets ...
			targets = {key if key in tokens else chr(288) + key if chr(288) + key in tokens else '' : v for key, v in targets.items()}
			targets = {key : v for key, v in targets.items() if k}
			targets = {key if len(self.tokenizer.tokenize(re.sub(chr(288), ' ', key))) == 1 and not self.tokenizer.tokenize(re.sub(chr(288), ' ', key)) == self.tokenizer.unk_token else (' ' + key) if len(self.tokenizer.tokenize(' ' + key)) == 1 and not self.tokenizer.tokenize(' ' + key) == self.tokenizer.unk_token else key : 
					   v if len(self.tokenizer.tokenize(re.sub(chr(288), ' ', key))) == 1 and not self.tokenizer.tokenize(re.sub(chr(288), ' ', key)) == self.tokenizer.unk_token else v if len(self.tokenizer.tokenize(' ' + key)) == 1 and not self.tokenizer.tokenize(' ' + key) == self.tokenizer.unk_token else [] for key, v in targets.items()}
			targets = {key : v for key, v in targets.items() if targets[key]}
			targets = {re.sub('^ ', chr(288), key) : v for key, v in targets.items()}
			
			# ... and the values
			for key in targets:
				targets[key] = [t for t in targets[key] if (t.startswith(chr(288)) and t in self.tokens_to_maskey) or not t in self.tokens_to_mask]
				targets[key] = [' ' + t if key.startswith(chr(288)) else t for t in targets[key]] # if the key has a preceding space, then we're only interested in predictions for tokens with preceding spaces
				targets[key] = [t if len(self.tokenizer.tokenize(re.sub(chr(288), ' ', t))) == 1 and not self.tokenizer.tokenize(re.sub(chr(288), ' ', t)) == self.tokenizer.unk_token else None for t in targets[key]]
				targets[key] = [t for t in targets[key] if t is not None]
				targets[key] = [re.sub('^ ', chr(288), t) for t in targets[key]]
			
			targets = {key : v for key, v in targets.items() if all(targets[key])}
		else:
			tokens = [t for t in tokens if len(self.tokenizer.tokenize(t)) == 1 and not self.tokenizer.tokenize(t) == self.tokenizer.unk_token]
			targets = {key : v for key, v in targets.items() if len(self.tokenizer.tokenize(key)) == 1 and not self.tokenizer.tokenize(key) == self.tokenizer.unk_token}
			for key in targets:
				targets[key] = [t for t in targets[key] if len(self.tokenizer.tokenize(t)) == 1 and not self.tokenizer.tokenize(t) == self.tokenizer.unk_token]
			
			targets = {key : v for key, v in targets.items() if all(targets[key])}
		
		cos = nn.CosineSimilarity(dim=-1)
		
		most_similar = {}
		
		for token in tokens:
			token_id = self.tokenizer.get_vocab()[token]
			token_embed = word_embeddings[token_id]
			token_cossim = {i : cossim for i, cossim in enumerate(cos(token_embed, word_embeddings))}
			
			if not token in targets:
				token_cossim = {k : v for k, v in sorted(token_cossim.items(), key = lambda item: -item[1])}
				token_cossim = list(token_cossim.items())
				k_most_similar = []
				for tok_id, cossim in token_cossim:
					most_similar_word = self.tokenizer.convert_ids_to_tokens(tok_id)
					if not tok_id == token_id:
						k_most_similar.extend([(tok_id, str(k) + ' most similar', most_similar_word, cossim.item())])
					
					if len(k_most_similar) == k:
						break
				else:
					continue
				
				most_similar[token] = k_most_similar
			else:
				target_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in targets[token]]
				token_cossim_in_group = {i : cossim for i, cossim in token_cossim.items() if i in target_ids}
				token_cossim_in_group = list(zip([self.tokenizer.convert_ids_to_tokens(i) for i, _ in token_cossim_in_group.items()], [token for t in token_cossim_in_group.items()], list(token_cossim_in_group.items())))
				token_cossim_in_group = [(tok_id, group, t, cossim.item()) for (t, group, (tok_id, cossim)) in token_cossim_in_group]
				most_similar[token] = token_cossim_in_group
				
				out_groups = {k : v for k, v in targets.items() if not k == token}
				if out_groups:
					for out_group_token in out_groups:
						target_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in out_groups[out_group_token]]
						token_cossim_out_group = {i : cossim for i, cossim in token_cossim.items() if i in target_ids}
						token_cossim_out_group = list(zip([self.tokenizer.convert_ids_to_tokens(i) for i, _ in token_cossim_out_group.items()], [out_group_token for t in token_cossim_out_group.items()], list(token_cossim_out_group.items())))
						token_cossim_out_group = [(tok_id, group, t, cossim.item()) for (t, group, (tok_id, cossim)) in token_cossim_out_group]
						most_similar[token].extend(token_cossim_out_group)
		
		if most_similar:
			most_similar_df = pd.DataFrame.from_dict({
				(predicted_arg, *result) :
				(predicted_arg, *result)
				for predicted_arg in most_similar
					for result in most_similar[predicted_arg]
				},
				orient = 'index',
				columns = ['predicted_arg', 'token_id', 'target_group', 'token', 'cossim']
			).reset_index(drop=True).assign(
				model_id = os.path.normpath(os.getcwd()).split(os.sep)[-2] + '-' + self.model_bert_name[0],
				model_name = self.model_bert_name,
				tuning = self.cfg.tuning.name,
				masked = self.masked,
				masked_tuning_style = self.masked_tuning_style,
				strip_punct = self.cfg.hyperparameters.strip_punct
			)
			
			return most_similar_df
		else:
			return pd.DataFrame()
	
	def plot_save_tsnes(self, summary: pd.DataFrame, eval_cfg: DictConfig) -> None:
		n = eval_cfg.num_tsne_words
		set_targets = eval_cfg.data.masked_token_targets if 'masked_token_targets' in eval_cfg.data else {}
		
		dataset_name = summary.eval_data.unique()[0]
		
		epoch_label = '-' + summary.epoch_criteria.unique()[0]
		magnitude = floor(1 + np.log10(summary.total_epochs.unique()[0]))
		epoch_label = f'{str(summary.eval_epoch.unique()[0]).zfill(magnitude)}{epoch_label}'
		
		pos = 'nouns' if not eval_cfg.data.exp_type == 'newverb' else 'verbs'
		
		with open(os.path.join(hydra.utils.get_original_cwd(), 'conf', pos + '.txt'), 'r') as f:
			targets = [w.lower().strip() for w in f.readlines()]
		
		first_n = {k : v for k, v in self.tokenizer.get_vocab().items() if k.replace(chr(288), '').lower() in targets}
		set_targets_dict = {k : v for k, v in self.tokenizer.get_vocab().items() if k.replace(chr(288), '').lower() in list(itertools.chain(*list(set_targets.values())))}
		
		# if we are using roberta, filter to tokens that start with a preceeding space and are not followed by a capital letter (to avoid duplicates))
		if self.model_bert_name == 'roberta':
			first_n = {k : v for k, v in first_n.items() if k.startswith(chr(288)) and not re.search('^' + chr(288) + '[A-Z]', k)}
			set_targets_dict = {k : v for k, v in set_targets_dict.items() if k.startswith(chr(288)) and not re.search('^' + chr(288) + '[A-Z]', k)}
		
		first_n = dict(tuple(first_n.items())[:n])
		
		first_n_embeddings = {k : getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight[v] for k, v in first_n.items()}
		first_n_word_vectors = torch.cat([first_n_embeddings[w].reshape(1, -1) for w in first_n_embeddings], dim=0)
		first_n_words = list(first_n_embeddings.keys())
		
		set_targets_embeddings = {k : getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight[v] for k, v in set_targets_dict.items()}
		set_targets_word_vectors = torch.cat([set_targets_embeddings[w].reshape(1, -1) for w in set_targets_embeddings], dim=0) if set_targets_embeddings else None
		set_targets_words = list(set_targets_embeddings.keys())
		
		added_words = self.tokens_to_mask
		
		# filter out the tokens without added spaces in roberta (since we are not currently doing experiments where those are used)
		if self.model_bert_name == 'roberta':
			added_words = [token for token in added_words if token.startswith(chr(288))]
		
		first_n_words += added_words
		set_targets_words += added_words if set_targets_words else []
		
		added_word_vectors = torch.cat([getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight[token_id].reshape(1,-1) for token_id in self.tokenizer.convert_tokens_to_ids(added_words)], dim=0)
		
		tsne_df = pd.DataFrame(columns=['target_group', 'target_group_label', 'token', 'tsne1', 'tsne2'])
		
		# this conditional can be removed later when everything is updated
		# if 'masked_token_target_labels' in eval_cfg.data.keys():
		target_group_labels = {(k.lower() if 'uncased' in self.string_id else k) : v for k, v in eval_cfg.data.masked_token_target_labels.items()} if 'masked_token_target_labels' in eval_cfg.data else {}
		# else:
		#	target_group_labels = {(k.lower() if 'uncased' in self.string_id else k) : (k.lower() if 'uncased' in self.string_id else k) for k in eval_cfg.data.masked_token_targets}
		
		random_tsne_state = 0
		
		with PdfPages(f'{dataset_name}-{epoch_label}-tsne-plots.pdf') as pdf:
			for word_vectors, words in ((first_n_word_vectors, first_n_words), (set_targets_word_vectors, set_targets_words)):
				if word_vectors is not None:
					tsne = TSNE(2, random_state=random_tsne_state, learning_rate='auto', init='pca')
					with torch.no_grad():
						two_dim = tsne.fit_transform(torch.cat((word_vectors, added_word_vectors)))
					
					two_dim_df = pd.DataFrame(list(zip(words, two_dim[:,0], two_dim[:,1])), columns = ['token', 'tsne1', 'tsne2'])
					two_dim_df['token_category'] = ['existing' if not w in added_words else 'novel' for w in two_dim_df.token.values]
					target_group = [f'first {n}' if not w in added_words else 'novel token' for w in two_dim_df.token.values] if words == first_n_words else []
					if not target_group:
						for w in two_dim_df.token.values:
							if w.replace(chr(288), '') in list(itertools.chain(*list(set_targets.values()))):
								for k in set_targets:
									if w.replace(chr(288), '') in set_targets[k]:
										target_group.append((k.lower() if 'uncased' in self.string_id else k) + ' target')
							else:
								target_group.append('novel token')
					
					two_dim_df['target_group'] = target_group
					two_dim_df['target_group_label'] = [target_group_labels[target_group.replace(' target', '')] if target_group.replace(' target', '') in target_group_labels else target_group for target_group in two_dim_df.target_group]
					two_dim_df['tsne_type'] = f'first {n}' if words == first_n_words else 'set targets'
					
					fig, ax = plt.subplots(1)
					fig.set_size_inches(12, 10)
					
					sns.scatterplot(data = two_dim_df.sort_values(by=['target_group'], key = lambda col: -col.str.match('^novel token$')), x = 'tsne1', y = 'tsne2', s=18, ax=ax, hue='target_group_label', legend='full')
					v_adjust = (ax.get_ylim()[1] - ax.get_ylim()[0])/150
					
					for line in range(len(two_dim_df)):
						if two_dim_df.loc[line].token in added_words:
							ax.text(two_dim_df.loc[line].tsne1, two_dim_df.loc[line].tsne2-v_adjust, two_dim_df.loc[line].token.replace(chr(288), ''), size=10, horizontalalignment='center', verticalalignment='top', color='black')
						else:
							ax.text(two_dim_df.loc[line].tsne1, two_dim_df.loc[line].tsne2-v_adjust, two_dim_df.loc[line].token.replace(chr(288), ''), size=6, horizontalalignment='center', verticalalignment='top', color='black')
					
					legend = [c for c in ax.get_children() if isinstance(c, matplotlib.legend.Legend)][0]
					legend._legend_title_box._text._text = legend._legend_title_box._text._text.replace('_', ' ').replace(' label', '')
					
					ax.set_xlabel('t-SNE 1', fontsize=8)
					ax.set_ylabel('t-SNE 2', fontsize=8)
					
					if words == first_n_words:
						title = f'{self.model_bert_name} t-SNEs of first {n} token(s) and novel token(s) (filtered)'
					else:
						title = f'{self.model_bert_name} t-SNEs of {summary.eval_data.unique()[0]} target group(s) token(s) and novel token(s) (filtered)'
					
					title += f' @ epoch {summary.eval_epoch.unique()[0]}/{summary.total_epochs.unique()[0]} ({summary.epoch_criteria.unique()[0].replace("_", " ")})\n'
					title += f'min epochs: {summary.min_epochs.unique()[0]}, '
					title += f'max epochs: {summary.max_epochs.unique()[0]}'
					title += f', patience: {summary.patience.unique()[0]}'
					title += f' (\u0394={summary.delta.unique()[0]})\n'
					title += f'tuning: {summary.tuning.unique()[0]}, '
					title += ((f'masking: ' + summary.masked_tuning_style.unique()[0]) if summary.masked.unique()[0] else "unmasked") + ', '
					title += f'{"with punctuation" if not summary.strip_punct.unique()[0] else "no punctuation"}'
					
					fig.suptitle(title)
					fig.tight_layout()
					
					pdf.savefig()
					plt.close()
					
					tsne_df_tmp = two_dim_df
					tsne_df_tmp['token_id'] = [self.tokenizer.convert_tokens_to_ids(token) for token in words]
					tsne_df_tmp = tsne_df_tmp.assign(
						model_id = summary.model_id.unique()[0],
						model_name = summary.model_name.unique()[0],
						eval_data = summary.eval_data.unique()[0],
						tuning = summary.tuning.unique()[0],
						masked = summary.masked.unique()[0],
						masked_tuning_style = summary.masked_tuning_style.unique()[0],
						strip_punct = summary.strip_punct.unique()[0],
						eval_epoch = summary.eval_epoch.unique()[0],
						total_epochs = summary.total_epochs.unique()[0],
						patience = summary.patience.unique()[0],
						delta = summary.delta.unique()[0],
						min_epochs = summary.min_epochs.unique()[0],
						max_epochs = summary.max_epochs.unique()[0],
						epoch_criteria = summary.epoch_criteria.unique()[0],
						random_seed = summary.random_seed.unique()[0],
						random_tsne_state = random_tsne_state,
					)
					
					tsne_df = pd.concat([tsne_df, tsne_df_tmp], ignore_index=True)
		
		tsne_df.to_csv(f'{dataset_name}-{epoch_label}-tsne.csv.gz', index=False)
	
	def plot_cossims(self, cossims: pd.DataFrame) -> None:
		cossims = cossims[~cossims.target_group.str.endswith('most similar')].copy().reset_index(drop=True)
		if cossims.empty:
			log.info('No target groups were provided for cosine similarities. No comparison plots for cosine similarities can be created.')
			return
		
		if len(cossims.predicted_arg.unique()) <= 1:
			log.info(f'One or fewer predicted arguments were provided for cosine similarities ({cossims.target_group.unique()[0]}). No comparison plots for cosine similarities can be created.')
			return
		
		# we do this swap to fix the labels (without losing any data)
		# if the dataframe contains info about models other than roberta, this will already have been fixed the multieval script, so we don't touch it
		if 'roberta' in cossims.model_name.unique() and len(cossims.model_name.unique()) == 1:
			for col in ['predicted_arg', 'target_group']:
				# first, replace the ones that don't start with spaces before with a preceding ^
				cossims.loc[(cossims['model_name'] == 'roberta') & ~(cossims[col].str.startswith(chr(288))), col] = \
					cossims[(cossims['model_name'] == 'roberta') & ~(cossims[col].str.startswith(chr(288)))][col].str.replace(r'^(.)', r'^\1', regex=True)
				
				# then, replace the ones with the preceding special character (since we are mostly using them in the middle of sentences)
				cossims.loc[(cossims['model_name'] == 'roberta') & (cossims.token.str.startswith(chr(288))), col] = \
					cossims[(cossims['model_name'] == 'roberta') & (cossims.token.str.startswith(chr(288)))][col].str.replace(chr(288), '')
		
		if len(cossims.model_id.unique()) > 1:
			cossims['cossim'] = cossims['mean']
			cossims = cossims.drop('mean', axis = 1)
		else:
			cossims['sem'] = 0
		
		filename = cossims.eval_data.unique()[0] + '-'
		epoch_label = cossims.epoch_criteria.unique()[0] if len(cossims.epoch_criteria.unique()) == 1 else ''
		if len(cossims.model_id.unique()) == 1:
			epoch_label = '-' + epoch_label
			magnitude = floor(1 + np.log10(cossims.total_epochs.unique()[0]))
			epoch_label = f'{str(cossims.eval_epoch.unique()[0]).zfill(magnitude)}{epoch_label}'
		
		filename += epoch_label + '-cossims-plot.pdf'
		
		idx_col = 'token' if len(cossims.model_id.unique()) == 1 else 'model_id'
		
		group = cossims[['predicted_arg', 'target_group_label', idx_col, 'cossim']]
		group_sems = cossims[['predicted_arg', 'target_group_label', idx_col, 'sem']]
		
		if idx_col == 'model_id' and len(cossims.model_name.unique()) > 1:
			model_means = cossims.groupby(['model_name', 'predicted_arg']).cossim.agg('mean')
			model_means = model_means.reset_index()
		
		group = group.pivot(index=['target_group_label', idx_col], columns='predicted_arg', values='cossim')
		group.columns.name = None
		group = group.reset_index()
		
		group_sems = group_sems.pivot(index=['target_group_label', idx_col], columns='predicted_arg', values='sem')
		group_sems.columns.name = None
		group_sems = group_sems.reset_index()
		
		pairs = [c for c in group.columns if not c in [idx_col, 'target_group_label']]
		pairs = [pair for pair in itertools.combinations(pairs, 2) if not pair[0] == pair[1]]
		pairs = list(set(tuple(sorted(pair)) for pair in pairs))
		
		fig, ax = plt.subplots(len(pairs), 2)
		ax = ax.reshape(len(pairs), 2)
		fig.set_size_inches(12.5, (6*len(pairs))+(0.6*len(cossims.predicted_arg.unique()))+(0.6*len(cossims.target_group.unique()))+0.25)
		
		for i, (in_token, out_token) in enumerate(pairs):
			# we might be able to use a sns.jointplot to plot histograms instead of just ticks for the means,
			# but this causes other complex problems that I haven't figured out yet. So we'll stick with the simple thing for now
			sns.scatterplot(data=group, x=in_token, y=out_token, ax=ax[i][0], zorder=5, hue='target_group_label', linewidth=0)
			legend = [c for c in ax[i][0].get_children() if isinstance(c, matplotlib.legend.Legend)][0]
			legend._legend_title_box._text._text = legend._legend_title_box._text._text.replace('_', ' ').replace(' label', '')
			
			collections = ax[i][0].collections[1:].copy()
			for (_, eb_group), (_, eb_group_sems), collection in zip(group.groupby('target_group_label'), group_sems.groupby('target_group_label'), collections):
				ax[i][0].errorbar(data=eb_group, x=in_token, xerr=eb_group_sems[in_token], y=out_token, yerr=eb_group_sems[out_token], color=collection._original_edgecolor, ls='none', zorder=2.5)
			
			ulim = max([*ax[i][0].get_xlim(), *ax[i][0].get_ylim()])
			llim = min([*ax[i][0].get_xlim(), *ax[i][0].get_ylim()])
			
			# we do this so longer text can fit inside the plot instead of overflowing
			v_adjust = (ulim-llim)/90 if idx_col == 'token' else 0
			range_mean_tick = (ulim-llim)/90
			
			ulim += v_adjust + (ulim-llim)/90
			llim -= (v_adjust + (ulim-llim)/90)
			
			ax[i][0].set_xlim((llim, ulim))
			ax[i][0].set_ylim((llim, ulim))
			
			# here we add ticks to show the mean and standard errors along each axis
			group_means = group.drop(idx_col, axis=1).groupby(['target_group_label']).agg({'mean', 'sem'})
			cols = list(set([c[0] for c in group_means.columns]))
			group_means.columns = ['_'.join(c) for c in group_means.columns]
			for predicted_arg in cols:
				for target_group, collection in zip(group_means.index, collections):
					if predicted_arg == in_token:
						ax[i][0].plot((group_means.loc[target_group][predicted_arg + '_mean'], group_means.loc[target_group][predicted_arg + '_mean']), (llim, llim+range_mean_tick*3), linestyle='-', color=collection._original_edgecolor, zorder=0, scalex=False, scaley=False, alpha=0.3)
						ax[i][0].plot(
							(
								group_means.loc[target_group][predicted_arg + '_mean']-group_means.loc[target_group][predicted_arg + '_sem'],
							 	group_means.loc[target_group][predicted_arg + '_mean']+group_means.loc[target_group][predicted_arg + '_sem']
							), 
							(llim+range_mean_tick*1.5, llim+range_mean_tick*1.5),
							linestyle='-', linewidth=0.75, color=collection._original_edgecolor, zorder=0, scalex=False, scaley=False, alpha=0.3
						)
					else:
						ax[i][0].plot((llim, llim+range_mean_tick*3), (group_means.loc[target_group][predicted_arg + '_mean'], group_means.loc[target_group][predicted_arg + '_mean']), linestyle='-', color=collection._original_edgecolor, zorder=0, scalex=False, scaley=False, alpha=0.3)
						ax[i][0].plot(
							(llim+range_mean_tick*1.5, llim+range_mean_tick*1.5), 
							(
								group_means.loc[target_group][predicted_arg + '_mean']-group_means.loc[target_group][predicted_arg + '_sem'], 
								group_means.loc[target_group][predicted_arg + '_mean']+group_means.loc[target_group][predicted_arg + '_sem']
							), 
							linestyle='-', linewidth=0.75, color=collection._original_edgecolor, zorder=0, scalex=False, scaley=False, alpha=0.3
						)
			
			ax[i][0].set_aspect(1./ax[i][0].get_data_ratio(), adjustable='box')
			ax[i][0].plot((llim, ulim), (llim, ulim), linestyle='--', color='black', scalex=False, scaley=False, zorder=0, alpha=0.3)
			
			if idx_col == 'token':
				for line in range(0, len(group)):
					ax[i][0].text(group.loc[line][in_token], group.loc[line][out_token]-(v_adjust if group_sems.loc[line][out_token] == 0 else (group_sems.loc[line][out_token]+(v_adjust/2))), group.loc[line].token.replace(chr(288), ''), size=6, horizontalalignment='center', verticalalignment='top', color='black', zorder=15)
			elif len(cossims.model_name.unique()) > 1:
				for model_name in model_means.model_name:
					ax[i][0].text(
						model_means[(model_means.model_name == model_name) & (model_means.predicted_arg == in_token)].cossim.values[0], 
						model_means[(model_means.model_name == model_name) & (model_means.predicted_arg == out_token)].cossim.values[0], 
						model_name, size=10, horizontalalignment='center', verticalalignment='center', color='black', zorder=15, alpha=0.65, fontweight='bold', path_effects=[pe.withStroke(linewidth=2, foreground='white')]
					)
			
			ax[i][0].set_xlabel(f'{in_token} cosine similarity')
			ax[i][0].set_ylabel(f'{out_token} cosine similarity')
			
			# y = y - x plot, to show the extent to which the out group token is more similar to the target group tokens than the desired token
			sns.scatterplot(data=group, x=in_token, y=group[out_token]-group[in_token], ax=ax[i][1], zorder=10, hue='target_group_label', linewidth=0)
			legend = [c for c in ax[i][1].get_children() if isinstance(c, matplotlib.legend.Legend)][0]
			legend._legend_title_box._text._text = legend._legend_title_box._text._text.replace('_', ' ').replace(' label', '')
			
			collections = ax[i][1].collections[1:].copy()
			for (_, eb_group), (_, eb_group_sems), collection in zip(group.groupby('target_group_label'), group_sems.groupby('target_group_label'), collections):
				ax[i][1].errorbar(x=eb_group[in_token], xerr=eb_group_sems[in_token], y=eb_group[out_token]-eb_group[in_token], yerr=eb_group_sems[out_token], color=collection._original_edgecolor, ls='none', zorder=2.5)
			
			ax[i][1].set_xlim((llim, ulim))
			ax[i][1].plot((llim, ulim), (0, 0), linestyle='--', color='black', scalex=False, scaley=False, zorder=0, alpha=0.3)
			
			ulim = max([abs(v) for v in [*ax[i][1].get_ylim()]])
			llim = -ulim
			
			v_adjust = (ulim-llim)/90 if idx_col == 'token' else 0
			# we do this so longer text can fit inside the plot instead of overflowing
			ulim += v_adjust + (ulim-llim)/90
			llim -= (v_adjust + (ulim-llim)/90)
			ax[i][1].set_ylim((llim, ulim))
			
			if idx_col == 'token':
				for line in range(0, len(group)):
					ax[i][1].text(group.loc[line][in_token], group.loc[line][out_token]-group.loc[line][in_token]-(v_adjust if group_sems.loc[line][out_token] == 0 else (group_sems.loc[line][out_token]+(v_adjust/2))), group.loc[line].token.replace(chr(288), ''), size=6, horizontalalignment='center', verticalalignment='top', color='black', zorder=10)
			elif len(cossims.model_name.unique()) > 1:
				for model_name in model_means.model_name:
					ax[i][1].text(
						model_means[(model_means.model_name == model_name) & (model_means.predicted_arg == in_token)].cossim.values[0], 
						model_means[(model_means.model_name == model_name) & (model_means.predicted_arg == out_token)].cossim.values[0] - model_means[(model_means.model_name == model_name) & (model_means.predicted_arg == in_token)].cossim.values[0],
						model_name, size=10, horizontalalignment='center', verticalalignment='center', color='black', zorder=15, alpha=0.65, fontweight='bold', path_effects=[pe.withStroke(linewidth=2, foreground='white')])
			
			ax[i][1].set_aspect(1./ax[i][1].get_data_ratio(), adjustable='box')
			
			ax[i][1].set_xlabel(f'{in_token} cosine similarity')
			ax[i][1].set_ylabel(f'{out_token} \u2212 {in_token} cosine similarity')
		
		title = cossims.model_name.unique()[0] if len(cossims.model_name.unique()) == 1 else f"Multiple models'"
		title += f' cosine similarities to '
		title += cossims.eval_data.unique()[0] if len(cossims.eval_data.unique()) == 1 else f"{len(cossims.eval_data.unique())} eval sets'"
		title += f' target group tokens'
		title += (' @ epoch ' + str(cossims.eval_epoch.unique()[0]) + '/') if len(cossims.eval_epoch.unique()) == 1 else ', epochs: '
		title += str(cossims.total_epochs.unique()[0]) if len(cossims.total_epochs.unique()) == 1 else 'multiple'
		title += f' ({cossims.epoch_criteria.unique()[0].replace("_", " ")})' if len(cossims.epoch_criteria.unique()) == 1 else ' (multiple criteria)'
		title += f'\nmin epochs: {cossims.min_epochs.unique()[0] if len(cossims.min_epochs.unique()) == 1 else "multiple"}, '
		title += f'max epochs: {cossims.max_epochs.unique()[0] if len(cossims.max_epochs.unique()) == 1 else "multiple"}'
		title += f', patience: {cossims.patience.unique()[0] if len(cossims.patience.unique()) == 1 else "multiple"}'
		title += f' (\u0394={cossims.delta.unique()[0] if len(cossims.delta.unique()) == 1 else "multiple"})'
		title += '\ntuning: ' + (cossims.tuning.unique()[0].replace("_", " ") if len(cossims.tuning.unique()) == 1 else "multiple")
		title += ', masking' if all(cossims.masked == True) else ' unmasked' if all(1 - (cossims.masked == True)) else ''
		title += (': ' + cossims.masked_tuning_style[(cossims.masked == True)].unique()[0] if cossims.masked_tuning_style[(cossims.masked == True)].unique().size == 1 else '') if not 'multiple' in cossims.masked_tuning_style[cossims.masked == True].unique() else ', masking: multiple' if any(cossims.masked == 'multiple') or any(cossims.masked == True) else ''
		title += ', ' + ('no punctuation' if all(cossims.strip_punct == True) else "with punctuation" if len(cossims.strip_punct.unique()) == 1 and not any(cossims.strip_punct == True) else 'multiple punctuation')
		title += '\n'
		
		# this conditional is a workaround for now. it should be able to be removed later once we rerun the results and add this info to every file
		# if 'target_group_label' in cossims.columns:
		target_group_labels = cossims[['target_group', 'target_group_label']].drop_duplicates()
		target_group_labels = target_group_labels.groupby('target_group').apply(lambda x: x.to_dict(orient='records')[0]['target_group_label']).to_dict()
		# else:
		#	target_group_labels = {target_group : target_group for target_group in cossims.target_group.unique()}
		
		if len(cossims.target_group.unique()) > 1:
			for target_group, df in cossims.groupby('target_group'):
				means = df.groupby('predicted_arg').cossim.agg({'mean', 'sem', 'std', 'size'})
				out_group_means = means.loc[[i for i in means.index if not i == target_group]]
				exprs = [
					(
						f'\nMean cosine similarity of {target_group} to {target_group_labels[target_group]} \u2212 {arg} to {target_group_labels[target_group]} targets: ' +
						'{:.4f}'.format(means['mean'][target_group]) + ' (\u00b1' + '{:.4f}'.format(means['sem'][target_group]) + ') \u2212 ' +
						'{:.4f}'.format(out_group_means['mean'][arg]) + ' (\u00b1' + '{:.4f}'.format(out_group_means['sem'][arg]) + ') = ' +
						'{:.4f}'.format(means['mean'][target_group] - out_group_means['mean'][arg]) + ' (\u00b1' + '{:.4f}'.format(sqrt(((means['std'][target_group]**2)/means['size'][target_group]) + ((out_group_means['std'][arg]**2)/out_group_means['size'][arg]))) + ')'
					).replace('-', '\u2212') 
					for arg in out_group_means.index
				]
				
				for expr in exprs:
					title += expr
			
			title += '\n'
		
		for predicted_arg, df in cossims.groupby('predicted_arg'):
			means = df.groupby('target_group').cossim.agg({'mean', 'sem', 'std', 'size'})
			out_group_means = means.loc[[i for i in means.index if not i == predicted_arg]]
			exprs = [
				(
					f'\nMean cosine similarity of {predicted_arg} to {target_group_labels[predicted_arg]} \u2212 {predicted_arg} to {target_group_labels[arg]} targets: ' +
					'{:.4f}'.format(means['mean'][predicted_arg]) + ' (\u00b1' + '{:.4f}'.format(means['sem'][predicted_arg]) + ') \u2212 ' +
					'{:.4f}'.format(out_group_means['mean'][arg]) + ' (\u00b1' + '{:.4f}'.format(out_group_means['sem'][arg]) + ') = ' + 
					'{:.4f}'.format(means['mean'][predicted_arg] - out_group_means['mean'][arg]) + ' (\u00b1' + '{:.4f}'.format(sqrt(((means['std'][predicted_arg]**2)/means['size'][predicted_arg]) + ((out_group_means['std'][arg]**2)/out_group_means['size'][arg]))) + ')'
				).replace('-', '\u2212')
				for arg in out_group_means.index
			]
			
			for expr in exprs:
				title += expr
		
		fig.suptitle(title)
		
		fig.tight_layout()
		
		plt.savefig(filename)
		plt.close()
	
	def get_original_random_seed(self) -> int:
		path = 'tune.log'
		if not path in os.listdir(os.getcwd()):
			path = os.path.join('..', path)
		
		try:
			with open(path, 'r') as logfile_stream:
				logfile = logfile_stream.read()
			
			seed = int(re.findall(r'Seed set to ([0-9]*)\n', logfile)[0])
			return seed
		except (IndexError, FileNotFoundError):
			pass
		
		path = 'weights.pkl.gz'
		if not path in os.listdir(os.getcwd()):
			path = os.path.join('..', path)
		
		try: 
			with gzip.open(path, 'rb') as weightsfile_stream:
				weights = pkl.load(weightsfile_stream)
			
			seed = weights['random_seed']
			return seed
		except (IndexError, FileNotFoundError):
			log.error(f'Seed not found in log file or weights file in {os.path.split(path)[0]}!')
			return
	
	
	def eval(self, eval_cfg: DictConfig, checkpoint_dir: str) -> None:
		self.model.eval()
		epoch_label = ('-' + eval_cfg.epoch) if isinstance(eval_cfg.epoch, str) else '-manual'
		epoch, total_epochs = self.restore_weights(checkpoint_dir, eval_cfg.epoch)
		
		dataset_name = eval_cfg.data.friendly_name
		magnitude = floor(1 + np.log10(total_epochs))
		epoch_label = f'{str(epoch).zfill(magnitude)}{epoch_label}'
		most_similar_tokens = self.most_similar_tokens(k=eval_cfg.k).assign(eval_epoch=epoch, total_epochs=total_epochs)
		most_similar_tokens = pd.concat([most_similar_tokens, self.most_similar_tokens(targets=eval_cfg.data.masked_token_targets).assign(eval_epoch=epoch, total_epochs=total_epochs)], ignore_index=True)
		
		predicted_roles = {(v.lower() if 'uncased' in self.string_id else v) : k for k, v in eval_cfg.data.eval_groups.items()}
		target_group_labels = {(k.lower() if 'uncased' in self.string_id else k) : v for k, v in eval_cfg.data.masked_token_target_labels.items()}
		
		most_similar_tokens = most_similar_tokens.assign(
			predicted_role=[predicted_roles[arg.replace(chr(288), '')] for arg in most_similar_tokens['predicted_arg']],
			target_group_label=[target_group_labels[group.replace(chr(288), '')] if not group.endswith('most similar') and group.replace(chr(288), '') in target_group_labels else group for group in most_similar_tokens.target_group],
			eval_data=eval_cfg.data.friendly_name,
			patience=self.cfg.hyperparameters.patience,
			delta=self.cfg.hyperparameters.delta,
			min_epochs=self.cfg.hyperparameters.min_epochs,
			max_epochs=self.cfg.hyperparameters.max_epochs,
			epoch_criteria=eval_cfg.epoch if isinstance(eval_cfg.epoch, str) else 'manual',
		)
		
		most_similar_tokens.to_csv(f'{dataset_name}-{epoch_label}-cossims.csv.gz', index=False)
		
		log.info('Creating cosine similarity plots')
		self.plot_cossims(most_similar_tokens)
		
		# Load data
		# the use of eval_cfg.data.to_mask will probably need to be updated here for roberta now
		inputs, labels, sentences = self.load_eval_file(eval_cfg.data.name, eval_cfg.data.to_mask)
		
		# Calculate results on given data
		with torch.no_grad():	
			log.info("Evaluating model on testing data")
			outputs = self.model(**inputs)
		
		results = self.collect_results(inputs, eval_cfg.data.eval_groups, outputs)
		summary = self.summarize_results(results, labels)
		
		log.info(f'Creating t-SNE plots')
		self.plot_save_tsnes(summary, eval_cfg)
		
		log.info("Creating aconf and entropy plots")
		self.graph_results(results, summary, eval_cfg)
	
	def load_eval_file(self, data_path: str, replacing: Dict[str,str]) -> Tuple[Dict,Dict,List[str]]:
		"""
		Loads a file from the specified path, returning a tuple of (input, label)
		for model evaluation.
		"""
		resolved_path = os.path.join(
			hydra.utils.get_original_cwd(),
			"data",
			data_path
		)
		
		with open(resolved_path, "r") as f:
			raw_sentences = [line.strip() for line in f]
			raw_sentences = [r.lower() for r in raw_sentences] if 'uncased' in self.string_id else raw_sentences
			sentences = raw_sentences
		
		masked_sentences = []
		for s in sentences:
			m = s
			for tok in self.tokens_to_mask:
				m = m.replace(tok, self.mask_tok)
			masked_sentences.append(m)
		
		inputs = self.tokenizer(masked_sentences, return_tensors="pt", padding=True)
		labels = self.tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"]
		
		if not verify_tokenization_of_sentences(self.tokenizer, [sentences] + [masked_sentences], self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs):
			log.warning('Tokenization of sentences was affected by the new tokens! Try choosing a new string.')
			return
		
		return inputs, labels, sentences
	
	def summarize_results(self, results: Dict, labels: torch.Tensor) -> Dict:
		
		summary = {}
		
		# Define theme and recipient ids
		ricket = 'RICKET' if not 'uncased' in self.string_id else 'ricket'
		thax = 'THAX' if not 'uncased' in self.string_id else 'thax'
		
		ricket = self.tokenizer(ricket, return_tensors="pt")["input_ids"][:,1]
		thax = self.tokenizer(thax, return_tensors="pt")["input_ids"][:,1]
		
		# Cumulative log probabilities for <token> in <position>
		theme_in_theme = []
		theme_in_recipient = []
		recipient_in_theme = []
		recipient_in_recipeint = []
		
		# Confidence in predicting <token> over the alternative
		ricket_confidence = []
		thax_confidence = []
		
		# Confidence that position is an <animacy> noun
		animate_confidence = []
		inanimate_confidence = []
		
		# Entropies in various positions
		theme_entropy = []
		recipient_entropy = []
		
		for i in results:
			label = labels[i]
			result = results[i]
			
			for idx in result:
				
				target = label[idx.item()]
				scores = result[idx]['mean grouped log probability']
				probabilities = result[idx]['probabilities']
				
				categorical_distribution = Categorical(probs=probabilities)
				entropy = categorical_distribution.entropy()
				
				if target == ricket:
					theme_in_recipient.append(scores['theme'])
					recipient_in_recipeint.append(scores['recipient'])
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
	
	def graph_results(self, results: Dict, summary: Dict, eval_cfg: DictConfig) -> None:
		
		dataset = str(eval_cfg.data.name).split('.')[0]
		
		fig, axs = plt.subplots(2, 2, sharey='row', sharex='row', tight_layout=True)
		
		theme_entr = [x.item() for x in summary['theme']['entropy']]
		recip_entr = [x.item() for x in summary['recipient']['entropy']]
		
		inan = summary['theme']['animacy_conf']
		anim = summary['recipient']['animacy_conf']
		
		# Entropy Plots
		axs[0][0].hist(theme_entr)
		axs[0][0].axvline(np.mean(theme_entr), color='r')
		axs[0][0].set_title('entropy [theme]')
		
		axs[0][1].hist(recip_entr)
		axs[0][1].axvline(np.mean(recip_entr), color='r')
		axs[0][1].set_title('entropy [recipient]')
		
		# Animacy Plots
		
		axs[1][0].hist(inan)
		axs[1][0].axvline(np.mean(inan), color='r')
		axs[1][0].set_title('animacy confidence [theme]')
		
		axs[1][1].hist(anim)
		axs[1][1].axvline(np.mean(anim), color='r')
		axs[1][1].set_title('animacy confidence [recipient]')
		
		fig.suptitle(f"{eval_cfg.data.description}")
		
		plt.savefig(f"{dataset}.png")
		
		with open(f"{dataset}-scores.npy", "wb") as f:
			np.save(f, np.array(theme_entr))
			np.save(f, np.array(recip_entr))
			np.save(f, np.array(inan))
			np.save(f, np.array(anim))
	
	
	def eval_entailments(self, eval_cfg: DictConfig, checkpoint_dir: str) -> None:
		"""
		Computes model performance on data consisting of 
			sentence 1 , sentence 2 , [...]
		where credit for a correct prediction on sentence 2[, 3, ...] is contingent on
		also correctly predicting sentence 1
		"""
		log.info(f"SAVING TO: {os.getcwd().replace(hydra.utils.get_original_cwd(), '')}")
		
		# Load model
		self.model.eval()
		epoch_label = ('-' + eval_cfg.epoch) if isinstance(eval_cfg.epoch, str) else '-manual'
		epoch, total_epochs = self.restore_weights(checkpoint_dir, eval_cfg.epoch)
		
		dataset_name = eval_cfg.data.friendly_name
		magnitude = floor(1 + np.log10(total_epochs))
		epoch_label = f'{str(epoch).zfill(magnitude)}{epoch_label}'
		
		most_similar_tokens = self.most_similar_tokens(k=eval_cfg.k).assign(eval_epoch=epoch, total_epochs=total_epochs)
		most_similar_tokens = pd.concat([most_similar_tokens, self.most_similar_tokens(targets=eval_cfg.data.masked_token_targets).assign(eval_epoch=epoch, total_epochs=total_epochs)], ignore_index=True)
		
		predicted_roles = {(v.lower() if 'uncased' in self.string_id else v) : k for k, v in eval_cfg.data.eval_groups.items()}
		target_group_labels = {(k.lower() if 'uncased' in self.string_id else k) : v for k, v in eval_cfg.data.masked_token_target_labels.items()}
		
		most_similar_tokens = most_similar_tokens.assign(
			predicted_role=[predicted_roles[arg.replace(chr(288), '')] for arg in most_similar_tokens['predicted_arg']],
			target_group_label=[target_group_labels[group.replace(chr(288), '')] if not group.endswith('most similar') and group.replace(chr(288), '') in target_group_labels else group for group in most_similar_tokens.target_group],
			eval_data=eval_cfg.data.friendly_name,
			patience=self.cfg.hyperparameters.patience,
			delta=self.cfg.hyperparameters.delta,
			min_epochs=self.cfg.hyperparameters.min_epochs,
			max_epochs=self.cfg.hyperparameters.max_epochs,
			epoch_criteria=eval_cfg.epoch if isinstance(eval_cfg.epoch, str) else 'manual',
			random_seed=self.get_original_random_seed()
		)
		
		most_similar_tokens.to_csv(f'{dataset_name}-{epoch_label}-cossims.csv.gz', index=False)
		
		log.info('Creating cosine similarity plots')
		self.plot_cossims(most_similar_tokens)
		
		data = self.load_eval_entail_file(eval_cfg.data.name, eval_cfg.data.to_mask)
		inputs = data["inputs"]
		labels = data["labels"]
		sentences = data["sentences"]
		
		assert len(inputs) == len(labels), f"Inputs (size {len(inputs)}) must match labels (size {len(labels)}) in length"
		
		# Calculate performance on data
		with torch.no_grad():
			log.info('Evaluating model on testing data')
			outputs = [self.model(**i) for i in tqdm(inputs)]
		
		summary = self.get_entailed_summary(sentences, outputs, labels, eval_cfg)
		summary = summary.assign(eval_epoch=epoch, total_epochs=total_epochs)
		
		# save the summary as a pickle and as a csv so that we have access to the original tensors
		# these get converted to text in the csv, but the csv is easier to work with otherwise
		summary.to_pickle(f"{dataset_name}-{epoch_label}-odds_ratios.pkl.gz")
		
		summary_csv = summary.copy()
		summary_csv['odds_ratio'] = summary_csv.odds_ratio.astype(float)
		summary_csv.to_csv(f"{dataset_name}-{epoch_label}-odds_ratios.csv.gz", index = False, na_rep = 'NaN')
		
		log.info('Creating t-SNE plots')
		self.plot_save_tsnes(summary, eval_cfg)
		
		log.info('Creating odds ratios plots')
		self.graph_entailed_results(summary, eval_cfg)
		
		acc = self.get_entailed_accuracies(summary)
		acc.to_csv(f'{dataset_name}-{epoch_label}-accuracies.csv.gz', index = False, na_rep = 'NaN')
		
		log.info('Evaluation complete')
		print('')
	
	def load_eval_entail_file(self, data_path: str, replacing: Dict[str, str]) -> Dict:
		resolved_path = os.path.join(hydra.utils.get_original_cwd(),"data",data_path)
		
		with open(resolved_path, "r") as f:
			raw_input = [line.strip() for line in f]
		
		raw_input = [r.lower() for r in raw_input] if 'uncased' in self.string_id else raw_input
		
		if self.cfg.hyperparameters.strip_punct:
			raw_input = [strip_punct(line) for line in raw_input]
				
		sentences = [[s.strip() for s in r.split(' , ')] for r in raw_input]
		
		masked_sentences = []
		for s_group in sentences:
			m_group = []
			for s in s_group:
				m = s
				for val in self.tokens_to_mask:
					m = m.replace(val, self.mask_tok)
				
				m_group.append(m)
			
			masked_sentences.append(m_group)

		sentences_transposed = list(map(list, zip(*sentences)))
		masked_transposed = list(map(list, zip(*masked_sentences)))
		
		inputs = [self.tokenizer(m, return_tensors="pt", padding=True) for m in masked_transposed]
		labels = [self.tokenizer(s, return_tensors="pt", padding=True)["input_ids"] for s in sentences_transposed]
		
		if not verify_tokenization_of_sentences(self.tokenizer, [sentences] + [masked_sentences], self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs):
			log.warning('Tokenization of sentences was affected by the new tokens! Try choosing a new string.')
			return
		
		return {"inputs" : inputs, "labels" : labels, "sentences" : sentences}
	
	def get_entailed_summary(self, sentences: List[List[str]], outputs: List['MaskedLMOutput'], labels: List[torch.Tensor], eval_cfg: DictConfig) -> pd.DataFrame:
		"""
		Returns a pandas.DataFrame summarizing the model state.
		The dataframe contains the log odds ratios for all target tokens relative to all non-target tokens
		for role position and sentence type.
		Output columns are:
			sentence_type: the sentence_type label as set in the config file
			ratio_name: text description of the odds ratio
			odds_ratio: the numerical value of the odds ratio described by ratio_name
			role_position: the expected thematic role associated with the position
			position_num: the linear order of the position among the masked tokens in the sentence
			sentence: the raw sentence
		"""
		sentence_types = eval_cfg.data.sentence_types
		tokens_to_roles = {v : k for k, v in eval_cfg.data.eval_groups.items()}
		# convert the tokens to lowercase if we are using an uncased model
		if 'uncased' in self.string_id:
			tokens_to_roles = {k.lower() : v for k, v in tokens_to_roles.items()}
		# we need to add the special 'space before' versions of the tokens if we're using roberta
		if self.model_bert_name == 'roberta':
			old_tokens_to_roles = tokens_to_roles.copy()
			for token in old_tokens_to_roles:
				tokens_to_roles.update({chr(288) + token : old_tokens_to_roles[token]})
		
		sentence_type_logprobs = {}
		
		for output, sentence_type in zip(outputs, sentence_types):
			sentence_type_logprobs[sentence_type] = nn.functional.log_softmax(output.logits, dim = 2)
		
		# Get the positions of the tokens in each sentence of each type
		tokens_indices = dict(zip(
			self.tokens_to_mask, 
			self.tokenizer.convert_tokens_to_ids(self.tokens_to_mask)
		))
		
		############################################################# Temporary, until I think of a better way
		############################################################# We are currently only using the tokens with spaces before them in RoBERTa
		############################################################# so it doesn't make sense to include the ones without spaces in the results
		############################################################# we exclude them here (as long as we are not using the swarm data, where these tokens are used)
		if self.model_bert_name == 'roberta' and not 'swarm' in eval_cfg.data.friendly_name:
			tokens_indices = {k : v for k, v in tokens_indices.items() if k.startswith(chr(288))}
		
		# Get the expected positions for each token in the eval data
		# all_combinations = pd.DataFrame(columns = ['sentence_type', 'token'],
		# 	data = itertools.product(*[eval_cfg.data.sentence_types, list(tokens_indices.keys())]))
		
		# cols = ['eval_data', 'exp_token', 'focus', 
		# 		'sentence_type', 'sentence_num', 'exp_logit', 
		# 		'logit', 'ratio_name', 'odds_ratio']
		
		sentences_transposed = tuple(zip(*sentences))
		orders = [[{token : 'position ' + str([w for w in strip_punct(s).split(' ') if w in self.tokens_to_mask].index(token.replace(chr(288), ''))+1) for token in tokens_indices} for s in s_tuple] for s_tuple in sentences_transposed]
		
		# summary = pd.DataFrame(columns = cols)
		summary = []
		for exp_token in tokens_indices:
			for sentence_type, label, s_tuple, order_tuple in zip(sentence_types, labels, sentences_transposed, orders):
				# token_summary = pd.DataFrame(columns = cols)
				if (indices := torch.where(label == tokens_indices[exp_token])[1]).nelement() != 0:
					# token_summary = token_summary.assign(
					# 	focus = indices,
					# 	exp_token = token,
					# 	sentence_type = sentence_type,
					# 	sentence_num = lambda df: list(range(len(df.index)))
					# )
					
					# token_summary = token_summary.merge(all_combinations, how = 'left').fillna(0)
					# logits = []
					# exp_logits = []
					# for row, idx in enumerate(token_summary['focus']):
					# 	row_sentence_num = token_summary['sentence_num'][row]
						
					# 	row_token = token_summary['token'][row]
					# 	idx_row_token = tokens_indices[row_token]
					# 	logits.append(sentence_type_logprobs[sentence_type][row_sentence_num,idx,idx_row_token])
						
					# 	exp_row_token = token_summary['exp_token'][row]
					# 	idx_exp_row_token = tokens_indices[exp_row_token]
					# 	exp_logits.append(sentence_type_logprobs[sentence_type][row_sentence_num,idx,idx_exp_row_token])
					ratio_names = []
					odds_ratios = []
					s = []
					position_nums = []
					for row, (idx, sentence, order) in enumerate(zip(indices, s_tuple, order_tuple)):
						for token in [token for token in tokens_indices if not token == exp_token]:
							row_token = token
							idx_row_token = tokens_indices[row_token]
							logit = sentence_type_logprobs[sentence_type][row,idx,idx_row_token]
							
							exp_row_token = exp_token
							idx_exp_row_token = tokens_indices[exp_row_token]
							exp_logit = sentence_type_logprobs[sentence_type][row,idx,idx_exp_row_token]
							
							ratio_names.append(exp_token + '/' + token)
							odds_ratios.append(exp_logit - logit)
							position_nums.append(order[exp_token])
							s.append(sentence)
					
					token_summary = {
						# 'focus' : indices,
						# 'exp_token' : token,
						'sentence_type' : [sentence_type for idx in indices],
						'sentence_num' : list(range(len(indices))),
						'ratio_name' : ratio_names,
						'odds_ratio' : odds_ratios,
						'role_position' : [tokens_to_roles[exp_token] for idx in indices],
						'position_num' : position_nums,
						'sentence' : s,
						# 'logit' : logits,
						# 'exp_logit' : exp_logits,
					}
					
					# token_summary = token_summary.assign(
					# 	logit = logits,
					# 	exp_logit = exp_logits,
					# 	# convert the case of the token columns to deal with uncased models; 
					# 	# otherwise we won't be able to directly
					# 	# compare them to cased models since the tokens will be different
					# 	#### actually, don't: do this later during the comparison itself. it's more accurate
					# 	# exp_token = [token.upper() for token in token_summary['exp_token']],
					# 	# exp_token = token_summary['exp_token'],
					# 	# token = [token.upper() for token in token_summary['token']],
					# 	token = token_summary['token'],
					# ).query('exp_token != token').copy().assign(
					# 	ratio_name = lambda df: df["exp_token"] + '/' + df["token"],
					# 	odds_ratio = lambda df: df['exp_logit'] - df['logit'],
					# )
					
					# summary = pd.concat([summary, token_summary], ignore_index = True)
					summary.append(token_summary)
		
		# summary['role_position'] = [tokens_to_roles[token] + ' position' for token in summary['exp_token']]
		
		# Get formatting for linear positions instead of expected tokens
		# summary = summary.sort_values(['sentence_type', 'sentence_num', 'focus'])
		# summary['position_num'] = summary.groupby(['sentence_num', 'sentence_type'])['focus'].cumcount() + 1
		# summary['position_num'] = ['position ' + str(num) for num in summary['position_num']]
		# summary = summary.sort_index()
		
		# Add the actual sentences to the summary
		# sentences_with_types = tuple(zip(*[tuple(zip(sentence_types, s_tuples)) for s_tuples in sentences]))
		
		# sentences_with_types = [
		# 	(i, *sentence) 
		# 	for s_type in sentences_with_types 
		# 		for i, sentence in enumerate(s_type)
		# ]
		
		# sentences_df = pd.DataFrame({
		# 	'sentence_num' : [t[0] for t in sentences_with_types],
		# 	'sentence_type' : [t[1] for t in sentences_with_types],
		# 	'sentence' : [t[2] for t in sentences_with_types]
		# })
		
		# summary = summary.merge(sentences_df, how = 'left')
		# summary = summary.drop(['exp_logit', 'logit', 'token', 'exp_token', 'focus'], axis = 1)
		
		# Add a unique model id to the summary as well to facilitate comparing multiple runs
		# The ID comes from the runtime of the model plus the first letter of its
		# model name to ensure that it matches when the 
		# model is evaluated on different data sets
		summary = pd.concat([pd.DataFrame(d) for d in summary])
		model_id = os.path.normpath(os.getcwd()).split(os.sep)[-2] + '-' + self.model_bert_name[0]
		summary.insert(0, 'model_id', model_id)
		summary.insert(1, 'eval_data', eval_cfg.data.friendly_name)
		
		summary = summary.assign(
			model_name = self.model_bert_name,
			masked = self.masked,
			masked_tuning_style = self.masked_tuning_style,
			tuning = self.cfg.tuning.name.replace('_', ' '),
			strip_punct = self.cfg.hyperparameters.strip_punct,
			patience = self.cfg.hyperparameters.patience,
			delta = self.cfg.hyperparameters.delta,
			epoch_criteria = eval_cfg.epoch if isinstance(eval_cfg.epoch, str) else 'manual',
			min_epochs = self.cfg.hyperparameters.min_epochs, 
			max_epochs = self.cfg.hyperparameters.max_epochs,
			random_seed = self.get_original_random_seed()
		)
		
		return summary
	
	def graph_entailed_results(self, summary: pd.DataFrame, eval_cfg: DictConfig, axis_size: int = 8, pt_size: int = 24) -> None:
		summary = summary.copy()
		
		# we do this so we can add the information to the plot labels
		acc = self.get_entailed_accuracies(summary)
		
		if len(summary.model_id.unique()) > 1:
			summary['odds_ratio'] = summary['mean']
			summary = summary.drop('mean', axis = 1)
		else:
			summary['sem'] = 0
		
		# if we are dealing with bert/distilbert and roberta models, replace the strings with uppercase ones for comparison
		if 'roberta' in summary.model_name.unique() and len(summary.model_name.unique()) > 1:
			# if we are dealing with multiple models, we want to compare them by removing the idiosyncratic variation in how
			# tokenization works. bert and distilbert are uncased, which means the tokens are converted to lower case.
			# here, we convert them back to upper case so they can be plotted in the same group as the roberta tokens,
			# which remain uppercase
			summary.loc[(summary.model_name == 'bert') | (summary.model_name == 'distilbert'), 'ratio_name'] = \
				summary[(summary.model_name == 'bert') | (summary.model_name == 'distilbert')].ratio_name.str.upper()
			
		# for roberta, strings with spaces in front of them are tokenized differently from strings without spaces
		# in front of them. so we need to remove the special characters that signals that, and add a new character
		# signifying 'not a space in front' to the appropriate cases instead
		
		# first, check whether doing this will alter information
		if 'roberta' in summary.model_name.unique():
			roberta_summary = summary[summary.model_name == 'roberta'].copy()
			num_tokens_in_summary = len(set(list(itertools.chain(*[ratio_name.split('/') for ratio_name in roberta_summary.ratio_name.unique().tolist()]))))
			
			roberta_summary['ratio_name'] = [re.sub(chr(288), '', ratio_name) for ratio_name in roberta_summary.ratio_name]
			num_tokens_after_change = len(set(list(itertools.chain(*[ratio_name.split('/') for ratio_name in roberta_summary.ratio_name.unique().tolist()]))))
			if num_tokens_in_summary != num_tokens_after_change:
				# this isn't going to actually get rid of any info, but it's worth logging
				log.warning('RoBERTa tokens were used with and without preceding spaces. This may complicate comparing results to BERT models.')
			
			# first, replace the ones that don't start with spaces before with a preceding ^
			summary.loc[(summary.model_name == 'roberta') & ~(summary.ratio_name.str.startswith(chr(288))), 'ratio_name'] = \
				summary[(summary.model_name == 'roberta') & ~(summary.ratio_name.str.startswith(chr(288)))].ratio_name.str.replace(r'((^.)|(?<=\/).)', r'^\1', regex=True)
			
			# then, replace the ones with the preceding special character (since we are mostly using them in the middle of sentences)
			summary.loc[(summary.model_name == 'roberta') & (summary.ratio_name.str.startswith(chr(288))), 'ratio_name'] = \
				summary[(summary.model_name == 'roberta') & (summary.ratio_name.str.startswith(chr(288)))].ratio_name.str.replace(chr(288), '')
		
		# Set colors for every unique odds ratio we are plotting
		all_ratios = summary.ratio_name.unique()
		colors = dict(zip(all_ratios, ['teal', 'r', 'forestgreen', 'darkorange', 'indigo', 'slategray']))
		
		# Get each unique pair of sentence types so we can create a separate plot for each pair
		sentence_types = summary.sentence_type.unique()
		sentence_types = sorted(sentence_types, key = lambda s_t: eval_cfg.data.sentence_types.index(s_t))
		paired_sentence_types = list(itertools.combinations(sentence_types, 2))
		
		# Sort so that the trained cases are first
		paired_sentence_types = [sorted(pair, key = lambda x: str(-int(x == self.reference_sentence_type)) + x) for pair in paired_sentence_types]
		
		# Filter to only cases including the reference sentence type for ease of interpretation
		paired_sentence_types = [(s1, s2) for s1, s2 in paired_sentence_types if s1 == self.reference_sentence_type] if self.reference_sentence_type != 'none' else [(s1, s2) for s1, s2 in paired_sentence_types]
		
		filename = summary.eval_data.unique()[0] + '-'
		epoch_label = summary.epoch_criteria.unique()[0] if len(summary.epoch_criteria.unique()) == 1 else ''
		if len(summary.model_id.unique()) == 1:
			epoch_label = '-' + epoch_label
			magnitude = floor(1 + np.log10(summary.total_epochs.unique()[0]))
			epoch_label = f'{str(summary.eval_epoch.unique()[0]).zfill(magnitude)}{epoch_label}'
		
		filename += epoch_label + '-odds_ratios-plots.pdf'
		
		# For each pair, we create a different plot
		# with PdfPages(f'{dataset_name}{epoch_label}-odds_ratio-plots.pdf') as pdf:
		with PdfPages(filename) as pdf:
			for pair in tqdm(paired_sentence_types, total = len(paired_sentence_types)):
				x_data = summary[summary['sentence_type'] == pair[0]].reset_index(drop = True)
				y_data = summary[summary['sentence_type'] == pair[1]].reset_index(drop = True)
							
				# Filter data to only odds ratios that exist in both sentence types
				# since we would only have one axis if a ratio exists in only one sentence type
				common_odds = set(x_data.ratio_name).intersection(y_data.ratio_name)
				x_data = x_data[x_data['ratio_name'].isin(common_odds)].reset_index(drop = True)
				y_data = y_data[y_data['ratio_name'].isin(common_odds)].reset_index(drop = True)
				
				x_odds = np.abs(x_data['odds_ratio'].values) + x_data['sem'].values
				y_odds = np.abs(y_data['odds_ratio'].values) + y_data['sem'].values
				lim = np.max([*x_odds, *y_odds]) + 1
								
				# Construct get number of linear positions (if there's only one position, we can't make plots by linear position)
				ratio_names_positions = x_data[['ratio_name', 'position_num']].drop_duplicates().reset_index(drop = True)
				ratio_names_positions = list(ratio_names_positions.to_records(index = False))
				ratio_names_positions = sorted(ratio_names_positions, key = lambda x: int(x[1].replace('position ', '')))
				
				if len(ratio_names_positions) > 1 and not all(x_data.position_num == y_data.position_num):
					fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
					fig.set_size_inches(10, 11.95)
				else:
					fig, (ax1, ax2) = plt.subplots(1, 2)
					fig.set_size_inches(10, 7.5)
				
				ax1.axis([-lim, lim, -lim, lim])
				
				# Plot data by odds ratios
				ratio_names_roles = x_data[['ratio_name', 'role_position']].drop_duplicates().reset_index(drop = True)
				ratio_names_roles = list(ratio_names_roles.to_records(index = False))
				
				for ratio_name, role in ratio_names_roles:
					x_idx = np.where(x_data.ratio_name == ratio_name)[0]
					y_idx = np.where(y_data.ratio_name == ratio_name)[0]
					
					x = x_data.odds_ratio[x_idx]
					y = y_data.odds_ratio[y_idx]
					
					color_map = x_data.loc[x_idx].ratio_name.map(colors)
					
					ax1.scatter(
						x = x, 
						y = y,
						c = color_map,
						label = f'{ratio_name} in {role.replace("_", " ")}',
						s = pt_size
					)
					
					ax1.errorbar(
						x = x, 
						xerr = x_data['sem'][x_idx],
						y = y,
						yerr = y_data['sem'][y_idx],
						ecolor = color_map,
						ls = 'none'
					)
				
				# Draw a diagonal to represent equal performance in both sentence types
				ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable = 'box')
				
				# Set labels and title
				ax1.set_xlabel(f"Confidence in {pair[0]} sentences\n", fontsize = axis_size)
				ax1.set_ylabel(f"Confidence in {pair[1]} sentences", fontsize = axis_size)
				
				ax1.plot((-lim, lim), (-lim, lim), linestyle = '--', color = 'k', scalex = False, scaley = False)
				
				ax1.legend(prop = {'size': axis_size})
				
				# Construct plot of confidence differences (a measure of transference)
				x_odds = np.abs(x_data.odds_ratio.values) + x_data['sem'].values
				y_odds = np.abs(y_data.odds_ratio - x_data.odds_ratio) + y_data['sem'].values
				ylim_diffs = np.max([*x_odds, *y_odds]) + 1
				
				ax2.axis([-lim, lim, -ylim_diffs, ylim_diffs])
				
				for ratio_name, role in ratio_names_roles:
					x_idx = np.where(x_data.ratio_name == ratio_name)[0]
					y_idx = np.where(y_data.ratio_name == ratio_name)[0]
					
					x = x_data.odds_ratio[x_idx].reset_index(drop = True)
					y = y_data.odds_ratio[y_idx].reset_index(drop = True)
					
					y = y - x
					
					color_map = x_data.loc[x_idx].ratio_name.map(colors)
					
					ax2.scatter(
						x = x, 
						y = y,
						c = color_map,
						label = f'{ratio_name} in {role.replace("_", " ")}',
						s = pt_size
					)
					
					ax2.errorbar(
						x = x, 
						xerr = x_data['sem'][x_idx],
						y = y,
						yerr = y_data['sem'][y_idx],
						ecolor = color_map,
						ls = 'none'
					)
				
				# Draw a line at zero to represent equal performance in both sentence types
				ax2.plot((-lim, lim), (0, 0), linestyle = '--', color = 'k', scalex = False, scaley = False)
				ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable = 'box')
				
				# Set labels and title
				ax2.set_xlabel(f"Confidence in {pair[0]} sentences\n", fontsize = axis_size)
				ax2.set_ylabel(f"Overconfidence in {pair[1]} sentences", fontsize = axis_size)
				
				ax2.legend(prop = {'size': axis_size})
				
				# Construct plots by linear position if they'll be different
				if len(ratio_names_positions) > 1 and not all(x_data.position_num == y_data.position_num):
					ax3.axis([-lim, lim, -lim, lim])
					
					xlabel = [f'Confidence in {pair[0]} sentences']
					ylabel = [f'Confidence in {pair[1]} sentences']
					
					# For every position in the summary, plot each odds ratio
					for ratio_name, position in ratio_names_positions:
						x_idx = np.where(x_data.position_num == position)[0]
						y_idx = np.where(y_data.position_num == position)[0]
						
						x_expected_token = x_data.loc[x_idx].ratio_name.unique()[0].split('/')[0]
						y_expected_token = y_data.loc[y_idx].ratio_name.unique()[0].split('/')[0]
						position_label = position
						#position_label = position.replace('position_', 'position ')
						
						xlabel.append(f"Expected {x_expected_token} in {position_label}")
						ylabel.append(f"Expected {y_expected_token} in {position_label}")
						
						x = x_data.odds_ratio[x_idx]
						y = y_data.odds_ratio[y_idx]
						
						# Flip the sign if the expected token isn't the same for x and y to get the correct values
						if not x_expected_token == y_expected_token:
							y = -y
						
						color_map = x_data[x_data['position_num'] == position].ratio_name.map(colors)
						
						ax3.scatter(
							x = x, 
							y = y,
							c = color_map,
							label = f'{ratio_name} in {position_label}',
							s = pt_size
						)
						
						ax3.errorbar(
							x = x, 
							xerr = x_data['sem'][x_idx],
							y = y,
							yerr = y_data['sem'][y_idx],
							ecolor = color_map,
							ls = 'none'
						)
					
					ax3.plot((-lim, lim), (-lim, lim), linestyle = '--', color = 'k', scalex = False, scaley = False)
					
					ax3.set_aspect(1.0/ax3.get_data_ratio(), adjustable = 'box')
					
					xlabel = '\n'.join(xlabel) + '\n'
					ylabel = '\n'.join(ylabel)
					
					ax3.set_xlabel(xlabel, fontsize = axis_size)
					ax3.set_ylabel(ylabel, fontsize = axis_size)
					
					ax3.legend(prop = {'size': axis_size})
					
					# Construct plot of confidence differences by linear position
					ylim_diffs = 0
					
					xlabel = [f'Confidence in {pair[0]} sentences']
					ylabel = [f'Overconfidence in {pair[1]} sentences']
					
					# For every position in the summary, plot each odds ratio
					for ratio_name, position in ratio_names_positions:
						x_idx = np.where(x_data.position_num == position)[0]
						y_idx = np.where(y_data.position_num == position)[0]
						
						x_expected_token = x_data.loc[x_idx].ratio_name.unique()[0].split('/')[0]
						y_expected_token = y_data.loc[y_idx].ratio_name.unique()[0].split('/')[0]
						position_label = position
						#position_label = position.replace('position_', 'position ')
						
						xlabel.append(f"Expected {x_expected_token} in {position_label}")
						ylabel.append(f"Expected {y_expected_token} in {position_label}")
						
						x = x_data.odds_ratio[x_idx].reset_index(drop = True)
						y = y_data.odds_ratio[y_idx].reset_index(drop = True)
						
						if not x_expected_token == y_expected_token:
							y = -y
						
						y = y - x
						
						x_odds = np.abs(x.values) + x_data['sem'][x_idx].values
						y_odds = np.abs(y.values) + y_data['sem'][y_idx].values
						ylim_diffs = np.max([ylim_diffs, np.max([*x_odds, *y_odds]) + 1])
						
						color_map = x_data[x_data['position_num'] == position].ratio_name.map(colors)
						
						ax4.scatter(
							x = x, 
							y = y,
							c = color_map,
							label = f'{ratio_name} in {position_label}',
							s = pt_size
						)
						
						ax4.errorbar(
							x = x, 
							xerr = x_data['sem'][x_idx],
							y = y,
							yerr = y_data['sem'][y_idx],
							ecolor = color_map,
							ls = 'none'
						)
					
					ax4.axis([-lim, lim, -ylim_diffs, ylim_diffs])
					ax4.plot((-lim, lim), (0, 0), linestyle = '--', color = 'k', scalex = False, scaley = False)
					
					ax4.set_aspect(1.0/ax4.get_data_ratio(), adjustable = 'box')
					
					xlabel = '\n'.join(xlabel) + '\n'
					ylabel = '\n'.join(ylabel)
					
					ax4.set_xlabel(xlabel, fontsize = axis_size)
					ax4.set_ylabel(ylabel, fontsize = axis_size)
					
					ax4.legend(prop = {'size': axis_size})
				
				# Set title
				title = re.sub(r"\'\s(.*?)", f"' {', '.join(pair)} ", eval_cfg.data.description.replace('tuples', 'pairs'))
				title += (' @ epoch ' + str(np.unique(summary.eval_epoch)[0]) + '/') if len(np.unique(summary.eval_epoch)) == 1 else ', epochs: '
				title += (str(np.unique(summary.total_epochs)[0])) if len(np.unique(summary.total_epochs)) == 1 else 'multiple'
				title += f' ({np.unique(summary.epoch_criteria)[0].replace("_", " ")})' if len(np.unique(summary.epoch_criteria)) == 1 else ' (multiple criteria)'
				title += f'\nmin epochs: {np.unique(summary.min_epochs)[0] if len(np.unique(summary.min_epochs)) == 1 else "multiple"}, '
				title += f'max epochs: {np.unique(summary.max_epochs)[0] if len(np.unique(summary.max_epochs)) == 1 else "multiple"}'
				title += f', patience: {np.unique(summary.patience)[0] if len(np.unique(summary.patience)) == 1 else "multiple"}'
				title += f' (\u0394={np.unique(summary.delta)[0] if len(np.unique(summary.delta)) == 1 else "multiple"})'
				
				model_name = np.unique(summary.model_name)[0] if len(np.unique(summary.model_name)) == 1 else 'multiple'
				masked_str = ', masking' if all(summary.masked) else ' unmasked' if all(1 - summary.masked) else ''
				masked_tuning_str = (': ' + np.unique(summary.masked_tuning_style[summary.masked])[0]) if len(np.unique(summary.masked_tuning_style[summary.masked])) == 1 else ', masking: multiple' if any(summary.masked) else ''
				subtitle = f'Model: {model_name}{masked_str}{masked_tuning_str}'
				
				tuning_data_str = np.unique(summary.tuning)[0] if len(np.unique(summary.tuning)) == 1 else 'multiple'
				subtitle += '\nTuning data: ' + tuning_data_str
				
				strip_punct_str = ' without punctuation' if all(summary.strip_punct) else " with punctuation" if all(~summary.strip_punct) else ', multiple punctuation'
				subtitle += strip_punct_str
				
				pair_acc = acc[(acc['s1'] == pair[0]) & (acc['s2'] == pair[1])]
				for arg in pair_acc.predicted_arg.unique():
					prefix = 'combined' if arg == 'any' else arg
					perc_correct_str = \
						'\n' + prefix + ' acc, Both: '    + str(round(pair_acc[pair_acc.predicted_arg == arg].both_correct.loc[0], 2)) + \
						', Neither: ' + str(round(pair_acc[pair_acc.predicted_arg == arg].both_incorrect.loc[0], 2)) + \
						', X only: '  + str(round(pair_acc[pair_acc.predicted_arg == arg].ref_correct_gen_incorrect.loc[0], 2)) + \
						', Y only: '  + str(round(pair_acc[pair_acc.predicted_arg == arg].ref_incorrect_gen_correct.loc[0], 2)) + \
						', Y|X: ' + str(round(pair_acc[pair_acc.predicted_arg == arg].gen_given_ref.loc[0], 2)) + \
						', MSE: ' + str(round(pair_acc[pair_acc.predicted_arg == arg]['specificity_(MSE)'].loc[0], 2)) + ' (\u00B1' + str(round(pair_acc[pair_acc.predicted_arg == arg].specificity_se.loc[0], 2)) + ')'
					subtitle += perc_correct_str
					
				subtitle += '\n\nX: ' + x_data[x_data.sentence_num == 0].sentence.values[0]
				subtitle += '\nY: ' + y_data[y_data.sentence_num == 0].sentence.values[0]
				
				fig.suptitle(title + '\n' + subtitle)
				
				fig.tight_layout()
				pdf.savefig()
				# plt.savefig(f"{dataset_name}-{pair[0]}-{pair[1]}-paired.pdf")
				plt.close('all')
				del fig
	
	def get_entailed_accuracies(self, summary: pd.DataFrame) -> pd.DataFrame:
		summary = summary.copy()
		
		if len(summary.model_id.unique()) > 1:
			summary['odds_ratio'] = summary['mean']
			summary = summary.drop('mean', axis = 1)
		else:
			summary['sem'] = 0
		
		# if we are dealing with bert/distilbert and roberta models, replace the strings with uppercase ones for comparison
		if 'roberta' in summary.model_name.unique() and len(summary.model_name.unique()) > 1:
			# if we are dealing with multiple models, we want to compare them by removing the idiosyncratic variation in how
			# tokenization works. bert and distilbert are uncased, which means the tokens are converted to lower case.
			# here, we convert them back to upper case so they can be plotted in the same group as the roberta tokens,
			# which remain uppercase
			summary.loc[(summary.model_name == 'bert') | (summary.model_name == 'distilbert'), 'ratio_name'] = \
				summary[(summary.model_name == 'bert') | (summary.model_name == 'distilbert')].ratio_name.str.upper()
			
		# for roberta, strings with spaces in front of them are tokenized differently from strings without spaces
		# in front of them. so we need to remove the special characters that signals that, and add a new character
		# signifying 'not a space in front' to the appropriate cases instead
		
		# first, check whether doing this will alter information
		if 'roberta' in summary.model_name.unique():
			roberta_summary = summary[summary.model_name == 'roberta'].copy()
			num_tokens_in_summary = len(set(list(itertools.chain(*[ratio_name.split('/') for ratio_name in roberta_summary.ratio_name.unique().tolist()]))))
			
			roberta_summary['ratio_name'] = [re.sub(chr(288), '', ratio_name) for ratio_name in roberta_summary.ratio_name]
			num_tokens_after_change = len(set(list(itertools.chain(*[ratio_name.split('/') for ratio_name in roberta_summary.ratio_name.unique().tolist()]))))
			if num_tokens_in_summary != num_tokens_after_change:
				# this isn't going to actually get rid of any info, but it's worth logging
				log.warning('RoBERTa tokens were used with and without preceding spaces. This may complicate comparing results to BERT models.')
			
			# first, replace the ones that don't start with spaces before with a preceding ^
			summary.loc[(summary.model_name == 'roberta') & ~(summary.ratio_name.str.startswith(chr(288))), 'ratio_name'] = \
				summary[(summary.model_name == 'roberta') & ~(summary.ratio_name.str.startswith(chr(288)))].ratio_name.str.replace(r'((^.)|(?<=\/).)', r'^\1', regex=True)
			
			# then, replace the ones with the preceding special character (since we are mostly using them in the middle of sentences)
			summary.loc[(summary.model_name == 'roberta') & (summary.ratio_name.str.startswith(chr(288))), 'ratio_name'] = \
				summary[(summary.model_name == 'roberta') & (summary.ratio_name.str.startswith(chr(288)))].ratio_name.str.replace(chr(288), '')
		
		sentence_types = summary['sentence_type'].unique()
		paired_sentence_types = list(itertools.combinations(sentence_types, 2))
		
		# Sort so that the trained cases are first
		paired_sentence_types = [
			sorted(pair, 
				   key = lambda x: str(-int(x == self.reference_sentence_type)) + x) 
			for pair in paired_sentence_types
		]
		
		# Filter to only cases including the reference sentence type for ease of interpretation
		paired_sentence_types = [(s1, s2) for s1, s2 in paired_sentence_types if s1 == self.reference_sentence_type]
		
		acc_columns = ['s1', 's2', 'predicted_arg', 'predicted_role', 'position_num_ref', 'position_num_gen', 'gen_given_ref', \
					   'both_correct', 'ref_correct_gen_incorrect', 'both_incorrect', 'ref_incorrect_gen_correct',\
					   'ref_correct', 'ref_incorrect', 'gen_correct', 'gen_incorrect', 'num_points', 'specificity_(MSE)', 'specificity_se', 
					   # 'specificity_(z)', 'specificity_se(z)',
					   's1_ex', 's2_ex']
		acc = pd.DataFrame(columns = acc_columns)
		
		for pair in paired_sentence_types:
			x_data = summary[summary['sentence_type'] == pair[0]].reset_index(drop = True)
			y_data = summary[summary['sentence_type'] == pair[1]].reset_index(drop = True)
			
			# Filter data to only odds ratios that exist in both sentence types
			common_odds = set(x_data.ratio_name).intersection(y_data.ratio_name)
			x_data = x_data[x_data['ratio_name'].isin(common_odds)].reset_index(drop = True)
			y_data = y_data[y_data['ratio_name'].isin(common_odds)].reset_index(drop = True)
			
			refs_correct = x_data.odds_ratio > 0
			gens_correct = y_data.odds_ratio > 0
			num_points = len(x_data.index)
			
			# Get the number of points in each quadrant
			gen_given_ref = (sum(y_data[y_data.index.isin(x_data.loc[x_data.odds_ratio > 0].index)].odds_ratio > 0)/len(x_data.loc[x_data.odds_ratio > 0].index) * 100) if len(x_data.loc[x_data.odds_ratio > 0].index) > 0 else np.nan
			both_correct = sum(refs_correct * gens_correct)/num_points * 100
			ref_correct_gen_incorrect = sum(refs_correct * -gens_correct)/num_points * 100
			both_incorrect = sum(-refs_correct * -gens_correct)/num_points * 100
			ref_incorrect_gen_correct = sum(-refs_correct * gens_correct)/num_points * 100
			ref_correct = sum(refs_correct)/num_points * 100
			ref_incorrect = sum(-refs_correct)/num_points * 100
			gen_correct = sum(gens_correct)/num_points * 100
			gen_incorrect = sum(-gens_correct)/num_points * 100
			
			specificity = np.mean((y_data.odds_ratio - x_data.odds_ratio)**2)
			spec_sem = np.std((y_data.odds_ratio - x_data.odds_ratio)**2)/np.sqrt(np.size((y_data.odds_ratio - x_data.odds_ratio)**2))
			
			# specificity_z = np.mean(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2)
			# specificity_z_sem = np.std(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2/np.sqrt(np.size(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2)))
			
			s1_ex = x_data[x_data.sentence_num == 0].sentence.values[0]
			s2_ex = y_data[y_data.sentence_num == 0].sentence.values[0]
			
			acc = pd.concat([acc, pd.DataFrame(
				[[pair[0], pair[1], 'any', 'any',
				  x_data.position_num.unique()[0] if len(x_data.position_num.unique()) == 1 else 'multiple',
				  y_data.position_num.unique()[0] if len(y_data.position_num.unique()) == 1 else 'multiple',
				  gen_given_ref,
				  both_correct, ref_correct_gen_incorrect, 
				  both_incorrect, ref_incorrect_gen_correct, 
				  ref_correct, ref_incorrect, 
				  gen_correct, gen_incorrect, 
				  num_points, specificity, spec_sem,
				  # specificity_z, specificity_z_sem,
				  s1_ex, s2_ex]],
				  columns = acc_columns
			)])
			
			for name, x_group in x_data.groupby('ratio_name'):
				arg = name.split('/')[0]
				y_group = y_data[y_data.ratio_name == name]
				
				refs_correct = x_group.odds_ratio > 0
				gens_correct = y_group.odds_ratio > 0
				num_points = len(x_group.index)
				
				# Get the number of points in each quadrant
				gen_given_ref = (sum(y_group[y_group.index.isin(x_group.loc[x_group.odds_ratio > 0].index)].odds_ratio > 0)/len(x_group.loc[x_group.odds_ratio > 0].index) * 100) if len(x_group.loc[x_group.odds_ratio > 0].index) > 0 else np.nan
				both_correct = sum(refs_correct * gens_correct)/num_points * 100
				ref_correct_gen_incorrect = sum(refs_correct * -gens_correct)/num_points * 100
				both_incorrect = sum(-refs_correct * -gens_correct)/num_points * 100
				ref_incorrect_gen_correct = sum(-refs_correct * gens_correct)/num_points * 100
				ref_correct = sum(refs_correct)/num_points * 100
				ref_incorrect = sum(-refs_correct)/num_points * 100
				gen_correct = sum(gens_correct)/num_points * 100
				gen_incorrect = sum(-gens_correct)/num_points * 100
				
				specificity = np.mean((y_group.odds_ratio - x_group.odds_ratio)**2)
				spec_sem = np.std((y_group.odds_ratio - x_group.odds_ratio)**2)/np.sqrt(np.size((y_group.odds_ratio - x_group.odds_ratio)**2))
				
				# specificity_z = np.mean(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2)
				# specificity_z_sem = np.std(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2/np.sqrt(np.size(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2)))
				
				acc = pd.concat([acc, pd.DataFrame(
					[[pair[0], pair[1], arg, x_group.role_position.unique()[0].split()[0],
					  x_group.position_num.unique()[0] if len(x_group.position_num.unique()) == 1 else 'multiple',
					  y_group.position_num.unique()[0] if len(y_group.position_num.unique()) == 1 else 'multiple',
					  gen_given_ref, 
					  both_correct, ref_correct_gen_incorrect, 
					  both_incorrect, ref_incorrect_gen_correct, 
					  ref_correct, ref_incorrect, 
					  gen_correct, gen_incorrect, 
					  num_points, specificity, spec_sem,
					  # specificity_z, specificity_z_sem,
					  s1_ex, s2_ex]],
					  columns = acc_columns
				)])
		
		acc = acc.assign(
			eval_epoch = np.unique(summary.eval_epoch)[0] if len(np.unique(summary.eval_epoch)) == 1 else 'multiple',
			total_epochs = np.unique(summary.total_epochs)[0] if len(np.unique(summary.total_epochs)) == 1 else 'multiple',
			min_epochs= np.unique(summary.min_epochs)[0] if len(np.unique(summary.min_epochs)) == 1 else 'multiple',
			max_epochs= np.unique(summary.max_epochs)[0] if len(np.unique(summary.max_epochs)) == 1 else 'multiple',
			model_id = np.unique(summary.model_id)[0] if len(np.unique(summary.model_id)) == 1 else 'multiple',
			eval_data = np.unique(summary.eval_data)[0] if len(np.unique(summary.eval_data)) == 1 else 'multiple',
			model_name = np.unique(summary.model_name)[0] if len(np.unique(summary.model_name)) == 1 else 'multiple',
			tuning = np.unique(summary.tuning)[0] if len(np.unique(summary.tuning)) == 1 else 'multiple',
			masked = np.unique(summary.masked)[0] if len(np.unique(summary.masked)) == 1 else 'multiple',
			masked_tuning_style = np.unique(summary.masked_tuning_style)[0] if len(np.unique(summary.masked_tuning_style)) == 1 else 'multiple',
			strip_punct = np.unique(summary.strip_punct)[0] if len(np.unique(summary.strip_punct)) == 1 else 'multiple',
			patience = np.unique(summary.patience)[0] if len(np.unique(summary.patience)) == 1 else 'multiple',
			delta = np.unique(summary.delta)[0] if len(np.unique(summary.delta)) == 1 else 'multiple',
			epoch_criteria = np.unique(summary.epoch_criteria)[0] if len(np.unique(summary.epoch_criteria)) == 1 else 'multiple',
			random_seed = np.unique(summary.random_seed)[0] if len(np.unique(summary.random_seed)) == 1 else 'multiple',
		)
		
		return acc
	
	
	def eval_newverb(self, eval_cfg: DictConfig, checkpoint_dir: str) -> None:
		"""
		Computes model performance on data with new verbs
		where this is determined as the difference in the odds ratios of
		each argument in the correct vs. incorrect position pre- and post-fine-tuning.
		"""
		self.model.eval()
		
		if self.cfg.tuning.which_args == 'model':
			with open_dict(self.cfg):
				self.cfg.tuning.args = self.cfg.tuning[self.cfg.model.friendly_name + '_args']
		else:
			with open_dict(self.cfg):
				self.cfg.tuning.args = self.cfg.tuning[self.cfg.tuning.which_args]
		
		data = self.load_eval_verb_file(eval_cfg)
		
		epoch_label = ('-' + eval_cfg.epoch) if isinstance(eval_cfg.epoch, str) else '-manual'
		epoch, total_epochs = self.restore_weights(checkpoint_dir, eval_cfg.epoch)
		magnitude = floor(1 + np.log10(total_epochs))
		
		dataset_name = eval_cfg.data.friendly_name
		epoch_label = f'{str(epoch).zfill(magnitude)}{epoch_label}'
		most_similar_tokens = self.most_similar_tokens(k=eval_cfg.k).assign(eval_epoch=epoch, total_epochs=total_epochs)
		# most_similar_tokens = pd.concat([most_similar_tokens, self.most_similar_tokens(targets=eval_cfg.data.masked_token_targets).assign(eval_epoch=epoch, total_epochs=total_epochs)], ignore_index=True)
		
		predicted_roles = {(v.lower() if 'uncased' in self.string_id else v) : k for k, v in eval_cfg.data.eval_groups.items()}
		# target_group_labels = {(k.lower() if 'uncased' in self.string_id else k) : v for k, v in eval_cfg.data.masked_token_target_labels.items()}
		
		most_similar_tokens = most_similar_tokens.assign(
			predicted_role=[predicted_roles[arg.replace(chr(288), '')] for arg in most_similar_tokens['predicted_arg']],
			# target_group_label=[target_group_labels[group.replace(chr(288), '')] if not group.endswith('most similar') and group.replace(chr(288), '') in target_group_labels else group for group in most_similar_tokens.target_group],
			eval_data=eval_cfg.data.friendly_name,
			patience=self.cfg.hyperparameters.patience,
			delta=self.cfg.hyperparameters.delta,
			min_epochs=self.cfg.hyperparameters.min_epochs,
			max_epochs=self.cfg.hyperparameters.max_epochs,
			epoch_criteria=eval_cfg.epoch if isinstance(eval_cfg.epoch, str) else 'manual',
			random_seed=self.get_original_random_seed()
		)
		
		most_similar_tokens.to_csv(f'{dataset_name}-{epoch_label}-cossims.csv.gz', index=False)
		
		# Currently we have only one new verb 
		# token and no targets to compare to,
		# meaning there is nothing to plot
		# log.info('Creating cosine similarity plots')
		# self.plot_cossims(most_similar_tokens)
		
		# Define a local function to get the odds ratios
		def get_odds_ratios(epoch: int, eval_cfg: DictConfig) -> List[Dict]:
			epoch, total_epochs = self.restore_weights(checkpoint_dir, epoch)
			
			which_args = self.cfg.tuning.which_args if not self.cfg.tuning.which_args == 'model' else self.cfg.model_friendly_name + '_args'
			
			args = self.cfg.tuning.args
			if 'added_args' in eval_cfg.data:
				if which_args in eval_cfg.data.added_args:
					args = {arg_type : args[arg_type] + eval_cfg.data.added_args[which_args][arg_type] for arg_type in args}
			
			log.info(f'Evaluating model on testing data')
			odds_ratios = []
			for sentence_type in data:
				with torch.no_grad():
					sentence_type_outputs = self.model(**data[sentence_type]['inputs'])
				
				sentence_type_logprobs = nn.functional.log_softmax(sentence_type_outputs.logits, dim=-1)
				
				for arg_type in args:
					for arg in args[arg_type]:
						for sentence_num, (arg_indices, sentence, logprob) in enumerate(zip(data[sentence_type]['sentence_arg_indices'], data[sentence_type]['sentences'], sentence_type_logprobs)):
							arg_name = chr(288) + arg if self.model_bert_name == 'roberta' and not sentence.startswith(arg_type) else arg
							arg_token_id = self.tokenizer.convert_tokens_to_ids(arg_name)
							if arg_token_id == self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token):
								raise ValueError(f'Argument {arg_name} was not tokenized correctly! Try using a different one instead.')
							
							positions = sorted(list(arg_indices.keys()), key = lambda arg_type: arg_indices[arg_type])
							positions = {p : positions.index(p) + 1 for p in positions}	
							
							for arg_position, arg_index in [(arg_position, arg_index) for arg_position, arg_index in arg_indices.items() if not arg_position == arg_type]:
								log_odds = logprob[arg_index,arg_token_id]
								exp_log_odds = logprob[arg_indices[arg_type],arg_token_id]
								odds_ratio = exp_log_odds - log_odds
								
								prediction_row = {
									'odds_ratio' : odds_ratio,
									'gf_ratio_name' : arg_type + '/' + arg_position,
									'position_ratio_name' : f'position {positions[arg_type]}/position {positions[arg_position]}',
									'token_type' : 'tuning' if arg in self.cfg.tuning.args[arg_type] else 'eval_only',
									'token_id' : arg_token_id,
									'token' : arg_name,
									'sentence' : sentence,
									'sentence_type' : sentence_type,
									'sentence_num' : sentence_num,
									'eval_epoch' : epoch,
									'total_epochs' : total_epochs
								}
								
								odds_ratios.append(prediction_row)
			
			return odds_ratios
		
		results = get_odds_ratios(epoch=0, eval_cfg=eval_cfg) + get_odds_ratios(epoch=epoch, eval_cfg=eval_cfg)
		
		summary = self.get_newverb_summary(results, eval_cfg)
		
		# Save the summary
		log.info(f"SAVING TO: {os.getcwd()}")
		summary.to_pickle(f"{dataset_name}-{epoch_label}-odds_ratios.pkl.gz")
		
		summary_csv = summary.copy()
		summary_csv.odds_ratio = summary_csv.odds_ratio.astype(float)
		summary_csv.odds_ratio_pre_post_difference = summary_csv.odds_ratio_pre_post_difference.astype(float)
		summary_csv.to_csv(f"{dataset_name}-{epoch_label}-odds_ratios.csv.gz", index = False, na_rep = 'NaN')
		
		# Create graphs
		log.info(f'Creating t-SNE plot(s)')
		self.plot_save_tsnes(summary, eval_cfg)
		
		log.info('Creating odds ratios differences plots')
		self.graph_newverb_results(summary, eval_cfg)
		
		acc = self.get_newverb_accuracies(summary)
		acc.to_csv(f'{dataset_name}-{epoch_label}-accuracies.csv.gz', index = False, na_rep = 'NaN')
		
		log.info('Evaluation complete')
		print('')
	
	def load_eval_verb_file(self, eval_cfg: DictConfig) -> Dict[str,Dict]:	
		resolved_path = os.path.join(hydra.utils.get_original_cwd(), 'data', eval_cfg.data.name)
		
		with open(resolved_path, "r") as f:
			raw_input = [line.strip() for line in f]
			raw_input = [r.lower() for r in raw_input] if 'uncased' in self.string_id else raw_input
		
		if self.cfg.hyperparameters.strip_punct:
			raw_input = [strip_punct(line) for line in raw_input]
		
		sentences = [[s.strip() for s in r.split(' , ')] for r in raw_input]
		
		transposed_sentences = list(map(list, zip(*sentences)))
		
		if not len(eval_cfg.data.sentence_types) == len(transposed_sentences):
			raise ValueError('Number of sentence types does not match in data config and data!')
		
		types_sentences = {}
		for i, sentence_type_group in enumerate(transposed_sentences):
			sentence_type = eval_cfg.data.sentence_types[i]
			types_sentences[sentence_type] = {}
			types_sentences[sentence_type]['sentences'] = []
			for sentence in sentence_type_group:
				for arg_type in self.cfg.tuning.args:
					sentence = sentence.replace(arg_type, self.tokenizer.mask_token)
				
				types_sentences[sentence_type]['sentences'].append(sentence)
			
			inputs = self.tokenizer(types_sentences[sentence_type]['sentences'], return_tensors='pt', padding=True)
			types_sentences[sentence_type]['inputs'] = inputs
			
			# get the order of the arguments so that we know which mask token corresponds to which argument type
			args_in_order = [[word for word in strip_punct(sentence).split(' ') if word in self.cfg.tuning.args] for sentence in sentence_type_group]
			masked_token_indices = [[index for index, token_id in enumerate(i) if token_id == self.mask_tok_id] for i in inputs['input_ids']]
			types_sentences[sentence_type]['sentence_arg_indices'] = [dict(zip(arg, index)) for arg, index in zip(args_in_order, masked_token_indices)]
		
		return types_sentences
	
	def get_newverb_summary(self, results: Dict, eval_cfg: DictConfig) -> pd.DataFrame:
		"""
		Convert the pre- and post-tuning results into a pandas.DataFrame
		that contains the pre- and post-fine-tuning difference in the odds ratios
		for each sentence/argument
		"""
		summary_zero = pd.concat([pd.DataFrame(d, index=[0]) for d in results if d['eval_epoch'] == 0]).reset_index(drop=True)
		summary_eval = pd.concat([pd.DataFrame(d, index=[0]) for d in results if d['eval_epoch'] != 0])
		
		if not summary_eval.empty:
			summary_eval = summary_eval.reset_index(drop=True)
			if not all(summary_eval[[c for c in summary_eval.columns if not c in ['odds_ratio', 'eval_epoch']]] == summary_zero[[c for c in summary_zero.columns if not c in ['odds_ratio', 'eval_epoch']]]):
				raise ValueError('Pre- and post-fine-tuning results do not match!')
				
			summary_eval['odds_ratio_pre_post_difference'] = summary_eval.odds_ratio - summary_zero.odds_ratio
			summary = summary_eval
		else:
			summary_zero['odds_ratio_pre_post_difference'] = np.nan
			summary = summary_zero
		
		model_id = os.path.normpath(os.getcwd()).split(os.sep)[-2] + '-' + self.model_bert_name[0]
		summary.insert(0, 'model_id', model_id)
		summary.insert(1, 'eval_data', eval_cfg.data.friendly_name)
		
		summary = summary.assign(
			model_name = self.model_bert_name,
			masked = self.masked,
			masked_tuning_style = self.masked_tuning_style,
			tuning = self.cfg.tuning.name.replace('_', ' '),
			strip_punct = self.cfg.hyperparameters.strip_punct,
			patience = self.cfg.hyperparameters.patience,
			delta = self.cfg.hyperparameters.delta,
			epoch_criteria = eval_cfg.epoch if isinstance(eval_cfg.epoch, str) else 'manual',
			min_epochs = self.cfg.hyperparameters.min_epochs, 
			max_epochs = self.cfg.hyperparameters.max_epochs,
			random_seed = self.get_original_random_seed()
		)
		
		return summary
	
	def graph_newverb_results(self, summary: pd.DataFrame, eval_cfg: DictConfig, axis_size: int = 8, pt_size: int = 24) -> None:
		# do this so we don't change any original info
		summary = summary.copy()
		
		# first, check whether doing this will alter information
		if 'roberta' in summary.model_name.unique() and 'token' in summary.columns:
			roberta_summary = summary[summary.model_name == 'roberta'].copy()
			num_tokens_in_summary = len(roberta_summary.token.unique())
			
			roberta_summary.token = [re.sub(chr(288), '', token) for token in roberta_summary.token]
			num_tokens_after_change = len(roberta_summary.token.unique())
			if num_tokens_in_summary != num_tokens_after_change:
				# this isn't going to actually get rid of any info, but it's worth logging
				log.warning('RoBERTa tokens were used with and without preceding spaces. This may complicate comparing results to BERT models.')
			
			# first, replace the ones that don't start with spaces before with a preceding ^
			summary.loc[(summary.model_name == 'roberta') & ~(summary.token.str.startswith(chr(288))), 'token'] = \
				summary[(summary.model_name == 'roberta') & ~(summary.token.str.startswith(chr(288)))].token.str.replace(r'((^.)|(?<=\/).)', r'^\1', regex=True)
			
			# then, replace the ones with the preceding special character (since we are mostly using them in the middle of sentences)
			summary.loc[(summary.model_name == 'roberta') & (summary.token.str.startswith(chr(288))), 'token'] = \
				summary[(summary.model_name == 'roberta') & (summary.token.str.startswith(chr(288)))].token.str.replace(chr(288), '')
		
		# Sort by grammatical function prominence (we need to do this because 'subj' alphabetically follows 'obj')
		summary['gf_ratio_order'] = [GF_ORDER.index(gf_ratio_name.split('/')[0]) for gf_ratio_name in summary.gf_ratio_name]
		summary = summary.sort_values(['model_id', 'gf_ratio_order'])
		summary = summary.drop('gf_ratio_order', axis=1)
		
		# Set colors for every unique odds ratio we are plotting
		all_ratios = [re.sub(r'\[|\]', '', gf_ratio_name) for gf_ratio_name in summary.gf_ratio_name.unique()]
		colors = dict(zip(all_ratios, ['teal', 'r', 'forestgreen', 'darkorange', 'indigo', 'slategray']))
		
		# we do this so we can add the information to the plot labels
		acc = self.get_newverb_accuracies(summary)
		
		if len(summary.model_id.unique()) > 1:
			summary['odds_ratio_pre_post_difference'] = summary.odds_ratio_pre_post_difference_mean
			summary['sem'] = summary.odds_ratio_pre_post_difference_sem
			summary = summary.drop('odds_ratio_pre_post_difference_mean', axis = 1)
			summary = summary.drop('odds_ratio_pre_post_difference_sem', axis = 1)
		else:
			summary['sem'] = 0
		
		# This occurs if we are evaluating on epoch 0
		# so we just plot the pairs of the odds ratios across sentence types rather than the differences
		if all(summary.odds_ratio_pre_post_difference.isnull()):
			summary.odds_ratio_pre_post_difference = summary.odds_ratio
			ax_label = 'Confidence'
		else:
			ax_label = 'Improvement'
		
		# Get each unique pair of sentence types so we can create a separate plot for each pair
		sentence_types = summary.sentence_type.unique()
		sentence_types = sorted(sentence_types, key = lambda s_t: eval_cfg.data.sentence_types.index(s_t))
		paired_sentence_types = list(itertools.combinations(sentence_types, 2))
		
		# Sort so that the trained cases are first
		paired_sentence_types = [sorted(pair, key = lambda x: str(-int(x == self.reference_sentence_type)) + x) for pair in paired_sentence_types]
		
		# Filter to only cases including the reference sentence type for ease of interpretation
		paired_sentence_types = [(s1, s2) for s1, s2 in paired_sentence_types if s1 == self.reference_sentence_type] if self.reference_sentence_type != 'none' else [(s1, s2) for s1, s2 in paired_sentence_types]
		
		filename = summary.eval_data.unique()[0] + '-'
		epoch_label = summary.epoch_criteria.unique()[0] if len(summary.epoch_criteria.unique()) == 1 else ''
		if len(summary.model_id.unique()) == 1:
			epoch_label = '-' + epoch_label
			magnitude = floor(1 + np.log10(summary.total_epochs.unique()[0]))
			epoch_label = f'{str(summary.eval_epoch.unique()[0]).zfill(magnitude)}{epoch_label}'
		
		filename += epoch_label + '-odds_ratios_diffs-plots.pdf'
		
		# For each pair, we create a different plot
		# with PdfPages(f'{dataset_name}{epoch_label}-odds_ratios_diffs-plots.pdf') as pdf:
		with PdfPages(filename) as pdf:
			for pair in tqdm(paired_sentence_types, total = len(paired_sentence_types)):
				x_data = summary[summary.sentence_type == pair[0]].reset_index(drop = True)
				y_data = summary[summary.sentence_type == pair[1]].reset_index(drop = True)
							
				# Filter data to only odds ratios that exist in both sentence types
				# since we would only have one axis if a ratio exists in only one sentence type
				# common_odds = set(x_data.ratio_name).intersection(y_data.ratio_name)
				# x_data = x_data[x_data['ratio_name'].isin(common_odds)].reset_index(drop = True)
				# y_data = y_data[y_data['ratio_name'].isin(common_odds)].reset_index(drop = True)
				
				x_odds = np.abs(x_data.odds_ratio_pre_post_difference.values) + x_data['sem'].values
				y_odds = np.abs(y_data.odds_ratio_pre_post_difference.values) + y_data['sem'].values
				lim = np.max([*x_odds, *y_odds]) + 1
								
				# Get number of linear positions (if there's only one position, we can't make plots by linear position)
				ratio_names_positions = x_data[['gf_ratio_name', 'position_ratio_name']].drop_duplicates().reset_index(drop = True)
				ratio_names_positions = list(ratio_names_positions.to_records(index = False))
				ratio_names_positions = sorted(ratio_names_positions, key = lambda x: int(re.sub('position ([0-9]+)/.*', '\\1', x[1])))
				
				if len(ratio_names_positions) > 1 and not all(x_data.position_ratio_name == y_data.position_ratio_name):
					fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
					fig.set_size_inches(11, 12.95)
				else:
					fig, (ax1, ax2) = plt.subplots(1, 2)
					fig.set_size_inches(11, 8.5)
			
				ax1.axis([-lim, lim, -lim, lim])
				
				x_data.gf_ratio_name = [re.sub(r'\[|\]', '', gf_ratio_name) for gf_ratio_name in x_data.gf_ratio_name]
				y_data.gf_ratio_name = [re.sub(r'\[|\]', '', gf_ratio_name) for gf_ratio_name in y_data.gf_ratio_name]
				
				for ratio_name in x_data.gf_ratio_name.unique():
					x_idx = np.where(x_data.gf_ratio_name == ratio_name)[0]
					y_idx = np.where(y_data.gf_ratio_name == ratio_name)[0]
					
					x = x_data.odds_ratio_pre_post_difference[x_idx]
					y = y_data.odds_ratio_pre_post_difference[y_idx]
					
					color_map = x_data.loc[x_idx].gf_ratio_name.map(colors)
					
					ax1.scatter(
						x = x, 
						y = y,
						c = color_map,
						label = ratio_name + re.sub("(.*)/.*", " position for \\1 arguments", ratio_name),
						s = pt_size
					)
					
					ax1.errorbar(
						x = x, 
						xerr = x_data['sem'][x_idx],
						y = y,
						yerr = y_data['sem'][y_idx],
						ecolor = color_map,
						ls = 'none'
					)
					
					if 'token' in x_data.columns:
						v_adjust = (ax1.get_ylim()[1] - ax1.get_ylim()[0])/100
						
						for line in x.index:
							color = ('blue' if 'token_type' in x_data.columns and x_data.loc[line].token_type == 'eval_only' else 'black')
							ax1.text(x.loc[line], y.loc[line]-v_adjust, x_data.loc[line].token.replace(chr(288), ''), size=6, horizontalalignment='center', verticalalignment='top', color=color)
				
				# Draw a diagonal to represent equal performance in both sentence types
				ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable = 'box')
				
				# Set labels and title
				ax1.set_xlabel(f"{ax_label} in {pair[0]} sentences\n", fontsize = axis_size)
				ax1.set_ylabel(f"{ax_label} in {pair[1]} sentences", fontsize = axis_size)
				
				ax1.plot((-lim, lim), (-lim, lim), linestyle = '--', color = 'k', scalex = False, scaley = False, zorder=0, alpha=0.3)
				
				ax1.legend(prop = {'size': axis_size})
				
				# Construct plot of confidence differences (a measure of transference)
				x_odds = np.abs(x_data.odds_ratio_pre_post_difference.values) + x_data['sem'].values
				y_odds = np.abs(y_data.odds_ratio_pre_post_difference - x_data.odds_ratio_pre_post_difference) + y_data['sem'].values
				ylim_diffs = np.max([*x_odds, *y_odds]) + 1
				
				ax2.axis([-lim, lim, -ylim_diffs, ylim_diffs])
				
				for ratio_name in x_data.gf_ratio_name.unique():
					x_idx = np.where(x_data.gf_ratio_name == ratio_name)[0]
					y_idx = np.where(y_data.gf_ratio_name == ratio_name)[0]
					
					x = x_data.odds_ratio_pre_post_difference[x_idx].reset_index(drop = True)
					y = y_data.odds_ratio_pre_post_difference[y_idx].reset_index(drop = True)
					
					y = y - x
					
					color_map = x_data.loc[x_idx].gf_ratio_name.map(colors)
					
					ax2.scatter(
						x = x, 
						y = y,
						c = color_map,
						label = ratio_name + re.sub("(.*)/.*", " position for \\1 arguments", ratio_name),
						s = pt_size
					)
					
					ax2.errorbar(
						x = x, 
						xerr = x_data['sem'][x_idx],
						y = y,
						yerr = y_data['sem'][y_idx],
						ecolor = color_map,
						ls = 'none'
					)
					
					if 'token' in x_data.columns:
						v_adjust = (ax2.get_ylim()[1] - ax2.get_ylim()[0])/100
						
						for line in x.index:
							color = ('blue' if 'token_type' in x_data.columns and x_data.loc[line].token_type == 'eval_only' else 'black')
							ax2.text(x.loc[line], y.loc[line]-v_adjust, x_data.loc[line].token.replace(chr(288), ''), size=6, horizontalalignment='center', verticalalignment='top', color=color)
				
				# Draw a line at zero to represent equal performance in both sentence types
				ax2.plot((-lim, lim), (0, 0), linestyle = '--', color = 'k', scalex = False, scaley = False, zorder=0, alpha=0.3)
				ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable = 'box')
				
				# Set labels and title
				ax2.set_xlabel(f"{ax_label} in {pair[0]} sentences\n", fontsize = axis_size)
				ax2.set_ylabel(f"Over{ax_label.lower()} in {pair[1]} sentences", fontsize = axis_size)
				
				ax2.legend(prop = {'size': axis_size})
				
				# Construct plots by linear position if they'll be different
				if len(ratio_names_positions) > 1 and not all(x_data.position_ratio_name == y_data.position_ratio_name):
					ax3.axis([-lim, lim, -lim, lim])
					
					xlabel = [f'{ax_label} in {pair[0]} sentences']
					ylabel = [f'{ax_label} in {pair[1]} sentences']
					
					# For every position in the summary, plot each odds ratio
					for position in x_data.position_ratio_name.unique():
						x_idx = np.where(x_data.position_ratio_name == position)[0]
						# y_idx = np.where(y_data.token.isin(x_data.loc[x_idx].token))[0]
						y_idx = x_idx
						
						expected_gf = x_data.loc[x_idx].gf_ratio_name.unique()[0].split('/')[0]
						x_position_label = position.split('/')[0]
						y_position_label = y_data.loc[y_idx].position_ratio_name.unique()[0].split('/')[0]
						
						xlabel.append(f"Expected {expected_gf} args in {x_position_label}")
						ylabel.append(f"Expected {expected_gf} args in {y_position_label}")
						
						x = x_data.odds_ratio_pre_post_difference[x_idx].reset_index(drop=True)
						y = y_data.odds_ratio_pre_post_difference[y_idx].reset_index(drop=True)
						
						# Flip the sign if the expected position isn't the same for x and y to get the correct values
						if not x_position_label == y_position_label:
							y = -y
						
						color_map = x_data[x_data.position_ratio_name == position].gf_ratio_name.map(colors)
						
						ax3.scatter(
							x = x, 
							y = y,
							c = color_map,
							label = expected_gf + ' args in ' + x_position_label,
							s = pt_size
						)
						
						ax3.errorbar(
							x = x, 
							xerr = x_data['sem'][x_idx],
							y = y,
							yerr = y_data['sem'][y_idx],
							ecolor = color_map,
							ls = 'none'
						)
						
						if 'token' in x_data.columns:
							v_adjust = (ax3.get_ylim()[1] - ax3.get_ylim()[0])/100
							
							for line in x.index:
								color = ('blue' if 'token_type' in x_data.columns and x_data.loc[line].token_type == 'eval_only' else 'black')
								ax3.text(x.loc[line], y.loc[line]-v_adjust, x_data.loc[line].token.replace(chr(288), ''), size=6, horizontalalignment='center', verticalalignment='top', color=color)
					
					ax3.plot((-lim, lim), (-lim, lim), linestyle = '--', color = 'k', scalex = False, scaley = False, zorder=0, alpha=0.3)
					
					ax3.set_aspect(1.0/ax3.get_data_ratio(), adjustable = 'box')
					
					xlabel = '\n'.join(sorted(xlabel, key=lambda item: 0 if item.startswith(ax_label) else int(re.sub(r'.*([0-9]+)$', '\\1', item)))) + '\n'
					ylabel = '\n'.join(sorted(ylabel, key=lambda item: 0 if item.startswith(ax_label) else int(re.sub(r'.*([0-9]+)$', '\\1', item))))
					
					ax3.set_xlabel(xlabel, fontsize = axis_size)
					ax3.set_ylabel(ylabel, fontsize = axis_size)
					
					ax3.legend(prop = {'size': axis_size})
					
					# Construct plot of confidence differences by linear position
					ylim_diffs = 0
					
					xlabel = [f'{ax_label} in {pair[0]} sentences']
					ylabel = [f'Over{ax_label.lower()} in {pair[1]} sentences']
					
					# For every position in the summary, plot each odds ratio
					for position in x_data.position_ratio_name.unique():
						x_idx = np.where(x_data.position_ratio_name == position)[0]
						# y_idx = np.where(y_data.token.isin(x_data.loc[x_idx].token))[0]
						y_idx = x_idx
						
						expected_gf = x_data.loc[x_idx].gf_ratio_name.unique()[0].split('/')[0]
						x_position_label = position.split('/')[0]
						y_position_label = y_data.loc[y_idx].position_ratio_name.unique()[0].split('/')[0]
						
						xlabel.append(f"Expected {expected_gf} args in {x_position_label}")
						ylabel.append(f"Expected {expected_gf} args in {y_position_label}")
						
						x = x_data.odds_ratio_pre_post_difference[x_idx].reset_index(drop=True)
						y = y_data.odds_ratio_pre_post_difference[y_idx].reset_index(drop=True)
						
						if not x_position_label == y_position_label:
							y = -y
						
						y = y - x
						
						x_odds = np.abs(x.values) + x_data['sem'][x_idx].values
						y_odds = np.abs(y.values) + y_data['sem'][y_idx].values
						ylim_diffs = np.max([ylim_diffs, np.max([*x_odds, *y_odds]) + 1])
						
						color_map = x_data[x_data.position_ratio_name == position].gf_ratio_name.map(colors)
						
						ax4.scatter(
							x = x, 
							y = y,
							c = color_map,
							label = expected_gf + ' args in ' + x_position_label,
							s = pt_size
						)
						
						ax4.errorbar(
							x = x, 
							xerr = x_data['sem'][x_idx],
							y = y,
							yerr = y_data['sem'][y_idx],
							ecolor = color_map,
							ls = 'none'
						)
						
						if 'token' in x_data.columns:
							v_adjust = (ax4.get_ylim()[1] - ax4.get_ylim()[0])/100
							
							for line in x.index:
								color = ('blue' if 'token_type' in x_data.columns and x_data.loc[line].token_type == 'eval_only' else 'black')
								ax4.text(x.loc[line], y.loc[line]-v_adjust, x_data.loc[line].token.replace(chr(288), ''), size=6, horizontalalignment='center', verticalalignment='top', color=color)
					
					ax4.axis([-lim, lim, -ylim_diffs, ylim_diffs])
					ax4.plot((-lim, lim), (0, 0), linestyle = '--', color = 'k', scalex = False, scaley = False, zorder=0, alpha=0.3)
					
					ax4.set_aspect(1.0/ax4.get_data_ratio(), adjustable = 'box')
					
					xlabel = '\n'.join(sorted(xlabel, key=lambda item: 0 if item.startswith(ax_label) else int(re.sub(r'.*([0-9]+)$', '\\1', item)))) + '\n'
					ylabel = '\n'.join(sorted(ylabel, key=lambda item: 0 if item.startswith(f'Over{ax_label.lower()}') else int(re.sub(r'.*([0-9]+)$', '\\1', item))))
					
					ax4.set_xlabel(xlabel, fontsize = axis_size)
					ax4.set_ylabel(ylabel, fontsize = axis_size)
					
					ax4.legend(prop = {'size': axis_size})
				
				# Set title
				title = re.sub(r"\'\s(.*?)", f"' {', '.join(pair)} ", eval_cfg.data.description.replace('tuples', 'pairs'))
				title += (' @ epoch ' + str(np.unique(summary.eval_epoch)[0]) + '/') if len(np.unique(summary.eval_epoch)) == 1 else ', epochs: '
				title += (str(np.unique(summary.total_epochs)[0])) if len(np.unique(summary.total_epochs)) == 1 else 'multiple'
				title += f' ({np.unique(summary.epoch_criteria)[0].replace("_", " ")})' if len(np.unique(summary.epoch_criteria)) == 1 else ' (multiple criteria)'
				title += f'\nmin epochs: {np.unique(summary.min_epochs)[0] if len(np.unique(summary.min_epochs)) == 1 else "multiple"}, '
				title += f'max epochs: {np.unique(summary.max_epochs)[0] if len(np.unique(summary.max_epochs)) == 1 else "multiple"}'
				title += f', patience: {np.unique(summary.patience)[0] if len(np.unique(summary.patience)) == 1 else "multiple"}'
				title += f' (\u0394={np.unique(summary.delta)[0] if len(np.unique(summary.delta)) == 1 else "multiple"})'
				
				model_name = np.unique(summary.model_name)[0] if len(np.unique(summary.model_name)) == 1 else 'multiple'
				masked_str = ', masking' if all(summary.masked) else ' unmasked' if all(1 - summary.masked) else ''
				masked_tuning_str = (': ' + np.unique(summary.masked_tuning_style[summary.masked])[0]) if len(np.unique(summary.masked_tuning_style[summary.masked])) == 1 else ', masking: multiple' if any(summary.masked) else ''
				subtitle = f'Model: {model_name}{masked_str}{masked_tuning_str}'
				
				tuning_data_str = np.unique(summary.tuning)[0] if len(np.unique(summary.tuning)) == 1 else 'multiple'
				subtitle += '\nTuning data: ' + tuning_data_str
				
				strip_punct_str = ' without punctuation' if all(summary.strip_punct) else " with punctuation" if all(~summary.strip_punct) else ', multiple punctuation'
				subtitle += strip_punct_str
				
				pair_acc = acc[(acc.s1 == pair[0]) & (acc.s2 == pair[1]) & (acc.token == 'any')]
				
				# Kind of hacky way to make sure subjects go above objects in the label
				pair_acc = pair_acc.sort_values('arg_type', key = lambda ser: pd.Series(sorted(ser.tolist(), key= lambda item: '0' if item == 'any' else '1' if item == 'subj' else item))).reset_index(drop=True)
				
				for arg_type, arg_group in pair_acc.groupby('arg_type', sort=False):
					arg_group = arg_group.reset_index(drop=True)
					prefix = 'combined' if arg_type == 'any' else arg_type
					perc_correct_str = \
						'\n' + prefix + ' acc, Both: '    + str(round(arg_group.both_correct[0], 2)) + \
						', Neither: ' + str(round(arg_group.both_incorrect[0], 2)) + \
						', X only: '  + str(round(arg_group.ref_correct_gen_incorrect[0], 2)) + \
						', Y only: '  + str(round(arg_group.ref_incorrect_gen_correct[0], 2)) + \
						', Y|X: ' + str(round(arg_group.gen_given_ref[0], 2)) + \
						', MSE: ' + str(round(arg_group['specificity_(MSE)'][0], 2)) + ' (\u00B1' + str(round(arg_group.specificity_se[0], 2)) + ')'
					subtitle += perc_correct_str
				
				first_x_rows = x_data[x_data.sentence == x_data.loc[0].sentence][['gf_ratio_name', 'position_ratio_name', 'sentence']].drop_duplicates().reset_index(drop=True)
				x_gf_position_map = {}
				for row in first_x_rows.index:
					x_gf_position_map.update({gf : position for gf, position in tuple(zip(first_x_rows.loc[row].gf_ratio_name.split('/'), [int(p) for p in first_x_rows.loc[row].position_ratio_name.replace('position ', '').split('/')]))})
				
				x_gf_position_map = dict(sorted(x_gf_position_map.items(), key=lambda item: item[1]))
				x_sentence_ex = first_x_rows.sentence[0]
				for gf in x_gf_position_map:
					# this is bit hacky, but it's to ensure it'll work for plots with multiple models' data
					# where we can't rely on the mask token for each model being available.
					x_sentence_ex = re.sub(r'^(.*?)(\[MASK\]|\<mask\>)', f'\\1[{gf}]', x_sentence_ex)
				
				first_y_rows = y_data[y_data.sentence == y_data.loc[0].sentence][['gf_ratio_name', 'position_ratio_name', 'sentence']].drop_duplicates().reset_index(drop=True)
				y_gf_position_map = {}
				for row in first_y_rows.index:
					y_gf_position_map.update({gf : position for gf, position in tuple(zip(first_y_rows.loc[row].gf_ratio_name.split('/'), [int(p) for p in first_y_rows.loc[row].position_ratio_name.replace('position ', '').split('/')]))})
				
				y_gf_position_map = dict(sorted(y_gf_position_map.items(), key=lambda item: item[1]))
				y_sentence_ex = first_y_rows.sentence[0]
				for gf in y_gf_position_map:
					# this is bit hacky, but it's to ensure it'll work for plots with multiple models' data
					# where we can't rely on the mask token for each model being available.
					y_sentence_ex = re.sub(r'^(.*?)(\[MASK\]|\<mask\>)', f'\\1[{gf}]', y_sentence_ex)
				
				subtitle += '\n\nX: ' + x_sentence_ex
				subtitle += '\nY: ' + y_sentence_ex
				
				fig.suptitle(title + '\n' + subtitle)
				
				fig.tight_layout()
				pdf.savefig()
				plt.close('all')
				del fig
	
	def get_newverb_accuracies(self, summary: pd.DataFrame) -> pd.DataFrame:
		summary = summary.copy()
		
		if len(summary.model_id.unique()) > 1:
			summary['odds_ratio_pre_post_difference'] = summary.odds_ratio_pre_post_difference_mean
			summary['sem'] = summary.odds_ratio_pre_post_difference_sem
			summary = summary.drop('odds_ratio_pre_post_difference_mean', axis = 1)
			summary = summary.drop('odds_ratio_pre_post_difference_sem', axis = 1)
		else:
			summary['sem'] = 0
		
		sentence_types = summary.sentence_type.unique()
		paired_sentence_types = list(itertools.combinations(sentence_types, 2))
		
		# Sort so that the trained cases are first
		paired_sentence_types = [
			sorted(pair, 
				   key = lambda x: str(-int(x == self.reference_sentence_type)) + x) 
			for pair in paired_sentence_types
		]
		
		# Filter to only cases including the reference sentence type for ease of interpretation
		paired_sentence_types = [(s1, s2) for s1, s2 in paired_sentence_types if s1 == self.reference_sentence_type]
		
		acc_columns = ['s1', 's2', 'arg_type', 'token', 'token_id', 'token_type', 'gf_ratio_name_ref', 'gf_ratio_name_gen', 'position_ratio_name_ref', 'position_ratio_name_gen', 
					   'gen_given_ref', 'both_correct', 'ref_correct_gen_incorrect', 'both_incorrect', 'ref_incorrect_gen_correct',
					   'ref_correct', 'ref_incorrect', 'gen_correct', 'gen_incorrect', 'num_points', 'specificity_(MSE)', 'specificity_se', 
					   's1_ex', 's2_ex']
		acc = pd.DataFrame(columns = acc_columns)
		
		for pair in paired_sentence_types:
			x_data = summary[summary.sentence_type == pair[0]].reset_index(drop = True)
			y_data = summary[summary.sentence_type == pair[1]].reset_index(drop = True)
			
			# Filter data to only odds ratios that exist in both sentence types
			# common_odds = set(x_data.gf_ratio_name).intersection(y_data.gf_ratio_name)
			# x_data = x_data[x_data.gf_ratio_name.isin(common_odds)].reset_index(drop = True)
			# y_data = y_data[y_data.gf_ratio_name.isin(common_odds)].reset_index(drop = True)
			
			refs_correct = x_data.odds_ratio_pre_post_difference > 0
			gens_correct = y_data.odds_ratio_pre_post_difference > 0
			num_points = len(x_data.index)
			
			# Get the number of points in each quadrant
			gen_given_ref = (sum(y_data[y_data.index.isin(x_data.loc[x_data.odds_ratio_pre_post_difference > 0].index)].odds_ratio_pre_post_difference > 0)/len(x_data.loc[x_data.odds_ratio_pre_post_difference > 0].index) * 100) if len(x_data.loc[x_data.odds_ratio_pre_post_difference > 0].index) > 0 else np.nan
			both_correct = sum(refs_correct * gens_correct)/num_points * 100
			ref_correct_gen_incorrect = sum(refs_correct * -gens_correct)/num_points * 100
			both_incorrect = sum(-refs_correct * -gens_correct)/num_points * 100
			ref_incorrect_gen_correct = sum(-refs_correct * gens_correct)/num_points * 100
			ref_correct = sum(refs_correct)/num_points * 100
			ref_incorrect = sum(-refs_correct)/num_points * 100
			gen_correct = sum(gens_correct)/num_points * 100
			gen_incorrect = sum(-gens_correct)/num_points * 100
			
			specificity = np.mean((y_data.odds_ratio_pre_post_difference - x_data.odds_ratio_pre_post_difference)**2)
			spec_sem = np.std((y_data.odds_ratio_pre_post_difference - x_data.odds_ratio_pre_post_difference)**2)/np.sqrt(np.size((y_data.odds_ratio_pre_post_difference - x_data.odds_ratio_pre_post_difference)**2))
			
			# specificity_z = np.mean(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2)
			# specificity_z_sem = np.std(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2/np.sqrt(np.size(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2)))
			
			first_x_rows = x_data[x_data.sentence == x_data.loc[0].sentence][['gf_ratio_name', 'position_ratio_name', 'sentence']].drop_duplicates().reset_index(drop=True)
			x_gf_position_map = {}
			for row in first_x_rows.index:
				x_gf_position_map.update({gf : position for gf, position in tuple(zip(first_x_rows.loc[row].gf_ratio_name.split('/'), [int(p) for p in first_x_rows.loc[row].position_ratio_name.replace('position ', '').split('/')]))})
			
			x_gf_position_map = dict(sorted(x_gf_position_map.items(), key=lambda item: item[1]))
			x_sentence_ex = first_x_rows.sentence[0]
			for gf in x_gf_position_map:
				# this is bit hacky, but it's to ensure it'll work for plots with multiple models' data
				# where we can't rely on the mask token for each model being available.
				x_sentence_ex = re.sub(r'^(.*?)(\[MASK\]|\<mask\>)', f'\\1[{gf}]', x_sentence_ex)
			
			first_y_rows = y_data[y_data.sentence == y_data.loc[0].sentence][['gf_ratio_name', 'position_ratio_name', 'sentence']].drop_duplicates().reset_index(drop=True)
			y_gf_position_map = {}
			for row in first_y_rows.index:
				y_gf_position_map.update({gf : position for gf, position in tuple(zip(first_y_rows.loc[row].gf_ratio_name.split('/'), [int(p) for p in first_y_rows.loc[row].position_ratio_name.replace('position ', '').split('/')]))})
			
			y_gf_position_map = dict(sorted(y_gf_position_map.items(), key=lambda item: item[1]))
			y_sentence_ex = first_y_rows.sentence[0]
			for gf in y_gf_position_map:
				# this is bit hacky, but it's to ensure it'll work for plots with multiple models' data
				# where we can't rely on the mask token for each model being available.
				y_sentence_ex = re.sub(r'^(.*?)(\[MASK\]|\<mask\>)', f'\\1{gf}', y_sentence_ex)
			
			s1_ex = x_sentence_ex
			s2_ex = y_sentence_ex
			
			acc = pd.concat([acc, pd.DataFrame(
				[[pair[0], pair[1], 'any', 'any', np.nan,
				  x_data.token_type.unique()[0] if len(x_data.token_type.unique()) == 1 else 'multiple',
				  x_data.gf_ratio_name.unique()[0] if len(x_data.gf_ratio_name.unique()) == 1 else 'multiple',
				  y_data.gf_ratio_name.unique()[0] if len(y_data.gf_ratio_name.unique()) == 1 else 'multiple',
				  x_data.position_ratio_name.unique()[0] if len(x_data.position_ratio_name.unique()) == 1 else 'multiple',
				  y_data.position_ratio_name.unique()[0] if len(y_data.position_ratio_name.unique()) == 1 else 'multiple',
				  gen_given_ref,
				  both_correct, ref_correct_gen_incorrect, 
				  both_incorrect, ref_incorrect_gen_correct, 
				  ref_correct, ref_incorrect, 
				  gen_correct, gen_incorrect, 
				  num_points, specificity, spec_sem,
				  # specificity_z, specificity_z_sem,
				  s1_ex, s2_ex]],
				  columns = acc_columns
			)])
			
			for name, x_group in x_data.groupby('gf_ratio_name'):
				y_group = y_data[y_data.gf_ratio_name == name]
				
				refs_correct = x_group.odds_ratio_pre_post_difference > 0
				gens_correct = y_group.odds_ratio_pre_post_difference > 0
				num_points = len(x_group.index)
				
				# Get the number of points in each quadrant
				gen_given_ref = (sum(y_group[y_group.index.isin(x_group.loc[x_group.odds_ratio_pre_post_difference > 0].index)].odds_ratio_pre_post_difference > 0)/len(x_group.loc[x_group.odds_ratio_pre_post_difference > 0].index) * 100) if len(x_group.loc[x_group.odds_ratio_pre_post_difference > 0].index) > 0 else np.nan
				both_correct = sum(refs_correct * gens_correct)/num_points * 100
				ref_correct_gen_incorrect = sum(refs_correct * -gens_correct)/num_points * 100
				both_incorrect = sum(-refs_correct * -gens_correct)/num_points * 100
				ref_incorrect_gen_correct = sum(-refs_correct * gens_correct)/num_points * 100
				ref_correct = sum(refs_correct)/num_points * 100
				ref_incorrect = sum(-refs_correct)/num_points * 100
				gen_correct = sum(gens_correct)/num_points * 100
				gen_incorrect = sum(-gens_correct)/num_points * 100
				
				specificity = np.mean((y_group.odds_ratio_pre_post_difference - x_group.odds_ratio_pre_post_difference)**2)
				spec_sem = np.std((y_group.odds_ratio_pre_post_difference - x_group.odds_ratio_pre_post_difference)**2)/np.sqrt(np.size((y_group.odds_ratio_pre_post_difference - x_group.odds_ratio_pre_post_difference)**2))
				
				# specificity_z = np.mean(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2)
				# specificity_z_sem = np.std(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2/np.sqrt(np.size(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2)))
				
				acc = pd.concat([acc, pd.DataFrame(
					[[pair[0], pair[1], re.sub(r'\[(.*)\]/.*', '\\1', name), 'any', np.nan,
					  x_data.token_type.unique()[0] if len(x_data.token_type.unique()) == 1 else 'multiple',
				  	  x_group.gf_ratio_name.unique()[0] if len(x_group.gf_ratio_name.unique()) == 1 else 'multiple',
				  	  y_group.gf_ratio_name.unique()[0] if len(y_group.gf_ratio_name.unique()) == 1 else 'multiple',
				  	  x_group.position_ratio_name.unique()[0] if len(x_group.position_ratio_name.unique()) == 1 else 'multiple',
				  	  y_group.position_ratio_name.unique()[0] if len(y_group.position_ratio_name.unique()) == 1 else 'multiple',
					  gen_given_ref, 
					  both_correct, ref_correct_gen_incorrect, 
					  both_incorrect, ref_incorrect_gen_correct, 
					  ref_correct, ref_incorrect, 
					  gen_correct, gen_incorrect, 
					  num_points, specificity, spec_sem,
					  # specificity_z, specificity_z_sem,
					  s1_ex, s2_ex]],
					  columns = acc_columns
				)])
				
				if 'token' in x_group.columns:
					for token, x_token_group in x_group.groupby('token'):
						token_id = x_token_group.token_id.unique()[0] if len(x_token_group.token_id.unique()) == 1 else 'multiple'
						token_type = x_token_group.token_type.unique()[0] if len(x_token_group.token_type.unique()) == 1 else 'multiple'
						y_token_group = y_data[y_data.token == token]
						
						refs_correct = x_token_group.odds_ratio_pre_post_difference > 0
						gens_correct = y_token_group.odds_ratio_pre_post_difference > 0
						num_points = len(x_token_group.index)
						
						# Get the number of points in each quadrant
						gen_given_ref = (sum(y_token_group[y_token_group.index.isin(x_token_group.loc[x_token_group.odds_ratio_pre_post_difference > 0].index)].odds_ratio_pre_post_difference > 0)/len(x_token_group.loc[x_token_group.odds_ratio_pre_post_difference > 0].index) * 100) if len(x_token_group.loc[x_token_group.odds_ratio_pre_post_difference > 0].index) > 0 else np.nan
						both_correct = sum(refs_correct * gens_correct)/num_points * 100
						ref_correct_gen_incorrect = sum(refs_correct * -gens_correct)/num_points * 100
						both_incorrect = sum(-refs_correct * -gens_correct)/num_points * 100
						ref_incorrect_gen_correct = sum(-refs_correct * gens_correct)/num_points * 100
						ref_correct = sum(refs_correct)/num_points * 100
						ref_incorrect = sum(-refs_correct)/num_points * 100
						gen_correct = sum(gens_correct)/num_points * 100
						gen_incorrect = sum(-gens_correct)/num_points * 100
						
						specificity = np.mean((y_token_group.odds_ratio_pre_post_difference - x_token_group.odds_ratio_pre_post_difference)**2)
						spec_sem = np.std((y_token_group.odds_ratio_pre_post_difference - x_token_group.odds_ratio_pre_post_difference)**2)/np.sqrt(np.size((y_token_group.odds_ratio_pre_post_difference - x_token_group.odds_ratio_pre_post_difference)**2))
						
						acc = pd.concat([acc, pd.DataFrame(
							[[pair[0], pair[1], re.sub(r'\[(.*)\]/.*', '\\1', name), token, token_id, token_type,
						  	  x_group.gf_ratio_name.unique()[0] if len(x_group.gf_ratio_name.unique()) == 1 else 'multiple',
						  	  y_group.gf_ratio_name.unique()[0] if len(y_group.gf_ratio_name.unique()) == 1 else 'multiple',
						  	  x_group.position_ratio_name.unique()[0] if len(x_group.position_ratio_name.unique()) == 1 else 'multiple',
						  	  y_group.position_ratio_name.unique()[0] if len(y_group.position_ratio_name.unique()) == 1 else 'multiple',
							  gen_given_ref, 
							  both_correct, ref_correct_gen_incorrect, 
							  both_incorrect, ref_incorrect_gen_correct, 
							  ref_correct, ref_incorrect, 
							  gen_correct, gen_incorrect, 
							  num_points, specificity, spec_sem,
							  s1_ex, s2_ex]],
							  columns = acc_columns
						)])				
		
		acc = acc.assign(
			eval_epoch = np.unique(summary.eval_epoch)[0] if len(np.unique(summary.eval_epoch)) == 1 else 'multiple',
			total_epochs = np.unique(summary.total_epochs)[0] if len(np.unique(summary.total_epochs)) == 1 else 'multiple',
			min_epochs= np.unique(summary.min_epochs)[0] if len(np.unique(summary.min_epochs)) == 1 else 'multiple',
			max_epochs= np.unique(summary.max_epochs)[0] if len(np.unique(summary.max_epochs)) == 1 else 'multiple',
			model_id = np.unique(summary.model_id)[0] if len(np.unique(summary.model_id)) == 1 else 'multiple',
			eval_data = np.unique(summary.eval_data)[0] if len(np.unique(summary.eval_data)) == 1 else 'multiple',
			model_name = np.unique(summary.model_name)[0] if len(np.unique(summary.model_name)) == 1 else 'multiple',
			tuning = np.unique(summary.tuning)[0] if len(np.unique(summary.tuning)) == 1 else 'multiple',
			masked = np.unique(summary.masked)[0] if len(np.unique(summary.masked)) == 1 else 'multiple',
			masked_tuning_style = np.unique(summary.masked_tuning_style)[0] if len(np.unique(summary.masked_tuning_style)) == 1 else 'multiple',
			strip_punct = np.unique(summary.strip_punct)[0] if len(np.unique(summary.strip_punct)) == 1 else 'multiple',
			patience = np.unique(summary.patience)[0] if len(np.unique(summary.patience)) == 1 else 'multiple',
			delta = np.unique(summary.delta)[0] if len(np.unique(summary.delta)) == 1 else 'multiple',
			epoch_criteria = np.unique(summary.epoch_criteria)[0] if len(np.unique(summary.epoch_criteria)) == 1 else 'multiple',
			random_seed = np.unique(summary.random_seed)[0] if len(np.unique(summary.random_seed)) == 1 else 'multiple',
		).reset_index(drop=True)
		
		return acc
	
	"""
	deprecated
		# no longer used anywhere
		def collect_entailed_results(self, inputs, eval_groups, outputs):
			
			results_arr = []
			
			for j in range(len(outputs)):
				
				results = {}
				
				logits = outputs[j].logits
				probabilities = nn.functional.softmax(logits, dim=2)
				log_probabilities = nn.functional.log_softmax(logits, dim=2)
				predicted_ids = torch.argmax(log_probabilities, dim=2)
				
				for i, _ in enumerate(predicted_ids):
				
				sentence_results = {}
				foci = torch.nonzero(inputs[j]["input_ids"][i]==self.mask_tok_id, as_tuple=True)[0]
				
				for idx in foci:
					idx_results = {}
					for group in eval_groups:
					tokens = eval_groups[group]
					group_mean = 0.0
					for token in tokens:
						token_id = self.tokenizer(token, return_tensors="pt")["input_ids"][:,1]
						group_mean += log_probabilities[:,idx,:][i,token_id].item()
					idx_results[group] = group_mean
					
					sentence_results[idx] = {
					'mean grouped log probability' : idx_results,
					'log_probabilities' : log_probabilities[:,idx,:][i,:],
					'probabilities' : probabilities[:,idx,:][i,:],
					'logits': logits[:,idx,:][i,:]
					}
				results[i] = sentence_results
				
				results_arr.append(results)
			
			return results_arr
		
		# no longer used anywhere
		def summarize_entailed_results(self, results_arr, labels_arr):
			
			# Define theme and recipient ids
			ricket = self.tokenizer(self.tokens_to_mask["RICKET"], return_tensors="pt")["input_ids"][:,1]
			thax = self.tokenizer(self.tokens_to_mask["THAX"], return_tensors="pt")["input_ids"][:,1]
			
			active_results = results_arr[0]
			active_labels = labels_arr[0]
			
			passive_results = results_arr[1]
			passive_labels = labels_arr[1]
			
			confidences = []
			
			for r in active_results:
				
				active_result = active_results[r]
				active_label = active_labels[r]
				
				passive_result = passive_results[r]
				passive_label = passive_labels[r]
				
				active_token_confidence = {}
				passive_token_confidence = {}
				
				for idx in active_result:
				
				target = active_label[idx.item()]
				scores = active_result[idx]['mean grouped log probability']
				
				token_conf = scores['theme'] - scores['recipient']
				
				if target == ricket:
					# print("I'm in a recipient position")
					active_token_confidence["recipient"] = -token_conf
				else:
					# print("I'm in a theme position")
					active_token_confidence["theme"] = token_conf
				
				for idx in passive_result:
				
				target = passive_label[idx.item()]
				scores = passive_result[idx]['mean grouped log probability']
				
				# print(scores)
				# raise SystemExit
				
				token_conf = scores['theme'] - scores['recipient']
				
				if target == ricket:
					# print("I'm in a recipient position")
					passive_token_confidence["recipient"] = -token_conf
				else:
					# print("I'm in a theme position")
					passive_token_confidence["theme"] = token_conf
				
				confidences.append({
				"active" : active_token_confidence,
				"passive" : passive_token_confidence
				})
			
			return confidences

		# no longer used anywhere
		@property
		def dev_tokens_to_mask(self) -> List[str]:
			dev_tokens_to_mask = {}
			for dataset in self.cfg.dev:
				# convert things to lowercase for uncased models
				tokens = [t.lower() for t in self.cfg.dev[dataset].to_mask] if 'uncased' in self.string_id else list(self.cfg.dev[dataset].to_mask)
				# add the versions of the tokens with preceding spaces to our targets for roberta
				if self.model_bert_name == 'roberta':
					tokens += [chr(288) + t for t in tokens]
				
				dev_tokens_to_mask.update({dataset: tokens})
			
		 	return dev_tokens_to_mask
	
		def eval_new_verb(self, eval_cfg: DictConfig, args_cfg: DictConfig, checkpoint_dir: str) -> None:
			\"""
			Computes model performance on data with new verbs
			where this is determined as the difference in the probabilities associated
			with each argument to be predicted before and after training.
			To do this, we check predictions for each arg, word pair in args_cfg on a fresh model, 
			and then check them on the fine-tuned model.
			\"""
			from transformers import pipeline
			
			data = self.load_eval_verb_file(args_cfg, eval_cfg.data.name, eval_cfg.data.to_mask)
					
			self.model.eval()
			epoch_label = ('-' + eval_cfg.epoch) if isinstance(eval_cfg.epoch, str) else '-manual'
			epoch, total_epochs = self.restore_weights(checkpoint_dir, eval_cfg.epoch)
			magnitude = floor(1 + np.log10(total_epochs))
			
			dataset_name = eval_cfg.data.friendly_name
			epoch_label = f'{str(epoch).zfill(magnitude)}{epoch_label}'
			most_similar_tokens = self.most_similar_tokens(k=eval_cfg.k).assign(eval_epoch=epoch, total_epochs=total_epochs)
			most_similar_tokens = pd.concat([most_similar_tokens, self.most_similar_tokens(targets=eval_cfg.data.masked_token_targets).assign(eval_epoch=epoch, total_epochs=total_epochs)], ignore_index=True)
			
			predicted_roles = {(v.lower() if 'uncased' in self.string_id else v) : k for k, v in eval_cfg.data.eval_groups.items()}
			target_group_labels = {(k.lower() if 'uncased' in self.string_id else k) : v for k, v in eval_cfg.data.masked_token_target_labels.items()}
			
			most_similar_tokens = most_similar_tokens.assign(
				predicted_role=[predicted_roles[arg.replace(chr(288), '')] for arg in most_similar_tokens['predicted_arg']],
				target_group_label=[target_group_labels[group.replace(chr(288), '')] if not group.endswith('most similar') and group.replace(chr(288), '') in target_group_labels else group for group in most_similar_tokens.target_group],
				eval_data=eval_cfg.data.friendly_name,
				patience=self.cfg.hyperparameters.patience,
				delta=self.cfg.hyperparameters.delta,
				min_epochs=self.cfg.hyperparameters.min_epochs,
				max_epochs=self.cfg.hyperparameters.max_epochs,
				epoch_criteria=eval_cfg.epoch if isinstance(eval_cfg.epoch, str) else 'manual',
				random_seed=self.get_original_random_seed()
			)
			
			most_similar_tokens.to_csv(f'{dataset_name}-{epoch_label}-cossim.csv.gz', index=False)
			
			# log.info('Creating cosine similarity plots')
			# self.plot_cossims(most_similar_tokens)
			
			# maybe switch this to getting the odds ratios like the previous one? YES---much faster and more comparable to the new args expts
			# Define a local function to get the probabilities
			def get_probs(epoch: int) -> Dict[int,Dict]:
				epoch, total_epochs = self.restore_weights(checkpoint_dir, epoch)
				filler = pipeline('fill-mask', model = self.model, tokenizer = self.tokenizer)
				
				log.info(f'Evaluating model @ epoch {epoch}/{total_epochs} on testing data')
				results = {'total_epochs' : total_epochs}
				for arg in data:
					results[arg] = {}
					for i, s_group in enumerate(data[arg]):
						results[arg][eval_cfg.data.sentence_types[i]] = []
						for s in s_group:
							s_dict = {}
							if self.mask_tok in s:
								s_dict['sentence'] = s
								s_dict['results'] = {}
								for arg2 in args_cfg:
									targets = args_cfg[arg2]
									if self.model_bert_name == 'roberta':
										targets = [' ' + t for t in targets]
									
									s_dict['results'][arg2] = filler(s, targets = targets)
									
									for i, d in s_dict['results'][arg2].iteritems():
										# Remove the preceding blanks so we can match them up later
										s_dict['results'][arg2][i]['token_str'] = s_dict['results'][arg2][i]['token_str'].replace(' ', '')
									
									s_dict['results'][arg2] = sorted(
										s_dict['results'][arg2],
										key = lambda x: args_cfg[arg2].index(x['token_str'])
									)
									
							results[arg][eval_cfg.data.sentence_types[i]].append(s_dict)
				
				return { epoch : results }
			
			results = {**get_probs(epoch = 0), **get_probs(epoch = epoch)}
			
			summary = self.get_new_verb_summary(results, args_cfg, eval_cfg)
			
			# Save the summary
			dataset_name = eval_cfg.data.friendly_name
			epoch = max(results.keys())
			
			log.info(f"SAVING TO: {os.getcwd()}")
			summary.to_pickle(f"{dataset_name}-0-{epoch}-surprisals.pkl.gz")
			summary.to_csv(f"{dataset_name}-0-{epoch}-surprisals.csv.gz", index = False, na_rep = 'NaN')
			
			# Create graphs
			log.info(f'Creating t-SNE plots')
			self.plot_save_tsnes(summary, eval_cfg)
			
			log.info('Creating surprisal plots')
			self.graph_new_verb_results(summary, eval_cfg)
			
			log.info('Evaluation complete')
			print('')
		
		def load_eval_verb_file(self, args_cfg: DictConfig, data_path: str, replacing: Dict[str,str]) -> Dict[str,List[str]]:
			
			resolved_path = os.path.join(hydra.utils.get_original_cwd(),"data",data_path)
			
			with open(resolved_path, "r") as f:
				raw_input = [line.strip() for line in f]
				raw_input = [r.lower() for r in raw_input] if 'uncased' in self.string_id else raw_input
			
			if self.cfg.hyperparameters.strip_punct:
				raw_input = [strip_punct(line) for line in raw_input]
			
			sentences = [[s.strip() for s in r.split(' , ')] for r in raw_input]
			
			arg_dicts = {}
			for arg in args_cfg:
				curr_dict = args_cfg.copy()
				curr_dict[arg] = [self.mask_tok]
				
				args, values = zip(*curr_dict.items())
				arg_combos = itertools.product(*list(curr_dict.values()))
				arg_combo_dicts = [dict(zip(args, t)) for t in arg_combos]
				arg_dicts[arg] = arg_combo_dicts
			
			filled_sentences = {}
			for arg in arg_dicts:
				filled_sentences[arg] = []
				for s_group in sentences:
					group = []
					for s in s_group:
						s_list = []
						s_tmp = s
						for arg_combo in arg_dicts[arg]:
							for arg2 in arg_combo:
								s = s.replace(arg2, arg_combo[arg2])
							
							s_list.append(s)
							s = s_tmp
							
						group.append(s_list)
					
					filled_sentences[arg].append(group)
			
			for arg in filled_sentences:
				filled_sentences[arg] = list(map(list, zip(*filled_sentences[arg])))
				filled_sentences[arg] = [list(itertools.chain(*sublist)) for sublist in filled_sentences[arg]]
			
			if not verify_tokenization_of_sentences(self.tokenizer, filled_sentences, self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs):
				log.warning('Tokenization of sentences was affected by the new tokens! Try choosing a new string.')
				return
			
			return filled_sentences
		
		def get_new_verb_summary(self, results: Dict, args_cfg: DictConfig, eval_cfg: DictConfig) -> pd.DataFrame:
			\"""
			Convert the pre- and post-tuning results into a pandas.DataFrame
			\"""
			
			# Define a local function to convert each set of results to a data frame
			def convert_results(results: dict, args_cfg) -> pd.DataFrame:
				summary = pd.DataFrame()
				
				for eval_epoch in results:
					total_epochs = results[eval_epoch].pop('total_epochs', 'unknown')
					for target_position in results[eval_epoch]:
						for sentence_type in results[eval_epoch][target_position]:
							for i, sentence in enumerate(results[eval_epoch][target_position][sentence_type]):
								if 'results' in sentence:
									for predicted_token_type in sentence['results']:
										for prediction in sentence['results'][predicted_token_type]:
											summary_ = pd.DataFrame()
											pred_seq = prediction['sequence']
											mask_seq = sentence['sentence']
											
											# replace the internal token(s) with the visible one
											for eval_group in eval_cfg.data.eval_groups:
												pred_seq = pred_seq.replace(
													eval_cfg.data.eval_groups[eval_group],
													eval_cfg.data.to_mask['[' + eval_group + ']']
												)
												
												# mask_seq = mask_seq.replace(
												# eval_cfg.data.eval_groups[eval_group],
												# eval_cfg.data.to_mask['[' + eval_group + ']']
												# )
											
											summary_ = summary_.assign(
												filled_sentence = [pred_seq],
												vocab_token_index = [prediction['token']],
												predicted_token = [prediction['token_str'].replace(' ', '')],
												p = [prediction['score']],
												surprisal = lambda df: -np.log2(df['p']),
												predicted_token_type = [re.sub(r'\[|\]', '', predicted_token_type)],
												masked_sentence = [mask_seq],
												sentence_type = [sentence_type],
												sentence_num = [i],
												target_position_name = [re.sub(r'\[|\]', '', target_position)],
												eval_epoch = eval_epoch,
												total_epochs = total_epochs,
									 			min_epochs = self.cfg.hyperparameters.min_epochs,
									 			max_epochs = self.cfg.hyperparameters.max_epochs
											)
											
											mask_pos = mask_seq.index(self.mask_tok)
											
											arg_positions = {}
											for arg in args_cfg:
												arg_indices = list(map(lambda x: mask_seq.index(x) if x in mask_seq else None, args_cfg[arg]))
												if any(arg_indices):
													arg_positions[arg] = [arg_index for arg_index in arg_indices if arg_index is not None]
													
											position_num = 1
											for arg in arg_positions:
												if any(list(map(lambda x: True if x < mask_pos else False, arg_positions[arg]))):
													position_num += 1
													
											target_position_num = ['position ' + str(position_num)],
											
											summary = pd.concat([summary, summary_], ignore_index = True)
				
				summary = summary.assign(
					model_id = os.path.normpath(os.getcwd()).split(os.sep)[-2],
					eval_data = eval_cfg.data.friendly_name,
					model_name = self.model_bert_name,
					masked = self.masked,
					masked_tuning_style = self.masked_tuning_style,
					tuning = self.cfg.tuning.name,
					strip_punct = self.cfg.hyperparameters.strip_punct,
					patience = self.cfg.hyperparameters.patience,
					delta = self.cfg.hyperparameters.delta,
					epoch_criteria = eval_cfg.epoch if isinstance(eval_cfg.epoch, str) else 'manual'
				)
				
				return summary
			
			summary = convert_results(results, args_cfg)
			
			# Reorder the columns
			columns = [
				'model_id', 'model_name', 'total_epochs', 'min_epochs', 'max_epochs',
				'tuning', 'strip_punct', 'masked', 'masked_tuning_style', 
				'patience', 'delta', # model properties
				'eval_epoch', 'eval_data', 'epoch_criteria' # eval properties
				'sentence_type', 'target_position_name', 'target_position_num', 
				'predicted_token_type', 'masked_sentence', 'sentence_num', # sentence properties
				'filled_sentence', 'predicted_token', 'vocab_token_index', 'surprisal', 'p' # observation properties
			]
			
			summary = summary[columns]
			
			# Sort the summary
			sort_columns = ['model_id', 'sentence_type', 'sentence_num', 'target_position_name', 'predicted_token_type', 'predicted_token', 'masked_sentence']
			summary = summary.sort_values(
				by = sort_columns, 
				ascending = [(column != 'predicted_token_type' and column != 'target_position_name') for column in sort_columns]
			).reset_index(drop = True)
			
			summary = summary.assign(random_seed=self.get_original_random_seed())
			
			return summary
		
		def graph_new_verb_results(self, summary: pd.DataFrame, eval_cfg: DictConfig, axis_size: int = 10, pt_size: int = 24) -> None:
			if len(np.unique(summary.model_id.values)) > 1:
				summary['surprisal'] = summary['mean']
				summary = summary.drop('mean', axis = 1)
			else:
				summary['sem'] = 0
			
			# Get each sentence type to compare them on pre- and post-tuning data
			sentence_types = summary['sentence_type'].unique()
			
			summary['surprisal_gf_label'] = [row['predicted_token_type'] + ' in ' + row['target_position_name'] + ' position' for _, row in summary.iterrows()]
			summary['surprisal_pos_label'] = [row['predicted_token_type'] + ' in ' + row['target_position_num'].replace('_', ' ') for _, row in summary.iterrows()]
			
			# Set colors for every unique surprisal type we are plotting
			# This is hacky; find a way to fix it
			colors1 = ['teal', 'darkturquoise', 'maroon', 'r', 'blueviolet', 'indigo']
			colors2 = ['teal', 'r']
			
			x_epoch = min(summary.eval_epoch)
			y_epoch = max(summary.eval_epoch)
			
			dataset_name = summary.eval_data.unique()[0]
			eval_epoch = '-' + str(np.unique(summary.eval_epoch)[0]) if len(np.unique(summary.eval_epoch)) == 1 else ''
			
			epoch_label = '-' + summary.epoch_criteria.unique()[0] if len(summary.epoch_criteria.unique()) == 1 else ''
			epoch_label = eval_epoch + epoch_label
			
			# For each sentence type, we create a different plot
			# with PdfPages(f'{dataset_name}{epoch_label}-surprisal-plots.pdf') as pdf:
			with PdfPages(f'{dataset_name}{epoch_label}-surprisals-plots.pdf') as pdf:
				for sentence_type in tqdm(sentence_types, total = len(sentence_types)):
					
					# Get x and y data. We plot the first member of each pair on x, and the second member on y
					x_data = summary.loc[(summary.eval_epoch == x_epoch) & (summary['sentence_type'] == sentence_type)].reset_index(drop = True)
					y_data = summary.loc[(summary.eval_epoch == y_epoch) & (summary['sentence_type'] == sentence_type)].reset_index(drop = True)
					
					lim = np.max([*[np.abs(x_data['surprisal'].values) + x_data['sem'].values], *[np.abs(y_data['surprisal'].values) + y_data['sem'].values]]) + 1
					
					# Get number of linear positions (if there's only one position, we can't make plots by linear position)
					sur_names_positions = x_data[['surprisal_gf_label', 'target_position_num']].drop_duplicates().reset_index(drop = True)
					sur_names_positions = list(sur_names_positions.to_records(index = False))
					sur_names_positions = sorted(sur_names_positions, key = lambda x: int(x[1].replace('position ', ' ')))
					
					# If there's more than one position, we'll create a
					fig, (ax1, ax2) = plt.subplots(1, 2)
					fig.set_size_inches(13.25, 6.5)
					
					ax1.set_xlim(-1, lim)
					ax1.set_ylim(-1, lim)
					
					# Plot data by surprisal labels
					surprisal_gf_labels = x_data[['surprisal_gf_label']].drop_duplicates().reset_index(drop = True)
					surprisal_gf_labels = sorted(list(itertools.chain(*surprisal_gf_labels.values.tolist())), key = lambda x: str(int(x.split(' ')[0] == x.split(' ')[2])) + x.split(' ')[2], reverse = True)
					
					x_data['Linear Order'] = [row['target_position_num'].replace('_', ' ') for _, row in x_data.iterrows()]
					x_data['Grammatical Function'] = x_data.surprisal_gf_label
					num_gfs = len(x_data['Grammatical Function'].unique().tolist())
					
					ax1 = sns.scatterplot(
						x = x_data.surprisal,
						y = y_data.surprisal,
						style = x_data['Linear Order'] if len(x_data['Linear Order'].unique().tolist()) > 1 else None,
						hue = x_data['Grammatical Function'],
						hue_order = surprisal_gf_labels,
						palette = colors1[:num_gfs] if len(x_data['Linear Order'].unique().tolist()) > 1 else colors2,
						s = pt_size,
						ax = ax1
					)
					
					ax1.errorbar(
						x = x_data.surprisal, 
						xerr = x_data['sem'],
						y = y_data.surprisal,
						yerr = y_data['sem'],
						ecolor = colors[:num_gfs],
						ls = 'none'
					)
					
					# Set labels and title
					ax1.set_xlabel(f"Surprisal @ epoch {np.unique(x_data.eval_epoch)[0]}", fontsize = axis_size)
					ax1.set_ylabel(f"Surprisal @ epoch {np.unique(y_data.eval_epoch)[0]}", fontsize = axis_size)
					
					# Draw a diagonal to represent equal performance in both sentence types
					ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable = 'box')
					ax1.plot((-1, lim), (-1, lim), linestyle = '--', color = 'k', scalex = False, scaley = False)
					
					box = ax1.get_position()
					ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
					ax1.get_legend().remove()
					
					# Construct plot of surprisal differences (a measure of transference)
					ax2.set_xlim(-1, lim)
					x_surs = np.abs(x_data.surprisal.values) + x_data['sem'].values
					y_surs = np.abs(y_data.surprisal - x_data.surprisal) + y_data['sem'].values
					ylim_diffs = np.max([*x_surs, *y_surs]) + 1
					
					ax2.set_ylim(-ylim_diffs, ylim_diffs)
					
					ax2 = sns.scatterplot(
						x = x_data.surprisal,
						y = y_data.surprisal - x_data.surprisal,
						style = x_data['Linear Order'] if len(x_data['Linear Order'].unique().tolist()) > 1 else None,
						hue = x_data['Grammatical Function'],
						hue_order = surprisal_gf_labels,
						palette = colors1[:num_gfs] if len(x_data['Linear Order'].unique().tolist()) > 1 else colors2,
						s = pt_size,
						ax = ax2
					)
					
					ax2.errorbar(
						x = x, 
						xerr = x_data['sem'],
						y = y_data.surprisal - x_data.surprisal,
						yerr = y_data['sem'],
						ecolor = colors[:num_gfs],
						ls = 'none'
					)
					
					# Set labels and title
					ax2.set_xlabel(f"Surprisal @ epoch {np.unique(x_data.eval_epoch)[0]}", fontsize = axis_size)
					ax2.set_ylabel(f"Δ surprisal ({np.unique(y_data.eval_epoch)[0]} - {np.unique(x_data.eval_epoch)[0]})", fontsize = axis_size)
					
					# Draw a line at zero to represent equal performance in both pre- and post-tuning
					ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable = 'box')
					ax2.plot((-1, lim), (0, 0), linestyle = '--', color = 'k', scalex = False, scaley = False)
					
					box = ax2.get_position()
					ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
					legend = ax2.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), fontsize = 9)
					
					handles, labels = ax2.get_legend_handles_labels()
					try:
						idx = labels.index('Linear Order')
					except ValueError:
						idx = None
						legend.set_title('Grammatical Function')
					
					if idx is not None:
						labels_handles = list(zip(labels[idx+1:], handles[idx+1:]))
						slabels, shandles = map(list, zip(*sorted(labels_handles, key = lambda x: x[0])))
					
						handles = handles[:idx+1] + shandles
						labels = labels[:idx+1] + slabels
					
						ax2.legend(handles, labels, loc = 'center left', bbox_to_anchor = (1, 0.5), fontsize = 9)
					
					# Set title
					title = f"{eval_cfg.data.description.replace(' tuples', '')} {sentence_type}s"
					
					model_name = np.unique(summary.model_name)[0] if len(np.unique(summary.model_name)) == 1 else 'multiple'
					masked_str = ', masking' if all(summary.masked) else ' unmasked' if all(1 - summary.masked) else ''
					masked_tuning_str = (': ' + np.unique(summary.masked_tuning_style[summary.masked])[0]) if len(np.unique(summary.masked_tuning_style[summary.masked])) == 1 else ', masking: multiple' if any(summary.masked) else ''
					subtitle = f'Model: {model_name}{masked_str}{masked_tuning_str}'
					subtitle += f', patience: {np.unique(summary.patience)[0] if len(np.unique(summary.patience)) == 1 else "multiple"}'
					subtitle += f' (\u0394={np.unique(summary.delta)[0] if len(np.unique(summary.delta)) == 1 else "multiple"})'
					
					tuning_data_str = np.unique(summary.tuning)[0] if len(np.unique(summary.tuning)) == 1 else 'multiple'
					subtitle += '\nTuning data: ' + tuning_data_str
					
					# Set title
					strip_punct_str = ' without punctuation' if all(summary.strip_punct) else " with punctuation" if all(~summary.strip_punct) else ', multiple punctuation'
					subtitle += strip_punct_str
					
					fig.suptitle(title + '\n' + subtitle)
					fig.tight_layout(rect=[-0.025,0.1,0.9625,1])
					pdf.savefig()
					plt.close('all')
					del fig
	"""
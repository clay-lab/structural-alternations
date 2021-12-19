# tuner.py
# 
# Tunes a model on training data and provides functions for evaluation
import os
import re
import sys
import hydra
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import logging
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import torch.nn as nn

from math import ceil, floor
from tqdm import trange, tqdm
from typing import Dict, List, Tuple, Union
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import logging as lg
from transformers import BertForMaskedLM, BertTokenizer
from transformers import RobertaForMaskedLM, RobertaTokenizer
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

from core.tuner_utils import *

lg.set_verbosity_error()

log = logging.getLogger(__name__)

class Tuner:

	# START Computed Properties
	
	@property
	def model_class(self):
		return eval(self.cfg.model.base_class) if isinstance(eval(self.cfg.model.base_class), type) else None
	
	@property
	def tokenizer_class(self):
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
	def dev_reference_sentence_type(self):
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
		to_mix = self.verb_tuning_data['data'] if self.cfg.tuning.new_verb else self.tuning_data
		
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
		to_mask = self.verb_tuning_data['data'] if self.cfg.tuning.new_verb else self.tuning_data
		
		data = []
		for s in to_mask:
			if self.cfg.hyperparameters.strip_punct:
				s = strip_punct(s)
			for tok in self.tokens_to_mask:
				s = s.replace(tok, self.mask_tok)
			
			data.append(s)
		
		return data
	
	@property
	def verb_tuning_data(self) -> Dict[str, List[str]]:
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
		return {
			'args' : to_replace,
			'data' : sentences
		}
	
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
		
		to_mix = {dataset: self.verb_dev_data[dataset]['data'] if self.cfg.dev[dataset].new_verb else self.dev_data[dataset] for dataset in self.cfg.dev}
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
	def masked_dev_data(self) -> List[str]:
		if not self.cfg.dev:
			return {}
		
		to_mask = {dataset: self.verb_dev_data[dataset]['data'] if self.cfg.dev[dataset].new_verb else self.dev_data[dataset] for dataset in self.cfg.dev}
		
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
	def verb_dev_data(self) -> Dict[str, List[str]]:
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
		Fine-tunes the model on the provided tuning data. Saves model state to disk.
		"""
		
		# function to return the weight updates so we can save them every epoch
		def get_updated_weights():
			updated_weights = {}
			for tok in self.tokens_to_mask:
				tok_id = self.tokenizer.get_vocab()[tok]
				updated_weights[tok_id] = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight[tok_id,:].clone()
			
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
			seed = int(torch.randint(2**32-1, (1,)))
			set_seed(seed)
			log.info(f"Seed set to {seed}")
			
			nn.init.normal_(new_embeds.weight, mean=mean, std=std)
			for i, tok in enumerate(self.tokens_to_mask):
				tok_id = self.tokenizer.get_vocab()[tok]
				getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight[tok_id] = new_embeds.weight[i]
			
		if not self.tuning_data:
			log.info(f'Saving randomly initialized weights')
			with open('weights.pkl', 'wb') as f:
				pkl.dump({0: get_updated_weights()}, f)
			return
		
		# Collect Hyperparameters
		lr = self.cfg.hyperparameters.lr
		epochs = self.cfg.hyperparameters.epochs
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
		
		# Determine what data to use based on the experiment
		# Do this once ahead of time if we are not changing it
		# but do it in the loop if we are using the bert-style randomized tuning data per epoch
		if self.cfg.tuning.new_verb:
			args = self.verb_tuning_data['args']
			with open('args.yaml', 'w') as outfile:
				outfile.write(OmegaConf.to_yaml(args))
		
		if self.cfg.tuning.new_verb and self.masked_tuning_style == 'none':
			inputs_data = self.verb_tuning_data['data']
			dev_inputs_data = {dataset: self.verb_dev_data[dataset]['data'] for dataset in self.verb_dev_data}
		elif self.masked and self.masked_tuning_style == 'always':
			inputs_data = self.masked_tuning_data
			dev_inputs_data = self.masked_dev_data
		elif self.masked and self.masked_tuning_style in ['bert', 'roberta']: # when using bert tuning or roberta tuning. For roberta tuning, this is done later on
			inputs_data = self.mixed_tuning_data
			dev_inputs_data = self.mixed_dev_data
		elif not self.masked:
			inputs_data = self.tuning_data
			dev_inputs_data = self.dev_data
		
		labels_data = self.verb_tuning_data['data'] if self.cfg.tuning.new_verb else self.tuning_data
		dev_labels_data = {dataset: self.verb_dev_data[dataset]['data'] for dataset in self.verb_dev_data} if self.cfg.tuning.new_verb else self.dev_data
		
		if not (verify_tokenization_of_sentences(self.tokenizer, inputs_data, self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs) and \
			    verify_tokenization_of_sentences(self.tokenizer, labels_data, self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs)):
			log.error('The new tokens added affected the tokenization of other elements in the inputs! Try using different strings.')
			return
		
		for dataset in dev_inputs_data:
			if not (verify_tokenization_of_sentences(self.tokenizer, dev_inputs_data[dataset], self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs) and \
				    verify_tokenization_of_sentences(self.tokenizer, dev_labels_data[dataset], self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs)):
				log.error(f'The new tokens added affected the tokenization of other elements in the dev inputs for dataset {dataset}! Try using different strings.')
				return
		
		inputs = self.tokenizer(inputs_data, return_tensors="pt", padding=True)
		labels = self.tokenizer(labels_data, return_tensors="pt", padding=True)["input_ids"]
		
		dev_inputs = {dataset: self.tokenizer(dev_inputs_data[dataset], return_tensors='pt', padding=True) for dataset in dev_inputs_data}
		dev_labels = {dataset: self.tokenizer(dev_labels_data[dataset], return_tensors='pt', padding=True)['input_ids'] for dataset in dev_labels_data}
		
		# used to calculate metrics during training
		masked_inputs = self.tokenizer(self.masked_tuning_data, return_tensors="pt", padding=True)
		masked_dev_inputs = {dataset: self.tokenizer(self.masked_dev_data[dataset], return_tensors='pt', padding=True) for dataset in self.masked_dev_data}
		
		log.info(f"Training model @ '{os.getcwd().replace(hydra.utils.get_original_cwd(), '')}'")
		
		# Store weights pre-training so we can inspect the initial status later
		saved_weights = {}
		saved_weights[0] = get_updated_weights()
		
		datasets = [self.cfg.tuning.name + ' (train)'] + [dataset + ' (dev)' for dataset in self.cfg.dev]
		
		metrics = pd.DataFrame(data = {
			'epoch' : list(range(1,epochs+1)) * len(datasets),
			'dataset' : np.repeat(datasets, [epochs] * len(datasets))
		})
		metrics['loss'] = np.nan
		
		writer = SummaryWriter()
		
		with trange(epochs) as t:
			for epoch in t:
				
				self.model.train()
				
				# optimizer.zero_grad()
				optimizer.zero_grad(set_to_none=True) # this is supposed to be faster
				
				# If we are using roberta-style masking, get new randomly changed inputs each epoch
				if self.masked_tuning_style == 'roberta':
					inputs_data = self.mixed_tuning_data
					dev_inputs_data = self.mixed_dev_data
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
					
					for dataset in dev_inputs_data:
						count = 0
						while not verify_tokenization_of_sentences(self.tokenizer, dev_inputs_data[dataset], self.tokens_to_mask, **self.cfg.model.tokenizer_kwargs):
							count += 1
							log.warning('The new tokens added affected the tokenization of dev sentences generated using roberta-style tuning!')
							log.warning(f'Affected: {dataset}, {dev_inputs_data}')
							log.warning('Rerolling to try again.')
							dev_inputs_data[dataset] = self.mixed_dev_data[dataset]
							if count > 10:
								log.error(f'Unable to find roberta-style masked dev data for {dataset} that was tokenized correctly after 10 tries. Exiting.')
								return
					
					inputs = self.tokenizer(inputs_data, return_tensors="pt", padding=True)
					dev_inputs = {dataset: self.tokenizer(dev_inputs_data[dataset], return_tensors='pt', padding=True) for dataset in dev_inputs_data}
				
				# Compute loss
				train_outputs = self.model(**inputs, labels=labels)
				train_loss = train_outputs.loss
				train_loss.backward()
				
				# Log result
				metrics.loc[(metrics.epoch == epoch + 1) & (metrics.dataset == self.cfg.tuning.name + ' (train)'), 'loss'] = train_loss.item()
				tb_loss_dict = {f'loss/{self.cfg.tuning.name.replace("_", " ") + " (train)"}': train_loss}
				
				train_results = self.collect_results(masked_inputs, labels, self.tokens_to_mask, train_outputs)
				
				# get metrics for plotting
				epoch_metrics = self.get_epoch_metrics(train_results)
				
				tb_metrics_dict = {}
				for metric in epoch_metrics:
					tb_metrics_dict[metric] = {}
					for token in epoch_metrics[metric]:
						tb_metrics_dict[metric][token] = {}
						metrics.loc[(metrics.epoch == epoch + 1) & (metrics.dataset == self.cfg.tuning.name + ' (train)'), f'{token} mean {metric} in expected position'] = epoch_metrics[metric][token]
						tb_metrics_dict[metric][token].update({f'{token} mean {metric}/{self.cfg.tuning.name.replace("_", " ") + " (train)"}': epoch_metrics[metric][token]})
				
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
				
				# # Check that we changed the correct number of parameters
				new_embeddings = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.clone()
				num_changed_params = torch.round(torch.sum(torch.mean(torch.ne(self.old_embeddings, new_embeddings) * 1., dim = -1))) # use torch.round to attempt to fix rare floating point rounding error
				num_expected_to_change = len(self.tokens_to_mask)
				assert num_changed_params == num_expected_to_change, f"Exactly {num_expected_to_change} embeddings should have been updated, but {num_changed_params} were!"
				
				# evaluate the model on the dev set
				self.model.eval()
				with torch.no_grad():
					dev_losses = []
					for dataset in dev_inputs:
						dev_outputs = self.model(**dev_inputs[dataset], labels=dev_labels[dataset])
						dev_loss = dev_outputs.loss
						dev_losses += [dev_loss.item()]
						
						metrics.loc[(metrics.epoch == epoch + 1) & (metrics.dataset == self.cfg.dev[dataset].name + ' (dev)'),'loss'] = dev_loss.item()
						tb_loss_dict.update({f'loss/{self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"}': dev_loss})
						
						dev_results = self.collect_results(masked_dev_inputs[dataset], dev_labels[dataset], self.tokens_to_mask, dev_outputs)
						
						dev_epoch_metrics = self.get_epoch_metrics(dev_results)
						
						for metric in dev_epoch_metrics:
							for token in dev_epoch_metrics[metric]:
								metrics.loc[(metrics['epoch'] == epoch + 1) & (metrics.dataset == self.cfg.dev[dataset].name + ' (dev)'), f'{token} mean {metric} in expected position'] = dev_epoch_metrics[metric][token]
								tb_metrics_dict[metric][token].update({f'{token} mean {metric}/{self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"}': dev_epoch_metrics[metric][token]})
				
				writer.add_scalars(f'{self.model_bert_name} loss; masking, {self.cfg.hyperparameters.masked_tuning_style}; {"no punctuation" if self.cfg.hyperparameters.strip_punct else "punctuation"}', tb_loss_dict, epoch)
				for metric in tb_metrics_dict:
					for token in tb_metrics_dict[metric]:
						writer.add_scalars(f'{self.model_bert_name} {token} {metric}; masking, {self.cfg.hyperparameters.masked_tuning_style}; {"no punctuation" if self.cfg.hyperparameters.strip_punct else "punctuation"}', tb_metrics_dict[metric][token], epoch)
				
				if self.cfg.dev:
					t.set_postfix(avg_dev_loss='{0:5.2f}'.format(np.mean(dev_losses)), train_loss='{0:5.2f}'.format(train_loss.item()))
				else:
					t.set_postfix(train_loss='{0:5.2f}'.format(train_loss.item()))
		
		log.info(f"Saving weights for each of {epochs} epochs")
		with open('weights.pkl', 'wb') as f:
			pkl.dump(saved_weights, f)
		
		metrics['dataset_type'] = ['dev' if re.search('(dev)', dataset) else 'train' for dataset in metrics.dataset]
		
		log.info(f'Plotting metrics')
		self.plot_metrics(metrics)
		
		metrics = pd.melt(
			metrics, 
			id_vars = ['epoch', 'dataset', 'dataset_type'], 
			value_vars = [c for c in metrics.columns if not c in ['epoch', 'dataset', 'dataset_type']], 
			var_name = 'metric'
		).assign(
			model_id = os.path.split(os.getcwd())[1],
			model_name = self.model_bert_name,
			tuning = self.cfg.tuning.name,
			masked = self.masked,
			masked_tuning_style = self.masked_tuning_style,
			strip_punct = self.cfg.hyperparameters.strip_punct,
			dataset = lambda df: [d.replace('_', ' ') for d in df.dataset]
		)
		
		log.info(f"Saving metrics")
		metrics.to_csv("metrics.csv", index = False)
		
		writer.flush()
		writer.close()
	
	def collect_results(self, masked_inputs, labels, eval_groups, outputs) -> Dict:
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
					token_id = self.tokenizer.get_vocab()[token]
					if self.cfg.tuning.entail and labels[sentence_num,focus] == token_id:
						focus_results[token] = {}
						focus_results[token]['log probability'] = log_probabilities[sentence_num,focus,token_id].item()
						focus_results[token]['surprisal'] = surprisals[sentence_num,focus,token_id].item()
					elif not self.cfg.tuning.entail:
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
	
	def restore_weights(self, checkpoint_dir: str, epoch: Union[int,str] = 'best') -> Tuple[int, int]:
		weights_path = os.path.join(checkpoint_dir, 'weights.pkl')
		
		with open(weights_path, 'rb') as f:
			weights = pkl.load(f)
			
		total_epochs = max(weights.keys())
		
		if epoch == None:
			epoch = total_epochs
		elif epoch == 'best':
			metrics = pd.read_csv(os.path.join(checkpoint_dir, 'metrics.csv'))
			loss_df = metrics[metrics.metric == 'loss']
			epoch = get_best_epoch(loss_df)
		
		log.info(f'Restoring saved weights from epoch {epoch}/{total_epochs}')
		
		with torch.no_grad():
			for tok_id in weights[epoch]:
				getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight[tok_id] = weights[epoch][tok_id]
		
		# return the epoch and total_epochs to help if we didn't specify it
		return epoch, total_epochs
	
	def plot_metrics(self, metrics: pd.DataFrame) -> None:
		
		def determine_int_xticks(target_num_ticks: int = 10) -> List[int]:
			lowest = metrics.epoch.min()
			highest = metrics.epoch.max()
			
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
			
			int_xticks = [int(i) for i in list(range(lowest - 1, highest + 1, int(ceil(highest/target_num_ticks))))]
			int_xticks = [i for i in int_xticks if i in metrics.epoch.values]
			
			return int_xticks
		
		all_metrics = [m for m in metrics.columns if not m in ['epoch', 'dataset', 'dataset_type']]
		
		xticks = determine_int_xticks()
		
		for metric in all_metrics:
			# Get the other metrics which are like this one but for different tokens so that we can
			# set the axis limits to a common value. This is so we can compare the metrics visually
			# for each token more easily
			like_metrics = []
			for m in all_metrics:
				if not m == metric:
					m1 = metric
					m2 = m
					for token in self.tokens_to_mask:
						m1 = m1.replace(token.upper(), '').replace(token.lower(), '') # do this to deal with both cased and uncased models
						m2 = m2.replace(token.upper(), '').replace(token.lower(), '') # do this to deal with both cased and uncased models
					
					if m1 == m2:
						like_metrics.append(m)
			
			ulim = np.max([*metrics[metric].dropna().values])
			llim = np.min([*metrics[metric].dropna().values])
			
			for m in like_metrics:
				ulim = np.max([ulim, *metrics[m].dropna().values])
				llim = np.min([llim, *metrics[m].dropna().values])
			
			adj = max(np.abs(ulim - llim)/40, 0.05)
			
			fig, ax = plt.subplots(1)
			fig.set_size_inches(8, 6)
			ax.set_ylim(llim - adj, ulim + adj)
			metrics.dataset = [dataset.replace('_', ' ') for dataset in metrics.dataset] # for legend titles
			if len(metrics[metric].index) > 1:
				sns.lineplot(data = metrics, x = 'epoch', y = metric, ax = ax, hue='dataset', style='dataset_type', legend='full')
				ax.legend(fontsize=9)
				
				# remove redundant information from the legend
				handles, labels = ax.get_legend_handles_labels()
				handles_labels = tuple(zip(handles, labels))
				handles_labels = [handle_label for handle_label in handles_labels if not handle_label[1] in ['dataset', 'dataset_type', 'train', 'dev']]
				handles = [handle for handle, _ in handles_labels]
				labels = [label for _, label in handles_labels]
				ax.legend(handles=handles, labels=labels)
			else:
				sns.scatterplot(data = metrics, x = 'epoch', y = metric, ax = ax, hue='dataset')
			
			plt.xticks(xticks)
			
			title = f'{self.model_bert_name} {metric}\n'
			title += f'tuning: {self.cfg.tuning.name.replace("_", " ")}, '
			title += ((f'masking: ' + self.masked_tuning_style) if self.masked else "unmasked") + ', '
			title += f'{"with punctuation" if not self.cfg.hyperparameters.strip_punct else "no punctuation"}\n\n'
			title += f'{self.cfg.tuning.name.replace("_", " ")} (training): max @ {metrics[metrics.dataset == self.cfg.tuning.name.replace("_"," ") + " (train)"].sort_values(by = metric, ascending = False).reset_index(drop = True)["epoch"][0]}: {round(metrics[metrics.dataset == self.cfg.tuning.name.replace("_", " ") + " (train)"].sort_values(by = metric, ascending = False).reset_index(drop = True)[metric][0],2)}, '
			title += f'min @ {metrics[metrics.dataset == self.cfg.tuning.name.replace("_"," ") + " (train)"].sort_values(by = metric).reset_index(drop = True)["epoch"][0]}: {round(metrics[metrics.dataset == self.cfg.tuning.name.replace("_", " ") + " (train)"].sort_values(by = metric).reset_index(drop = True)[metric][0],2)}'
			
			for dataset in self.cfg.dev:
				title += f'\n{dataset.replace("_", " ")} (dev): max @ {metrics[metrics.dataset == self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"].sort_values(by = metric, ascending = False).reset_index(drop = True)["epoch"][0]}: {round(metrics[metrics.dataset == self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"].sort_values(by = metric, ascending = False).reset_index(drop = True)[metric][0],2)}, '
				title += f'min @ {metrics[metrics.dataset == self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"].sort_values(by = metric).reset_index(drop = True)["epoch"][0]}: {round(metrics[metrics.dataset == self.cfg.dev[dataset].name.replace("_", " ") + " (dev)"].sort_values(by = metric).reset_index(drop = True)[metric][0],2)}'
			
			title = ax.set_title(title)
			fig.tight_layout()
			fig.subplots_adjust(top=0.75)
			plt.savefig(f"{metric}.pdf")
			plt.close('all')
			del fig
		
		# Combine the plots into a single pdf
		pdfs = [pdf for metric in all_metrics for pdf in os.listdir(os.getcwd()) if pdf == f'{metric}.pdf']
		merge_pdfs(pdfs, 'metrics.pdf')
	
	
	def eval(self, eval_cfg: DictConfig, checkpoint_dir: str, epoch: Union[int,str] = 'best') -> None:
		
		self.model.eval()
		_, _ = self.restore_weights(checkpoint_dir, epoch)
		
		# Load data
		# the use of eval_cfg.data.to_mask will probably need to be updated here for roberta now
		inputs, labels, sentences = self.load_eval_file(eval_cfg.data.name, eval_cfg.data.to_mask)
		
		# Calculate results on given data
		with torch.no_grad():	
			log.info("Evaluating model on testing data")
			outputs = self.model(**inputs)
		
		results = self.collect_results(inputs, eval_cfg.data.eval_groups, outputs)
		summary = self.summarize_results(results, labels)
		
		log.info("Creating graphs")
		self.graph_results(results, summary, eval_cfg)
	
	def load_eval_file(self, data_path: str, replacing: Dict[str, str]) -> Tuple:
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
	
	def summarize_results(self, results: Dict, labels) -> Dict:
		
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
	
	
	def eval_entailments(self, eval_cfg: DictConfig, checkpoint_dir: str, epoch: Union[int,str] = None) -> None:
		"""
		Computes model performance on data consisting of 
			sentence 1 , sentence 2 , [...]
		where credit for a correct prediction on sentence 2[, 3, ...] is contingent on
		also correctly predicting sentence 1
		"""
		log.info(f"SAVING TO: {os.getcwd().replace(hydra.utils.get_original_cwd(), '')}")
		
		# Load model
		self.model.eval()
		epoch_label = ('-' + epoch) if isinstance(epoch, str) else ''
		epoch, total_epochs = self.restore_weights(checkpoint_dir, epoch)
		
		data = self.load_eval_entail_file(eval_cfg.data.name, eval_cfg.data.to_mask)
		inputs = data["inputs"]
		labels = data["labels"]
		sentences = data["sentences"]
		
		assert len(inputs) == len(labels), f"Inputs (size {len(inputs)}) must match labels (size {len(labels)}) in length"
		
		# Calculate performance on data
		with torch.no_grad():
			log.info("Evaluating model on testing data")
			outputs = [self.model(**i) for i in inputs]
			
		summary = self.get_entailed_summary(sentences, outputs, labels, eval_cfg)
		summary = summary.assign(
			eval_epoch = epoch,
			total_epochs = total_epochs
		)
		# save the summary as a pickle and as a csv so that we have access to the original tensors
		# these get converted to text in the csv, but the csv is easier to work with otherwise
		dataset_name = eval_cfg.data.friendly_name
		epoch_label = f'{epoch}{epoch_label}'
		summary.to_pickle(f"{dataset_name}-{epoch_label}-scores.pkl")
		
		summary_csv = summary.copy()
		summary_csv['odds_ratio'] = summary_csv['odds_ratio'].astype(float).copy()
		summary_csv.to_csv(f"{dataset_name}-{epoch_label}-scores.csv", index = False)
		
		log.info('Creating plots')
		self.graph_entailed_results(summary, eval_cfg)
		
		if 'best' in epoch_label:
			plots_file = f'{dataset_name}-{epoch}-plots.pdf'
			os.rename(plots_file, f'{dataset_name}-{epoch_label}-plots.pdf')
		
		acc = self.get_entailed_accuracies(summary)
		acc.to_csv(f'{dataset_name}-{epoch_label}-accuracies.csv', index = False)
		
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
	
	def get_entailed_summary(self, sentences: List[List[str]], outputs: Dict, labels: Dict, eval_cfg: DictConfig) -> pd.DataFrame:
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
		all_combinations = pd.DataFrame(columns = ['sentence_type', 'token'],
			data = itertools.product(*[eval_cfg.data.sentence_types, list(tokens_indices.keys())]))
		
		cols = ['eval_data', 'exp_token', 'focus', 
				'sentence_type', 'sentence_num', 'exp_logit', 
				'logit', 'ratio_name', 'odds_ratio']
		
		summary = pd.DataFrame(columns = cols)
		for token in tokens_indices:
			for sentence_type, label in zip(sentence_types, labels):
				token_summary = pd.DataFrame(columns = cols)
				if (indices := torch.where(label == tokens_indices[token])[1]).nelement() != 0:
					token_summary = token_summary.assign(
						focus = indices,
						exp_token = token,
						sentence_type = sentence_type,
						sentence_num = lambda df: list(range(len(df.index)))
					)
					
					token_summary = token_summary.merge(all_combinations, how = 'left').fillna(0)
					logits = []
					exp_logits = []
					for row, idx in enumerate(token_summary['focus']):
						row_sentence_num = token_summary['sentence_num'][row]
						
						row_token = token_summary['token'][row]
						idx_row_token = tokens_indices[row_token]
						logits.append(sentence_type_logprobs[sentence_type][row_sentence_num,idx,idx_row_token])
						
						exp_row_token = token_summary['exp_token'][row]
						idx_exp_row_token = tokens_indices[exp_row_token]
						exp_logits.append(sentence_type_logprobs[sentence_type][row_sentence_num,idx,idx_exp_row_token])
					
					token_summary = token_summary.assign(
						logit = logits,
						exp_logit = exp_logits,
						# convert the case of the token columns to deal with uncased models; 
						# otherwise we won't be able to directly
						# compare them to cased models since the tokens will be different
						#### actually, don't: do this later during the comparison itself. it's more accurate
						#exp_token = [token.upper() for token in token_summary['exp_token']],
						exp_token = token_summary['exp_token'],
						#token = [token.upper() for token in token_summary['token']],
						token = token_summary['token'],
					).query('exp_token != token').copy().assign(
						ratio_name = lambda df: df["exp_token"] + '/' + df["token"],
						odds_ratio = lambda df: df['exp_logit'] - df['logit'],
					)
					
					summary = summary.append(token_summary, ignore_index = True)
		
		summary['role_position'] = [tokens_to_roles[token] + ' position' for token in summary['exp_token']]
		
		# Get formatting for linear positions instead of expected tokens
		summary = summary.sort_values(['sentence_type', 'sentence_num', 'focus'])
		summary['position_num'] = summary.groupby(['sentence_num', 'sentence_type'])['focus'].cumcount() + 1
		summary['position_num'] = ['position ' + str(num) for num in summary['position_num']]
		summary = summary.sort_index()
		
		# Add the actual sentences to the summary
		sentences_with_types = tuple(zip(*[tuple(zip(sentence_types, s_tuples)) for s_tuples in sentences]))
		
		sentences_with_types = [
			(i, *sentence) 
			for s_type in sentences_with_types 
				for i, sentence in enumerate(s_type)
		]
		
		sentences_df = pd.DataFrame({
			'sentence_num' : [t[0] for t in sentences_with_types],
			'sentence_type' : [t[1] for t in sentences_with_types],
			'sentence' : [t[2] for t in sentences_with_types]
		})
		
		summary = summary.merge(sentences_df, how = 'left')
		summary = summary.drop(['exp_logit', 'logit', 'token', 'exp_token', 'focus'], axis = 1)
		
		# Add a unique model id to the summary as well to facilitate comparing multiple runs
		# The ID comes from the runtime of the model plus the first letter of its
		# model name to ensure that it matches when the 
		# model is evaluated on different data sets
		model_id = os.path.normpath(os.getcwd()).split(os.sep)[-2] + '-' + self.model_bert_name[0]
		summary.insert(0, 'model_id', model_id)
		
		summary = summary.assign(
			eval_data = eval_cfg.data.friendly_name,
			model_name = self.model_bert_name,
			masked = self.masked,
			masked_tuning_style = self.masked_tuning_style,
			tuning = self.cfg.tuning.name.replace('_', ' '),
			strip_punct = self.cfg.hyperparameters.strip_punct
		)
		
		return summary
	
	def graph_entailed_results(self, summary: pd.DataFrame, eval_cfg: DictConfig, axis_size: int = 8, pt_size: int = 24) -> None:
		if len(np.unique(summary.model_id.values)) > 1:
			summary['odds_ratio'] = summary['mean']
			summary = summary.drop('mean', axis = 1)
			
			# if we are dealing with multiple models, we want to compare them by removing the idiosyncratic variation in how
			# tokenization works. bert and distilbert are uncased, which means the tokens are converted to lower case.
			# here, we convert them back to upper case so they can be plotted in the same group as the roberta tokens,
			# which remain uppercase
			summary.loc[(summary['model_name'] == 'bert') | (summary['model_name'] == 'distilbert'), 'ratio_name'] = \
			summary[(summary['model_name'] == 'bert') | (summary['model_name'] == 'distilbert')]['ratio_name'].str.upper()
			
			# for roberta, strings with spaces in front of them are tokenized differently from strings without spaces
			# in front of them. so we need to remove the special characters that signals that, and add a new character
			# signifying 'not a space in front' to the appropriate cases instead
			
			# first, check whether doing this will alter information
			if not summary[summary['model_name'] == 'roberta'].empty:
				roberta_summary = summary[summary['model_name'] == 'roberta'].copy()
				num_tokens_in_summary = len(set(list(itertools.chain(*[ratio_name.split('/') for ratio_name in roberta_summary.ratio_name.unique().tolist()]))))
				roberta_summary['ratio_name'] = [re.sub(chr(288), '', ratio_name) for ratio_name in roberta_summary['ratio_name']]
				num_tokens_after_change = len(set(list(itertools.chain(*[ratio_name.split('/') for ratio_name in roberta_summary.ratio_name.unique().tolist()]))))
				if num_tokens_in_summary != num_tokens_after_change:
					# this isn't going to actually get rid of any info, but it's worth logging
					log.warning('RoBERTa tokens were used with and without preceding spaces. This may complicate comparing results to BERT models.')
				
				# first, replace the ones that don't start with spaces before with a preceding ^
				summary.loc[(summary['model_name'] == 'roberta') & ~(summary['ratio_name'].str.startswith(chr(288))), 'ratio_name'] = \
				summary[(summary['model_name'] == 'roberta') & ~(summary['ratio_name'].str.startswith(chr(288)))].replace({r'((^\w)|(?<=\/)\w)' : r'^\1'})
				
				# then, replace the ones with the preceding special character (since we are mostly using them in the middle of sentences)
				summary.loc[(summary['model_name'] == 'roberta') & (summary['ratio_name'].str.startswith(chr(288))), 'ratio_name'] = \
				[re.sub(chr(288), '', ratio_name) for ratio_name in summary[(summary['model_name'] == 'roberta') & (summary['ratio_name'].str.startswith(chr(288)))].ratio_name]
		else:
			summary['sem'] = 0
		
		# Set colors for every unique odds ratio we are plotting
		all_ratios = summary['ratio_name'].unique()
		colors = dict(zip(all_ratios, ['teal', 'r', 'forestgreen', 'darkorange', 'indigo', 'slategray']))
		
		# we do this so we can add the information to the plot labels
		acc = self.get_entailed_accuracies(summary)
		
		dataset_name = eval_cfg.data.name.split('.')[0]
		
		# Get each unique pair of sentence types so we can create a separate plot for each pair
		sentence_types = summary['sentence_type'].unique()
		paired_sentence_types = list(itertools.combinations(sentence_types, 2))
		
		# Sort so that the trained cases are first
		paired_sentence_types = [
			sorted(pair, 
				   key = lambda x: str(-int(x == self.reference_sentence_type)) + x) 
			for pair in paired_sentence_types
		]
		
		# Filter to only cases including the reference sentence type for ease of interpretation
		paired_sentence_types = [(s1, s2) for s1, s2 in paired_sentence_types if s1 == self.reference_sentence_type] if self.reference_sentence_type != 'none' else [(s1, s2) for s1, s2 in paired_sentence_types]

		# For each pair, we create a different plot
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
				fig.set_size_inches(9, 10.5)
			else:
				fig, (ax1, ax2) = plt.subplots(1, 2)
				fig.set_size_inches(9, 6.5)
			
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
			ax1.set_xlabel(f"Confidence in {pair[0]} sentences", fontsize = axis_size)
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
			ax2.set_xlabel(f"Confidence in {pair[0]} sentences", fontsize = axis_size)
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
				
				xlabel = '\n'.join(xlabel)
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
				
				xlabel = '\n'.join(xlabel)
				ylabel = '\n'.join(ylabel)
				
				ax4.set_xlabel(xlabel, fontsize = axis_size)
				ax4.set_ylabel(ylabel, fontsize = axis_size)
				
				ax4.legend(prop = {'size': axis_size})
			
			# Set title
			title = re.sub(r"\'\s(.*?)", f"' {', '.join(pair)} ", eval_cfg.data.description.replace('tuples', 'pairs'))
			title += (' @ epoch ' + str(np.unique(summary.eval_epoch)[0])) if len(np.unique(summary.eval_epoch)) == 1 else ''
			title += ('/' + str(np.unique(summary.total_epochs)[0])) if len(np.unique(summary.total_epochs)) == 1 else ''
			
			model_name = np.unique(summary.model_name)[0] if len(np.unique(summary.model_name)) == 1 else 'multiple'
			masked_str = ', masking' if all(summary.masked) else 'unmasked' if all(1 - summary.masked) else 'multiple'
			masked_tuning_str = ': ' + np.unique(summary.masked_tuning_style[summary.masked])[0] if len(np.unique(summary.masked_tuning_style[summary.masked])) == 1 else ': multiple' if any(summary.masked) else ''
			subtitle = f'Model: {model_name}{masked_str}{masked_tuning_str}'
			
			tuning_data_str = np.unique(summary.tuning)[0] if len(np.unique(summary.tuning)) == 1 else 'multiple'
			subtitle += '\nTuning data: ' + tuning_data_str
			
			strip_punct_str = 'without punctuation' if all(summary.strip_punct) else "with punctuation" if all(~summary.strip_punct) else 'Multiple punctuation'
			subtitle += ' ' + strip_punct_str
			
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
			plt.savefig(f"{dataset_name}-{pair[0]}-{pair[1]}-paired.pdf")
			plt.close('all')
			del fig
		
		# Combine the plots into a single pdf
		pdfs = [pdf for sentence_type in sentence_types for pdf in os.listdir(os.getcwd()) if pdf.endswith(f'{sentence_type}-paired.pdf')]
		
		# Filter out duplicate file names
		pdfs = list(set(pdfs))
		keydict = eval_cfg.data.sentence_types
		keydict = {k : v for v, k in enumerate(keydict)}
		
		pdfs = sorted(pdfs, key = lambda pdf: keydict[pdf.replace(dataset_name + '-' + self.reference_sentence_type + '-', '').replace('.pdf', '').replace('-paired', '')])
		
		total_epochs = max(summary.total_epochs)
		magnitude = floor(1 + np.log10(total_epochs))
		
		all_epochs = '-'.join([str(x).zfill(magnitude) for x in sorted(np.unique(summary.eval_epoch).tolist(), key = lambda x: x)])
		merge_pdfs(pdfs, f'{dataset_name}-{all_epochs}-plots.pdf')
	
	def get_entailed_accuracies(self, summary: pd.DataFrame) -> pd.DataFrame:
		# Get each unique pair of sentence types so we can create a separate plot for each pair
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
					   'ref_correct', 'ref_incorrect', 'gen_correct', 'gen_incorrect', 'num_points', 'specificity_(MSE)', 'specificity_se', 'specificity_(z)', 'specificity_se(z)', 's1_ex', 's2_ex']
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
			
			specificity_z = np.mean(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2)
			specificity_z_sem = np.std(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2/np.sqrt(np.size(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2)))
			
			s1_ex = x_data[x_data.sentence_num == 0].sentence.values[0]
			s2_ex = y_data[y_data.sentence_num == 0].sentence.values[0]
			
			acc = acc.append(pd.DataFrame(
				[[pair[0], pair[1], 'any', 'any',
				  x_data.position_num.unique()[0] if len(x_data.position_num.unique()) == 1 else 'multiple',
				  y_data.position_num.unique()[0] if len(y_data.position_num.unique()) == 1 else 'multiple',
				  gen_given_ref,
				  both_correct, ref_correct_gen_incorrect, 
				  both_incorrect, ref_incorrect_gen_correct, 
				  ref_correct, ref_incorrect, 
				  gen_correct, gen_incorrect, 
				  num_points, specificity, spec_sem,
				  specificity_z, specificity_z_sem,
				  s1_ex, s2_ex]],
				  columns = acc_columns
			))
			
			for name, x_group in x_data.groupby('ratio_name'):
				arg = name.split('/')[0]
				y_group = y_data[y_data.ratio_name == name]
				
				refs_correct = x_group.odds_ratio > 0
				gens_correct = y_group.odds_ratio > 0
				num_points = len(x_group.index)
				
				# Get the number of points in each quadrant
				gen_given_ref = sum(y_group[y_group.index.isin(x_group.loc[x_group.odds_ratio > 0].index)].odds_ratio > 0)/len(x_group.loc[x_group.odds_ratio > 0].index) * 100
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
				
				specificity_z = np.mean(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2)
				specificity_z_sem = np.std(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2/np.sqrt(np.size(z_transform(y_data.odds_ratio - x_data.odds_ratio)**2)))
				
				acc = acc.append(pd.DataFrame(
					[[pair[0], pair[1], arg, x_group.role_position.unique()[0].split()[0],
					  x_group.position_num.unique()[0] if len(x_group.position_num.unique()) == 1 else 'multiple',
					  y_group.position_num.unique()[0] if len(y_group.position_num.unique()) == 1 else 'multiple',
					  gen_given_ref, 
					  both_correct, ref_correct_gen_incorrect, 
					  both_incorrect, ref_incorrect_gen_correct, 
					  ref_correct, ref_incorrect, 
					  gen_correct, gen_incorrect, 
					  num_points, specificity, spec_sem,
					  specificity_z, specificity_z_sem,
					  s1_ex, s2_ex]],
					  columns = acc_columns
				))
		
		acc = acc.assign(
			eval_epoch = np.unique(summary.eval_epoch)[0] if len(np.unique(summary.eval_epoch)) == 1 else 'multi',
			total_epochs = np.unique(summary.total_epochs)[0] if len(np.unique(summary.total_epochs)) == 1 else 'multi',
			model_id = np.unique(summary.model_id)[0] if len(np.unique(summary.model_id)) == 1 else 'multi',
			eval_data = np.unique(summary.eval_data)[0] if len(np.unique(summary.eval_data)) == 1 else 'multi',
			model_name = np.unique(summary.model_name)[0] if len(np.unique(summary.model_name)) == 1 else 'multi',
			tuning = np.unique(summary.tuning)[0] if len(np.unique(summary.tuning)) == 1 else 'multi',
			masked = np.unique(summary.masked)[0] if len(np.unique(summary.masked)) == 1 else 'multi',
			masked_tuning_style = np.unique(summary.masked_tuning_style)[0] if len(np.unique(summary.masked_tuning_style)) == 1 else 'multi',
			strip_punct = np.unique(summary.strip_punct)[0] if len(np.unique(summary.strip_punct)) == 1 else 'multi'
		)
		
		return acc
	
	
	def eval_new_verb(self, eval_cfg: DictConfig, args_cfg: DictConfig, checkpoint_dir: str, epoch: Union[int,str] = None) -> None:
		"""
		Computes model performance on data with new verbs
		where this is determined as the difference in the probabilities associated
		with each argument to be predicted before and after training.
		To do this, we check predictions for each arg, word pair in args_cfg on a fresh model, 
		and then check them on the fine-tuned model.
		"""
		from transformers import pipeline
		
		data = self.load_eval_verb_file(args_cfg, eval_cfg.data.name, eval_cfg.data.to_mask)
				
		self.model.eval()
		
		# Define a local function to get the probabilities
		def get_probs(epoch: int):
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
		summary.to_pickle(f"{dataset_name}-0-{epoch}-scores.pkl")
		summary.to_csv(f"{dataset_name}-0-{epoch}-scores.csv", index = False)
		
		# Create graphs
		log.info('Creating plots')
		self.graph_new_verb_results(summary, eval_cfg)
		
		log.info('Evaluation complete')
		print('')
	
	def load_eval_verb_file(self, args_cfg: DictConfig, data_path: str, replacing: Dict[str, str]) -> Dict[str, List[str]]:
		
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
		"""
		Convert the pre- and post-tuning results into a pandas.DataFrame
		"""
		
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
											total_epochs = total_epochs
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
										
										summary = summary.append(summary_, ignore_index = True)
			
			summary = summary.assign(
				model_id = os.path.normpath(os.getcwd()).split(os.sep)[-2],
				eval_data = eval_cfg.data.friendly_name,
				model_name = self.model_bert_name,
				masked = self.masked,
				masked_tuning_style = self.masked_tuning_style,
				tuning = self.cfg.tuning.name,
				strip_punct = self.cfg.hyperparameters.strip_punct
			)
			
			return summary
		
		summary = convert_results(results, args_cfg)
		
		# Reorder the columns
		columns = [
			'model_id', 'model_name', 'total_epochs', 
			'tuning', 'strip_punct', 'masked', 'masked_tuning_style', # model properties
			'eval_epoch', 'eval_data', # eval properties
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
		
		return summary
	
	def graph_new_verb_results(self, summary: pd.DataFrame, eval_cfg: DictConfig, axis_size: int = 10, pt_size: int = 24) -> None:
		
		if len(np.unique(summary.model_id.values)) > 1:
			summary['surprisal'] = summary['mean']
			summary = summary.drop('mean', axis = 1)
		else:
			summary['sem'] = 0
		
		# Get each sentence type to compare them on pre- and post-tuning data
		sentence_types = summary['sentence_type'].unique()
		
		dataset_name = eval_cfg.data.friendly_name
		
		summary['surprisal_gf_label'] = [row['predicted_token_type'] + ' in ' + row['target_position_name'] + ' position' for _, row in summary.iterrows()]
		summary['surprisal_pos_label'] = [row['predicted_token_type'] + ' in ' + row['target_position_num'].replace('_', ' ') for _, row in summary.iterrows()]
		
		# Set colors for every unique surprisal type we are plotting
		# This is hacky; find a way to fix it
		colors1 = ['teal', 'darkturquoise', 'maroon', 'r', 'blueviolet', 'indigo']
		colors2 = ['teal', 'r']
		
		x_epoch = min(summary.eval_epoch)
		y_epoch = max(summary.eval_epoch)

		# For each sentence type, we create a different plot
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
			fig.set_size_inches(12.25, 6)
			
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
			masked_str = 'masked' if all(summary.masked) else 'unmasked' if all(1 - summary.masked) else 'multiple'
			masked_tuning_str = ', Masking type: ' + np.unique(summary.masked_tuning_style[summary.masked])[0] if len(np.unique(summary.masked_tuning_style[summary.masked])) == 1 else ', Masked tuning style: multiple' if any(summary.masked) else ''
			subtitle = f'Model: {model_name} {masked_str}{masked_tuning_str}'
			
			tuning_data_str = np.unique(summary.tuning)[0] if len(np.unique(summary.tuning)) == 1 else 'multiple'
			subtitle += '\nTuning data: ' + tuning_data_str
			
			strip_punct_str = 'No punctuation' if all(summary.strip_punct) else "Punctuation" if all(~summary.strip_punct) else 'Multiple punctuation'
			subtitle += ', ' + strip_punct_str
			
			fig.suptitle(title + '\n' + subtitle)
			fig.tight_layout(rect=[-0.025,0.1,0.9625,1])
			plt.savefig(f"{dataset_name}-{sentence_type}.pdf")
			plt.close('all')
			del fig
		
		# Combine the plots into a single pdf
		pdfs = [pdf for sentence_type in sentence_types for pdf in os.listdir(os.getcwd()) if pdf.endswith(f'{sentence_type}.pdf')]
		
		# Filter out duplicate file names in case some sentence types are contained within others,
		# and sort so that the most relevant plot appears at the top
		pdfs = list(set(pdfs))
		keydict = eval_cfg.data.sentence_types
		keydict = {k : v for v, k in enumerate(keydict)}
		
		pdfs = sorted(pdfs, key = lambda pdf: keydict[pdf.replace(dataset_name + '-', '').replace('.pdf', '')])
		
		total_epochs = max(summary.total_epochs)
		magnitude = floor(1 + np.log10(total_epochs))
		
		all_epochs = '-'.join([str(x).zfill(magnitude) for x in sorted(np.unique(summary.eval_epoch).tolist(), key = lambda x: x)])
		merge_pdfs(pdfs, f'{dataset_name}-{all_epochs}-plots.pdf')
	
	
	# no longer used anywhere
	"""def collect_entailed_results(self, inputs, eval_groups, outputs):
		
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
		
		return results_arr"""
	
	# no longer used anywhere
	"""def summarize_entailed_results(self, results_arr, labels_arr):
		
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
		
		return confidences"""

	# no longer used anywhere
	"""@property
	def dev_tokens_to_mask(self) -> List[str]:
		dev_tokens_to_mask = {}
		for dataset in self.cfg.dev:
			# convert things to lowercase for uncased models
			tokens = [t.lower() for t in self.cfg.dev[dataset].to_mask] if 'uncased' in self.string_id else list(self.cfg.dev[dataset].to_mask)
			# add the versions of the tokens with preceding spaces to our targets for roberta
			if self.model_bert_name == 'roberta':
				tokens += [chr(288) + t for t in tokens]
			
			dev_tokens_to_mask.update({dataset: tokens})
		
	 	return dev_tokens_to_mask"""
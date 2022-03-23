# svm.py
#
# fit a svm to see if a hyperplane exists between selected token embeddings
import os
import gzip
import torch
import pandas as pd
import pickle as pkl
import torch.nn as nn
import logging

from omegaconf import DictConfig

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import logging as lg
lg.set_verbosity_error()

from time import sleep

from typing import *

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

log = logging.getLogger(__name__)

class EmbeddingClassifier:
	
	@property
	def grouped_token_embeddings(self) -> nn.Embedding:
		embeddings = []
		for group in self.grouped_tokens:
			group_tokens_ids_embeddings = [
				(group, token, token_id, self.embeddings[token_id,:]) 
				for token, token_id in zip(self.grouped_tokens[group], self.tokenizer.convert_tokens_to_ids(self.grouped_tokens[group]))
			]
			
			for _, token, token_id, _ in group_tokens_ids_embeddings:
				assert token_id != self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token), f"{token} does not exist in {self.cfg.model.friendly_name}'s vocabulary!"
				
			embeddings.extend(group_tokens_ids_embeddings)
		
		return embeddings
	
	def __init__(self, cfg_or_path: DictConfig) -> None:
		self.cfg = OmegaConf.load(os.path.join(cfg_or_path, '.hydra', 'config.yaml')) if isinstance(cfg_or_path, str) else cfg_or_path
		self.checkpoint_dir = cfg_or_path if isinstance(cfg_or_path, str) else os.getcwd()
		
		log.info(f'Initializing Tokenizer:\t{self.cfg.model.tokenizer}')
		self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.string_id, use_fast=False, **self.cfg.model.tokenizer_kwargs)
		
		log.info(f'Initializing Model:\t{self.cfg.model.base_class}')
		self.model = AutoModelForMaskedLM.from_pretrained(self.cfg.model.string_id, **self.cfg.model.model_kwargs)
		self.embeddings = getattr(self.model, self.model.config.model_type).embeddings.word_embeddings.weight.detach()
		
		if 'which_args' in self.cfg.tuning:
			self.initialize_grouped_tokens(self.cfg.tuning[self.cfg.tuning.which_args] if not self.cfg.tuning.which_args == 'model' else self.cfg.tuning[self.cfg.model.friendly_name])
		
		if os.path.isfile(os.path.join(self.checkpoint_dir, 'svm.pkl.gz')):
			log.info('Loading saved SVM state')
			with gzip.open(os.path.join(self.checkpoint_dir, 'svm.pkl.gz'), 'rb') as f:
				self.classifier = pkl.load(f)
		
	def initialize_grouped_tokens(self, grouped_tokens: Dict[str,List[str]] = None, overwrite: bool = True) -> None:
		if hasattr(self, 'grouped_tokens') and not overwrite:
			log.warning('EmbeddedClassifier already has defined grouped_tokens. Use overwrite=True to force.')
			return
		
		self.grouped_tokens, self.token_groups = self.get_grouped_tokens_and_groups(grouped_tokens)
		
	def get_grouped_tokens_and_groups(self, grouped_tokens: Dict[str,List[str]]):
		token_groups = [group for group in grouped_tokens]
		for group in grouped_tokens:
			_, grouped_tokens[group] = self.format_tokens(grouped_tokens[group])
		
		return grouped_tokens, token_groups
	
	def get_inputs_labels(self, grouped_tokens: Dict[str,List[str]] = None, overwrite: bool = False) -> Tuple[torch.Tensor, List[str]]:
		if not hasattr(self, 'grouped_tokens') and grouped_tokens is not None:
			self.initialize_grouped_tokens(grouped_tokens)
			grouped_tokens = self.grouped_tokens
		elif hasattr(self, 'grouped_tokens') and grouped_tokens is not None:
			grouped_tokens, _ = self.get_grouped_tokens_and_groups(grouped_tokens)
		elif hasattr(self, 'grouped_tokens') and grouped_tokens is None:
			grouped_tokens = self.grouped_tokens
		
		for group in grouped_tokens.copy():
			if not group in self.token_groups:
				log.warning(f'"{group}" is not a valid group identifier!')
				del grouped_tokens[group]
		
		groups_tokens_ids = [(g, t, self.tokenizer.convert_tokens_to_ids(t)) for g in grouped_tokens for t in grouped_tokens[g]]
		
		inputs = torch.stack([self.embeddings[token_id] for _, _, token_id in groups_tokens_ids])
		labels = [g for g, _, _ in groups_tokens_ids]
		
		return inputs, labels
	
	def fit(self, grouped_tokens: Dict[str,List[str]] = None, name: str = 'svm', overwrite: bool = False) -> None:
		inputs, labels = self.get_inputs_labels(grouped_tokens, overwrite)
		
		if hasattr(self, 'classifier') and not overwrite:
			log.warning('EmbeddedClassifier instance already has a classifier attribute. Use overwrite=True to force.')
			return
			
		if os.path.isfile(os.path.join(self.checkpoint_dir, f'{name}.pkl.gz')) and not overwrite:
			log.warning(f'An SVM checkpoint has already been saved at "{name}.pkl.gz". Choose a different name, or use overwrite=True to force.')
			return
		
		self.classifier = svm.SVC(kernel='linear')
		self.classifier.fit(inputs, labels)
		
		for group, token, _, embedding in self.grouped_token_embeddings:
			prediction = self.classifier.predict(torch.unsqueeze(embedding, dim=0)).item()
			if group != prediction:
				log.warning(f'"{token}" should have been classified as "{group}", but was instead classified as "{prediction}"!')
			
		with gzip.open(os.path.join(self.checkpoint_dir, f'{name}.pkl.gz'), 'wb') as f:
			pkl.dump(self.classifier, f)
			
		return self.get_metrics(inputs, labels)
	
	def get_metrics(self, inputs: torch.Tensor, labels: List[str], output_fun: Callable = log.info) -> Tuple:
		predictions = self.classifier.predict(inputs)
		report = classification_report(labels, predictions, output_dict=True)
		accuracy = accuracy_score(labels, predictions) * 100
		confusion = confusion_matrix(labels, predictions)
		
		if output_fun is not None:
			output_fun(f"Accuracy Score: {accuracy:.2f}%")
			output_fun(f"\n\nClassification report:\n{pd.DataFrame(report)}\n")
			output_fun(f"\n\nConfusion Matrix:\n{confusion}\n")
			sleep(2)
		
		return predictions, report, accuracy, confusion
	
	def classify(self, tokens: List[str]) -> Dict[str,str]:
		tokens, formatted_tokens = self.format_tokens(tokens)
		formatted_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in formatted_tokens]
		embeddings = torch.stack([self.embeddings[token_id] for token_id in formatted_token_ids])
		predictions = self.classifier.predict(embeddings)
		tokens_labels = list(zip(tokens, predictions))
		return tokens_labels
	
	def eval_classification(self, grouped_tokens: Dict[str,List[str]], **kwargs) -> Dict[str,str]:
		inputs, labels = self.get_inputs_labels(grouped_tokens)
		return self.classify([t for group in grouped_tokens for t in grouped_tokens[group]]), self.get_metrics(inputs, labels, **kwargs)
				
	def format_tokens(self, tokens: List[str]) -> Tuple[List[str]]:
		tokens = [tokens] if isinstance(tokens,str) else tokens
		
		# remove any duplicates
		tokens = list(set(tokens))
		
		formatted_tokens = tokens.copy()
		
		if self.cfg.model.friendly_name == 'roberta':
			formatted_tokens = [
				chr(288) + token 
				if not token.startswith('^') 
				else token.replace('^', '') 
				for token in formatted_tokens
			]
		
		# filter out inputs that don't exist in the tokenizer
		to_remove = []
		for i, token in enumerate(formatted_tokens.copy()):
			if self.tokenizer.convert_tokens_to_ids(token) == self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token):
				log.warning(f'{token} does not exist in {self.cfg.model.base_class}! No prediction will be made for this token.')
				to_remove.append(i)
		
		tokens = [token for token in tokens if not tokens.index(token) in to_remove]
		
		formatted_tokens = [
			token for token in formatted_tokens 
			if self.tokenizer.convert_tokens_to_ids(token) != self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
		]
		
		return tokens, formatted_tokens
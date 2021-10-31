# tuner.py
# 
# Tunes a model on training data.

import os
import hydra
from typing import Dict, List
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from transformers import	DistilBertForMaskedLM, DistilBertTokenizer, RobertaForMaskedLM, RobertaTokenizer, BertForMaskedLM, BertTokenizer
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
from tqdm import trange
import logging
import pickle as pkl
import pandas as pd
import re
import itertools
import sys

import random

log = logging.getLogger(__name__)

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

class Tuner:

	# START Computed Properties
	# gpu support removed due to system instability
	#@property
	#def device(self):
	#	return 'cuda' if torch.cuda.is_available() and self.model_bert_name == 'distilbert' else 'cpu'
	
	@property
	def model_class(self):
		if self.cfg.model.base_class == "DistilBertForMaskedLM":
			return DistilBertForMaskedLM
		elif self.cfg.model.base_class == "RobertaForMaskedLM":
			return RobertaForMaskedLM
		elif self.cfg.model.base_class == "BertForMaskedLM":
			return BertForMaskedLM
		elif self.cfg.model.base_class == 'multi':
			return None
	
	@property
	def tokenizer_class(self):
		if self.cfg.model.tokenizer == "DistilBertTokenizer":
			return DistilBertTokenizer
		elif self.cfg.model.tokenizer == "RobertaTokenizer":
			return RobertaTokenizer
		elif self.cfg.model.tokenizer == "BertTokenizer":
			return BertTokenizer
		elif self.cfg.model.base_class == 'multi':
			return None
	
	@property
	def model_bert_name(self) -> str:
		if self.cfg.model.base_class == "DistilBertForMaskedLM":
			return 'distilbert'
		elif self.cfg.model.base_class == "RobertaForMaskedLM":
			return 'roberta'
		elif self.cfg.model.base_class == "BertForMaskedLM":
			return 'bert'
		elif self.cfg.model.base_class == 'multi':
			return None
	
	@property
	def mask_tok(self):
		return self.tokenizer.mask_token
	
	@property
	def mask_tok_id(self):
		return self.tokenizer(self.mask_tok, return_tensors="pt")["input_ids"][:,1]
	
	@property
	def string_id(self) -> str:
		return self.cfg.model.string_id
	
	@property
	def masked_tuning_style(self):
		return self.cfg.hyperparameters.masked_tuning_style
		
	@property
	def masked(self):
		return self.cfg.hyperparameters.masked
	
	@property
	def tuning_data(self) -> List[str]:
		data = []
		for s in self.cfg.tuning.data:
			for key in self.tokens_to_mask:
				s = s.replace(key, self.tokens_to_mask[key])
			
			data.append(s)
		
		return [d.lower() for d in data]
	
	@property
	def mixed_tuning_data(self) -> List[str]:
		data = []
		for s in self.tuning_data:
			for val in list(self.tokens_to_mask.values()):
				r = np.random.random()
				# Bert tuning regimen
				# Masked tokens are masked 80% of the time, 
				# original 10% of the time,
				# and random word 10% of the time
				if r < 0.8:
					s = s.replace(val.lower(), self.mask_tok)
				elif 0.8 <= r < 0.9:
					pass
				elif 0.9 <= r:
					s = s.replace(val.lower(), np.random.choice(list(self.tokenizer.get_vocab().keys())))
				
			data.append(s)
		
		return data
	
	@property
	def masked_tuning_data(self) -> List[str]:
		data = []
		for s in self.tuning_data:
			for val in list(self.tokens_to_mask.values()):
				s = s.replace(val.lower(), self.mask_tok)
			
			data.append(s)
			
		return data
	
	@property
	def tokens_to_mask(self) -> Dict[str,str]:
		if self.model_bert_name != 'roberta':
			return self.cfg.tuning.to_mask
		# If we are using a roberta model, convert the masks to roberta-style masks
		else:
			try:
				if len(self.cfg.tuning.to_mask) > 3:
					raise ValueError("Insufficient unused tokens in RoBERTa vocabulary to train model on more than three novel tokens.")
				else:
					# We do this in a somewhat complex way to remain invariant to the order of the tokens in the cfg
					def atoi(text):
						return int(text) if text.isdigit() else text

					def natural_keys(text):
						return[atoi(c) for c in re.split(r'(\d+)', text)]
					
					orig_tokens = list(self.cfg.tuning.to_mask.values())
					orig_tokens.sort(key=natural_keys)

					bert_roberta_mapping = dict(zip(
							orig_tokens,
							('madeupword0000', 'madeupword0001', 'madeupword0002')
						))

					for token in self.cfg.tuning.to_mask:
						self.cfg.tuning.to_mask[token] = bert_roberta_mapping[self.cfg.tuning.to_mask[token]]

					return self.cfg.tuning.to_mask
			except ValueError as e:
				print(str(e))
				sys.exit(1)
	
	# END Computed Properties
	
	def __init__(self, cfg: DictConfig) -> None:
	
		self.cfg = cfg

		# Construct Model & Tokenizer
		if self.model_bert_name == 'roberta' and 'hyperparameters' in self.cfg:
			self.cfg.hyperparameters.masked_tuning_style = 'always'
		
		if self.string_id != 'multi':
			
			log.info(f"Initializing Tokenizer: {self.cfg.model.tokenizer}")
			
			self.tokenizer = self.tokenizer_class.from_pretrained(
				self.string_id, 
				do_basic_tokenize=False,
				local_files_only=True
			)

			# Re-add special tokens to roberta tokenizer for lookup purposes
			if self.tokenizer.name_or_path == 'roberta-base':
				self.tokenizer.add_tokens(
					['madeupword0000', 'madeupword0001', 'madeupword0002'], 
					special_tokens = True
				)

			log.info(f"Initializing Model: {self.cfg.model.base_class}")

			self.model = self.model_class.from_pretrained(
				self.string_id,
				local_files_only=True
			)
	
	def load_eval_data_file(self, data_path: str, replacing: Dict[str, str]):
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
			raw_sentences = [line.strip().lower() for line in f]
			sentences = []
			for s in raw_sentences:
				for key in replacing:
					s = s.replace(key, self.tokens_to_mask[replacing[key]])
				sentences.append(s)
			
		masked_sentences = []
		for s in sentences:
			m = s
			for val in list(self.tokens_to_mask.values()):
				m = m.replace(val, self.mask_tok)
			masked_sentences.append(m)
			
		inputs = self.tokenizer(masked_sentences, return_tensors="pt", padding=True)
		labels = self.tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"]
		
		return inputs, labels, sentences
	
	def load_eval_multi_file(self, data_path: str, replacing: Dict[str, str]):

		resolved_path = os.path.join(
			hydra.utils.get_original_cwd(),
			"data",
			data_path
		)

		with open(resolved_path, "r") as f:
			
			raw_input = [line.strip().lower() for line in f]
			sentences = []

			for r in raw_input:
				line = []
				s_splits = r.split(',')
				for s in s_splits:
					for key in replacing:
						s = s.replace(key, self.tokens_to_mask[replacing[key]])
					
					line.append(s.strip())
				
				sentences.append(line)
			
		masked_sentences = []
		for s_group in sentences:
			m_group = []
			for s in s_group:
				m = s
				for val in list(self.tokens_to_mask.values()):
					m = m.replace(val, self.mask_tok)
				
				m_group.append(m)
			
			masked_sentences.append(m_group)

		sentences_transposed = list(map(list, zip(*sentences)))
		masked_transposed = list(map(list, zip(*masked_sentences)))

		inputs = []
		labels = []

		for s in sentences_transposed:
			label = self.tokenizer(s, return_tensors="pt", padding=True)["input_ids"]
			labels.append(label)
		
		for m in masked_transposed:
			input = self.tokenizer(m, return_tensors="pt", padding=True)
			inputs.append(input)

		return {
			"inputs" : inputs,
			"labels" : labels,
			"sentences" : [[s.strip() for s in line.lower().split(',')] for line in raw_input]
		}
	
	def collect_results(self, inputs, eval_groups, outputs) -> Dict:

		results = {}
		
		logits = outputs.logits
		probabilities = torch.nn.functional.softmax(logits, dim=2)
		log_probabilities = torch.nn.functional.log_softmax(logits, dim=2)
		predicted_ids = torch.argmax(log_probabilities, dim=2)

		# print(f"Mask token id: {self.mask_tok_id}")
		# print("Inputs:")
		# print(inputs["input_ids"])

		for i, _ in enumerate(predicted_ids):

			sentence_results = {}

			# Foci = indices where input sentences have a [mask] token
			foci = torch.nonzero(inputs["input_ids"][i]==self.mask_tok_id, as_tuple=True)[0]
			
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
					'mean grouped log_probability' : idx_results,
					'log_probabilities' : log_probabilities[:,idx,:][i,:],
					'probabilities' : probabilities[:,idx,:][i,:],
					'logits': logits[:,idx,:][i,:]
				}
			
			results[i] = sentence_results
		
		return results
			
	def summarize_results(self, results: Dict, labels) -> Dict:
	
		summary = {}

		# Define theme and recipient ids
		ricket = self.tokenizer(self.tokens_to_mask["RICKET"], return_tensors="pt")["input_ids"][:,1]
		thax = self.tokenizer(self.tokens_to_mask["THAX"], return_tensors="pt")["input_ids"][:,1]

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
				scores = result[idx]['mean grouped log_probability']
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
	
	"""def collect_entailed_results(self, inputs, eval_groups, outputs):

		results_arr = []
		
		for j in range(len(outputs)):

			results = {}

			logits = outputs[j].logits
			probabilities = torch.nn.functional.softmax(logits, dim=2)
			log_probabilities = torch.nn.functional.log_softmax(logits, dim=2)
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
				'mean grouped log_probability' : idx_results,
				'log_probabilities' : log_probabilities[:,idx,:][i,:],
				'probabilities' : probabilities[:,idx,:][i,:],
				'logits': logits[:,idx,:][i,:]
				}
			results[i] = sentence_results
			
			results_arr.append(results)
		
		return results_arr"""
	
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
			scores = active_result[idx]['mean grouped log_probability']

			token_conf = scores['theme'] - scores['recipient']

			if target == ricket:
				# print("I'm in a recipient position")
				active_token_confidence["recipient"] = -token_conf
			else:
				# print("I'm in a theme position")
				active_token_confidence["theme"] = token_conf
			
			for idx in passive_result:

			target = passive_label[idx.item()]
			scores = passive_result[idx]['mean grouped log_probability']

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
	
	def graph_results(self, results: Dict, summary: Dict, eval_cfg: DictConfig):

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
	
	def get_entailed_summary(self, sentences, outputs, labels, eval_cfg: DictConfig):
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
		
		sentence_type_logprobs = {}
		
		for output, sentence_type in tuple(zip(outputs, sentence_types)):
			sentence_type_logprobs[sentence_type] = torch.nn.functional.log_softmax(output.logits, dim = 2)
		
		# Get the positions of the tokens in each sentence of each type
		tokens_indices = dict(zip(
			self.tokens_to_mask.keys(), 
			self.tokenizer.convert_tokens_to_ids(list(self.tokens_to_mask.values()))
		))
		
		# Get the expected positions for each token in the eval data
		all_combinations = pd.DataFrame(columns = ['sentence_type', 'token'],
			data = itertools.product(*[eval_cfg.data.sentence_types, list(tokens_indices.keys())]))
		
		cols = ['eval_data', 'exp_token', 'focus', 'sentence_type', 'sentence_num', 'exp_logit', 'logit', 'ratio_name', 'odds_ratio']
		
		summary = pd.DataFrame(columns = cols)
		
		for token in tokens_indices:
			for sentence_type, label in tuple(zip(sentence_types, labels)):
				token_summary = pd.DataFrame(columns = cols)
				if (indices := torch.where(label == tokens_indices[token])[1]).nelement() != 0:
					token_summary['focus'] = indices
					token_summary['exp_token'] = token
					token_summary['sentence_type'] = sentence_type
					token_summary['sentence_num'] = list(range(len(token_summary)))
					token_summary = token_summary.merge(all_combinations, how = 'left').fillna(0)
					logits = []
					exp_logits = []
					for i, idx in enumerate(token_summary['focus']):
						row_sentence_num = token_summary['sentence_num'][i]
						
						row_token = token_summary['token'][i]
						idx_row_token = tokens_indices[row_token]
						logits.append(sentence_type_logprobs[sentence_type][:,:,idx_row_token][row_sentence_num,idx])
							
						exp_row_token = token_summary['exp_token'][i]
						idx_exp_row_token = tokens_indices[exp_row_token]
						exp_logits.append(sentence_type_logprobs[sentence_type][:,:,idx_exp_row_token][row_sentence_num,idx])
						
					token_summary['logit'] = logits
					token_summary['exp_logit'] = exp_logits
					token_summary = token_summary.query('exp_token != token').copy()
					token_summary['ratio_name'] = token_summary["exp_token"] + '/' + token_summary["token"]
					token_summary['odds_ratio'] = token_summary['exp_logit'] - token_summary['logit']
				
				summary = summary.append(token_summary, ignore_index = True)
		
		tokens_roles = dict(zip(list(eval_cfg.data.to_mask.values()), list(eval_cfg.data.to_mask.keys())))
		tokens_roles = { k : v.strip('[]') for k, v in tokens_roles.items() }
		summary['role_position'] = [tokens_roles[token] + '_position' for token in summary['exp_token']]
		
		# Get formatting for linear positions instead of expected tokens
		summary = summary.sort_values(['sentence_type', 'sentence_num', 'focus'])
		summary['position_num'] = summary.groupby(['sentence_num', 'sentence_type'])['focus'].cumcount() + 1
		summary['position_num'] = ['position_' + str(num) for num in summary['position_num']]
		summary = summary.sort_index()
		
		# Add the actual sentences to the summary
		sentences_with_types = tuple(zip(*[tuple(zip(sentence_types, s_tuples)) for s_tuples in sentences]))
		
		sentences_with_types = [(i, *sentence) 
			for s_type in sentences_with_types 
			for i, sentence in enumerate(s_type)
		]
		
		sentences_df = pd.DataFrame()
		sentences_df['sentence_num'] = [t[0] for t in sentences_with_types]
		sentences_df['sentence_type'] = [t[1] for t in sentences_with_types]
		sentences_df['sentence'] = [t[2] for t in sentences_with_types]
		
		summary = summary.merge(sentences_df, how = 'left')
		
		summary = summary.drop(
			['exp_logit', 'logit', 'token', 'exp_token', 'focus'], axis = 1
		)
		
		# Add a unique model id to the summary as well to facilitate comparing multiple runs
		# The ID comes from the runtime of the model to ensure that it matches when the model is evaluated on different data sets
		model_id = os.path.normpath(os.getcwd()).split(os.sep)[-2]
		summary.insert(0, 'model_id', model_id)
		
		eval_data = eval_cfg.data.name.split('.')[0]
		summary['eval_data'] = eval_data
		summary['model_name'] = self.model_bert_name
		summary['masked'] = self.masked
		summary['masked_tuning_style'] = self.masked_tuning_style
			
		summary['tuning'] = self.cfg.tuning.name
		
		return summary
	
	# deprecated
	"""def get_entailed_summary(self, outputs, labels, eval_cfg: DictConfig):
	
		#Returns a dictionary of the form
		#{"{role}_position" : {"{sentence_type}" : { "{token1}/{token2}" : (n : tensor(...))}}} where 
		#role is the thematic role associated with token1,
		#sentence_type ranges over the sentence_types specified in the config,
		#token2 ranges over all tokens other than token1,
		#{token1}/{token2} is the log odds of predicting token1 compared to token2 in {role}_position,
		#and n is the index of the sentence in the testing data
		
		sentence_types = eval_cfg.data.sentence_types
		
		# Get the log probabilities for each word in the sentences
		sentence_type_logprobs = {}

		for output, sentence_type in tuple(zip(outputs, sentence_types)):
			sentence_type_logprobs[sentence_type] = torch.nn.functional.log_softmax(output.logits, dim = 2)

		# Get the positions of the tokens in each sentence of each type
		tokens_indices = dict(zip(
			self.tokens_to_mask.keys(), 
			self.tokenizer.convert_tokens_to_ids(list(self.tokens_to_mask.values()))
		))
		
		# Get the expected positions for each token in the eval data
		token_foci = {}

		for token in tokens_indices:
			token_foci[token] = {}
			for sentence_type, label in tuple(zip(sentence_types, labels)):
			if (indices := torch.where(label == tokens_indices[token])[1]).nelement() != 0:
				token_foci[token][sentence_type] = indices

			if token_foci[token] == {}: del token_foci[token]

		# Get odds for each role, sentence type, and token
		odds = {}
		for token_focus in token_foci:
			odds[token_focus + '_position'] = {}
			for sentence_type in token_foci[token_focus]:
			odds[token_focus + '_position'][sentence_type] = {}
			for token in tokens_indices:
				indices_logprobs = tuple(zip(token_foci[token_focus][sentence_type], sentence_type_logprobs[sentence_type][:,:,tokens_indices[token]]))
				odds[token_focus + '_position'][sentence_type][token] = torch.tensor([logprobs[idx] for idx, logprobs in indices_logprobs])

		# Get the odds ratio of each token compared to every other token in the correct position
		odds_ratios = {}
		for position in odds:
			odds_ratios[position] = {}
			for sentence_type in odds[position]:
			odds_ratios[position][sentence_type] = {}
			current_tokens = list(odds[position][sentence_type])
			current_pairs = [(token1, token2) for token1 in current_tokens for token2 in current_tokens if token1 != token2 and token1 == position.strip('_position')]
			for pair in current_pairs:
				odds_ratios[position][sentence_type][f'{pair[0]}/{pair[1]}'] = odds[position][sentence_type][pair[0]] - odds[position][sentence_type][pair[1]]

		# Relabel the summary keys to reflect intended roles rather than tokens
		tokens_roles = dict(zip(list(eval_cfg.data.to_mask.values()), list(eval_cfg.data.to_mask.keys())))
		for token in tokens_roles:
			tokens_roles[token] = tokens_roles[token].strip('[]')

		summary = { tokens_roles[position.replace('_position', '')] + '_position' : odds_ratios[position] for position in odds_ratios }

		# Reformat to organize by linear position instead of expected token
		sentence_positions = {}
		for sentence_type in sentence_types:
			for token in tokens_indices:
			if token in token_foci:
				if sentence_type in token_foci[token]:
				if not sentence_type in sentence_positions:
					sentence_positions[sentence_type] = {token : token_foci[token][sentence_type]}
				else:
					sentence_positions[sentence_type].update({token : token_foci[token][sentence_type]})

		position_foci = {}
		ordered_positions = {}
		for sentence_type in sentence_positions:
			ordered_positions[sentence_type] = { k : v 
			for k, v in sorted(
				sentence_positions[sentence_type].items(),
				key = lambda item: item[1][0]
			)}
			
			ordered_positions = {
			f'position_{i+1}' : {
				sentence_type : {
				list(ordered_positions[sentence_type].keys())[i] :
				ordered_positions[sentence_type][list(ordered_positions[sentence_type].keys())[i]]
				}
			}
			for i in range(len(ordered_positions[sentence_type]))
			}
			
			for position in ordered_positions:
			if not position in position_foci:
				position_foci[position] = ordered_positions[position]
			else:
				position_foci[position].update(ordered_positions[position])

		# Get odds for each linear position, sentence type, and token
		odds_arg_positions = {}
		for position in position_foci:
			odds_arg_positions[position] = {}
			for sentence_type in position_foci[position]:
			odds_arg_positions[position][sentence_type] = {}
			for token in tokens_indices:
				indices_logprobs = tuple(zip(
				*list(position_foci[position][sentence_type].values()), 
				sentence_type_logprobs[sentence_type][:,:,tokens_indices[token]]
				))
				odds_arg_positions[position][sentence_type][token] = torch.tensor([logprobs[idx] for idx, logprobs in indices_logprobs])
				odds_arg_positions[position][sentence_type]['expected'] = list(position_foci[position][sentence_type].keys())[0]

		# Get the odds ratio of each token compared to every other token in each position
		odds_ratios_arg_positions = {}
		for position in odds_arg_positions:
			odds_ratios_arg_positions[position] = {}
			for sentence_type in odds_arg_positions[position]:
			odds_ratios_arg_positions[position][sentence_type] = {}
			current_tokens = [item for item in list(odds_arg_positions[position][sentence_type]) if not item == 'expected']
			current_pairs = [(token1, token2) for token1 in current_tokens for token2 in current_tokens if token1 != token2]
			for pair in current_pairs:
				odds_ratios_arg_positions[position][sentence_type][f'{pair[0]}/{pair[1]}'] = odds_arg_positions[position][sentence_type][pair[0]] - odds_arg_positions[position][sentence_type][pair[1]]
			
			odds_ratios_arg_positions[position][sentence_type]['expected'] = odds_arg_positions[position][sentence_type]['expected']

		summary = {
			'by_roles' : summary,
			'by_positions' : odds_ratios_arg_positions
		}

		summary['by_model'] = {}

		for by_x in summary:
			if not by_x == 'by_model':
			summary['by_model'][by_x] = {}
			for position in summary[by_x]:
				summary['by_model'][by_x][position] = {}
				for sentence_type in summary[by_x][position]:
				summary['by_model'][by_x][position][sentence_type] = {}
				for odds_ratio in summary[by_x][position][sentence_type]:
					if not odds_ratio == 'expected':
					summary['by_model'][by_x][position][sentence_type][odds_ratio] = {
						'mean' : torch.mean(summary[by_x][position][sentence_type][odds_ratio]),
						'std'	: torch.std(summary[by_x][position][sentence_type][odds_ratio])
					}
					else:
					summary['by_model'][by_x][position][sentence_type]['expected'] = summary[by_x][position][sentence_type][odds_ratio]

		return summary"""
	
	def graph_entailed_results(self, summary, eval_cfg: DictConfig, axis_size = 8, multi = False, pt_size = 24):
		
		if multi:
			summary['odds_ratio'] = summary['mean']
			summary = summary.drop('mean', axis = 1)
		
		# Get each unique pair of sentence types so we can create a separate plot for each pair
		sentence_types = summary['sentence_type'].unique()
		paired_sentence_types = list(itertools.combinations(sentence_types, 2))
		paired_sentence_types = [sorted(pair) for pair in paired_sentence_types]

		# Set colors for every unique odds ratio we are plotting
		all_ratios = summary['ratio_name'].unique()
		colors = dict(zip(all_ratios, ['teal', 'r', 'forestgreen', 'darkorange', 'indigo', 'slategray']))

		# For each pair, we create a different plot
		for pair in paired_sentence_types:
			
			# Get x and y data. We plot the first member of each pair on x, and the second member on y
			x_data = summary[summary['sentence_type'] == pair[0]].reset_index(drop = True)
			y_data = summary[summary['sentence_type'] == pair[1]].reset_index(drop = True)
			
			# Filter data to only odds ratios that exist in both sentence types
			common_odds = set(x_data.ratio_name).intersection(y_data.ratio_name)
			x_data = x_data[x_data['ratio_name'].isin(common_odds)].reset_index(drop = True)
			y_data = y_data[y_data['ratio_name'].isin(common_odds)].reset_index(drop = True)
			
			if not multi:
				lim = np.max(np.abs([*x_data['odds_ratio'].values, *y_data['odds_ratio'].values])) + 0.5
			else:
				x_odds = np.abs(x_data['odds_ratio'].values) + x_data['sem'].values
				y_odds = np.abs(y_data['odds_ratio'].values) + y_data['sem'].values
				lim = np.max([*x_odds, *y_odds]) + 0.5
							
			# Construct get number of linear positions (if there's only one position, we can't make plots by linear position)
			ratio_names_positions = x_data[['ratio_name', 'position_num']].drop_duplicates().reset_index(drop = True)
			ratio_names_positions = list(ratio_names_positions.to_records(index = False))
			ratio_names_positions = sorted(ratio_names_positions, key = lambda x: int(x[1].replace('position_', '')))
			
			if len(ratio_names_positions) > 1:
				fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
			else:
				fig, (ax1, ax2) = plt.subplots(1, 2)
			
			ax1.set_xlim(-lim, lim)
			ax1.set_ylim(-lim, lim)
			
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
				
				if multi:
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
			
			ax1.legend()
			
			# Construct plot of confidence differences (a measure of transference)
			ax2.set_xlim(-lim, lim)
			if not multi:
				ylim_diffs = np.max(np.abs([*x_data.odds_ratio.values, *(y_data.odds_ratio - x_data.odds_ratio).values])) + 0.5
			else:
				x_odds = np.abs(x_data.odds_ratio.values) + x_data['sem'].values
				y_odds = np.abs(y_data.odds_ratio - x_data.odds_ratio) + y_data['sem'].values
				ylim_diffs = np.max([*x_odds, *y_odds]) + 0.5 
				
			ax2.set_ylim(-ylim_diffs, ylim_diffs)
			
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
				
				if multi:
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
			
			ax2.legend()
			
			# Construct plots by linear position if possible
			if len(ratio_names_positions) > 1:
				ax3.set_xlim(-lim, lim)
				ax3.set_ylim(-lim, lim)
				
				xlabel = [f'Confidence in {pair[0]} sentences']
				ylabel = [f'Confidence in {pair[1]} sentences']
				
				# For every position in the summary, plot each odds ratio
				for ratio_name, position in ratio_names_positions:
					x_idx = np.where(x_data.position_num == position)[0]
					y_idx = np.where(y_data.position_num == position)[0]
					
					x_expected_token = x_data.loc[x_idx].ratio_name.unique()[0].split('/')[0]
					y_expected_token = y_data.loc[y_idx].ratio_name.unique()[0].split('/')[0]
					position_label = position.replace('position_', 'position ')
					
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
					
					if multi:
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
				
				ax3.legend()
				
				# Construct plot of confidence differences by linear position
				ax4.set_xlim(-lim, lim)
				ylim_diffs = 0
				
				xlabel = [f'Confidence in {pair[0]} sentences']
				ylabel = [f'Overconfidence in {pair[1]} sentences']
				
				# For every position in the summary, plot each odds ratio
				for ratio_name, position in ratio_names_positions:
					x_idx = np.where(x_data.position_num == position)[0]
					y_idx = np.where(y_data.position_num == position)[0]
					
					x_expected_token = x_data.loc[x_idx].ratio_name.unique()[0].split('/')[0]
					y_expected_token = y_data.loc[y_idx].ratio_name.unique()[0].split('/')[0]
					position_label = position.replace('position_', 'position ')
					
					xlabel.append(f"Expected {x_expected_token} in {position_label}")
					ylabel.append(f"Expected {y_expected_token} in {position_label}")
					
					x = x_data.odds_ratio[x_idx].reset_index(drop = True)
					y = y_data.odds_ratio[y_idx].reset_index(drop = True)
					
					if not x_expected_token == y_expected_token:
						y = -y
					
					y = y - x
					
					if not multi:
						ylim_diffs = np.max([ylim_diffs, np.max(np.abs([*x, *y])) + 0.5])
					else:
						x_odds = np.abs(x.values) + x_data['sem'][x_idx].values
						y_odds = np.abs(y.values) + y_data['sem'][y_idx].values
						ylim_diffs = np.max([ylim_diffs, np.max([*x_odds, *y_odds]) + 0.5])
					
					color_map = x_data[x_data['position_num'] == position].ratio_name.map(colors)
					
					ax4.scatter(
						x = x, 
						y = y,
						c = color_map,
						label = f'{ratio_name} in {position_label}',
						s = pt_size
					)
					
					if multi:
						ax4.errorbar(
							x = x, 
							xerr = x_data['sem'][x_idx],
							y = y,
							yerr = y_data['sem'][y_idx],
							ecolor = color_map,
							ls = 'none'
						)
				
				ax4.set_ylim(-ylim_diffs, ylim_diffs)
				ax4.plot((-lim, lim), (0, 0), linestyle = '--', color = 'k', scalex = False, scaley = False)
				
				ax4.set_aspect(1.0/ax4.get_data_ratio(), adjustable = 'box')
				
				xlabel = '\n'.join(xlabel)
				ylabel = '\n'.join(ylabel)
				
				ax4.set_xlabel(xlabel, fontsize = axis_size)
				ax4.set_ylabel(ylabel, fontsize = axis_size)
				
				ax4.legend()
			
			# Set title
			title = re.sub(r"\' (.*\s)", f"' {', '.join(pair)} ", eval_cfg.data.description.replace('tuples', 'pairs'))
			fig.suptitle(title)
			
			# Save plot
			if len(ratio_names_positions) > 1:
				fig.set_size_inches(8, 8)
			else:
				fig.set_size_inches(8, 4)
			
			fig.tight_layout()
			dataset_name = eval_cfg.data.name.split('.')[0]
			plt.savefig(f"{dataset_name}-{pair[0]}-{pair[1]}-paired.png")
			plt.close()
	
	# deprecated
	"""def graph_entailed_results(self, summary, eval_cfg: DictConfig):

		# Get each unique pair of sentence types so we can create a separate plot for each pair
		sentence_types = get_unique([[sentence_type for sentence_type in summary['by_roles'][position]] for position in summary['by_roles']])
		paired_sentence_types = list(itertools.combinations(sentence_types, 2))

		# For each pair, we create a different plot
		for pair in paired_sentence_types:
			
			# Get and set x and y limits
			x_lim = np.max([
			np.array(
				[torch.max(torch.abs(summary['by_roles'][position][pair[0]][odds_ratio])) for odds_ratio in summary['by_roles'][position][pair[0]]]
			) 
			for position in summary['by_roles'] if pair[0] in summary['by_roles'][position]
			])

			y_lim = np.max([
			np.array(
				[torch.max(torch.abs(summary['by_roles'][position][pair[1]][odds_ratio])) for odds_ratio in summary['by_roles'][position][pair[1]]]
			) 
			for position in summary['by_roles'] if pair[1] in summary['by_roles'][position]
			])

			lim = np.max([x_lim + 0.5, y_lim + 0.5])
		
			fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

			ax1.set_xlim(-lim, lim)
			ax1.set_ylim(-lim, lim)

			# Set colors for every unique odds ratio we are plotting
			all_ratios = get_unique(
			[[list(summary['by_roles'][position][sentence_type].keys()) for sentence_type in summary['by_roles'][position]] for position in summary['by_roles']]
			)

			colors = dict(zip(all_ratios, ['teal', 'r', 'forestgreen', 'darkorange', 'indigo', 'slategray']))

			# For every position in the summary, plot each odds ratio
			for role_position in summary['by_roles']:
			for odds_ratio in summary['by_roles'][role_position][pair[0]]:
				if pair[0] in summary['by_roles'][role_position] and pair[1] in summary['by_roles'][role_position]:
				ax1.scatter(
					x = summary['by_roles'][role_position][pair[0]][odds_ratio].tolist(), 
					y = summary['by_roles'][role_position][pair[1]][odds_ratio].tolist(),
					c = colors[odds_ratio],
					label = f'{odds_ratio} in {role_position.replace("_position", " position")}'
				)

			# Draw a diagonal to represent equal performance in both sentence types
			ax1.plot((-lim, lim), (-lim, lim), linestyle = '--', color = 'k', scalex = False, scaley = False)

			ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable = 'box')
			
			# Set labels and title
			ax1.set_xlabel(f"Confidence in {pair[0]} sentences")
			ax1.set_ylabel(f"Confidence in {pair[1]} sentences")
			
			ax1.legend()

			# Construct plot of confidence differences (a measure of transference)
			ax2.set_xlim(-lim, lim)
			ax2.set_ylim(-lim, lim)

			for role_position in summary['by_roles']:
			for odds_ratio in summary['by_roles'][role_position][pair[0]]:
				if pair[0] in summary['by_roles'][role_position] and pair[1] in summary['by_roles'][role_position]:
				x = summary['by_roles'][role_position][pair[0]][odds_ratio]
				y = summary['by_roles'][role_position][pair[1]][odds_ratio]
				y = y - x
				ax2.scatter(
					x = x.tolist(),
					y = y.tolist(),
					c = colors[odds_ratio],
					label = f'{odds_ratio} in {role_position.replace("_position", " position")}'
				)

			# Draw a line at zero to represent equal performance in both sentence types
			ax2.plot((-lim, lim), (0, 0), linestyle = '--', color = 'k', scalex = False, scaley = False)

			ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable = 'box')
			
			# Set labels and title
			ax2.set_xlabel(f"Confidence in {pair[0]} sentences")
			ax2.set_ylabel(f"Overconfidence in {pair[1]} sentences")
			
			ax2.legend()

			# Construct plot of odds ratios by position
			ax3.set_xlim(-lim, lim)
			ax3.set_ylim(-lim, lim)

			xlabel = [f'Confidence in {pair[0]} sentences']
			ylabel = [f'Confidence in {pair[1]} sentences']

			all_positions = list(summary['by_positions'])

			all_ratios_by_position = [list(zip(p, all_positions)) for p in itertools.permutations(all_ratios, len(all_positions))]
			all_ratios_by_position = [' in '.join(i) for l in all_ratios_by_position for i in l]
			all_ratios_by_position.sort()

			colors = dict(zip(all_ratios_by_position, ['teal', 'r', 'forestgreen', 
										 'darkorange', 'indigo', 'slategray',
										 'peru', 'chartreuse', 'deepskyblue']))

			# For every position in the summary, plot each odds ratio
			for position in summary['by_positions']:
			xlabel.append(f"Expected {summary['by_positions'][position][pair[0]]['expected']} in {position.replace('_', ' ')}")
			ylabel.append(f"Expected {summary['by_positions'][position][pair[1]]['expected']} in {position.replace('_', ' ')}")
			for odds_ratio in summary['by_positions'][position][pair[0]]:
				if not odds_ratio == 'expected':
				if pair[0] in summary['by_positions'][position] and pair[1] in summary['by_positions'][position]:
					ax3.scatter(
					x = summary['by_positions'][position][pair[0]][odds_ratio].tolist(), 
					y = summary['by_positions'][position][pair[1]][odds_ratio].tolist(),
					c = colors[f'{odds_ratio} in {position}'],
					label = f'{odds_ratio} in {position.replace("position_", " position ")}'
					)

			ax3.plot((-lim, lim), (-lim, lim), linestyle = '--', color = 'k', scalex = False, scaley = False)

			ax3.set_aspect(1.0/ax3.get_data_ratio(), adjustable = 'box')

			xlabel = '\n'.join(xlabel)
			ylabel = '\n'.join(ylabel)

			ax3.set_xlabel(xlabel)
			ax3.set_ylabel(ylabel)

			ax3.legend(prop = {'size' : 8})

			# Construct plot of overconfidence by position
			xlim = torch.tensor(0)
			ylim = torch.tensor(0)

			xlabel = [f'Confidence in {pair[0]} sentences']
			ylabel = [f'Overconfidence in {pair[1]} sentences']

			# For every position in the summary, plot each odds ratio
			for position in summary['by_positions']:
			xlabel.append(f"Expected {summary['by_positions'][position][pair[0]]['expected']} in {position.replace('_', ' ')}")
			ylabel.append(f"Expected {summary['by_positions'][position][pair[1]]['expected']} in {position.replace('_', ' ')}")
			for odds_ratio in summary['by_positions'][position][pair[0]]:
				if not odds_ratio == 'expected' and odds_ratio.startswith(summary['by_positions'][position][pair[0]]['expected']):
				if pair[0] in summary['by_positions'][position] and pair[1] in summary['by_positions'][position]:
					x = summary['by_positions'][position][pair[0]][odds_ratio]
					y = summary['by_positions'][position][pair[1]][odds_ratio]
					y = y - x
					xlim = torch.max(xlim, torch.max(torch.abs(x)))
					ylim = torch.max(ylim, torch.max(torch.abs(y)))
					ax4.scatter(
					x = x.tolist(), 
					y = y.tolist(),
					c = colors[f'{odds_ratio} in {position}'],
					label = f'{odds_ratio} in {position.replace("position_", " position ")}'
					)

			lim = torch.max(xlim, ylim)
			lim += 0.5

			ax4.set_xlim(-lim, lim)
			ax4.set_ylim(-lim, lim)

			ax4.plot((-lim, lim), (0, 0), linestyle = '--', color = 'k', scalex = False, scaley = False)

			ax4.set_aspect(1.0/ax3.get_data_ratio(), adjustable = 'box')

			xlabel = '\n'.join(xlabel)
			ylabel = '\n'.join(ylabel)

			ax4.set_xlabel(xlabel)
			ax4.set_ylabel(ylabel)

			ax4.legend(prop = {'size' : 8})

			title = re.sub(r"\' (.*\s)", f"' {', '.join(pair)} ", eval_cfg.data.description.replace('tuples', 'pairs'))
			fig.suptitle(title)

			# Save plot
			dataset_name = eval_cfg.data.name.split('.')[0]
			fig.set_size_inches(8, 8)
			fig.tight_layout()
			plt.savefig(f"{dataset_name}-{pair[0]}-{pair[1]}-paired.png")
			plt.close()"""
	
	def eval_entailments(self, eval_cfg: DictConfig, checkpoint_dir: str):
		"""
		Computes model performance on data consisting of 
			sentence 1 , sentence 2 , [...]
		where credit for a correct prediction on sentence 2[, 3, ...] is contingent on
		also correctly predicting sentence 1
		"""
		print(f"SAVING TO: {os.getcwd()}")
		
		# Load model
		log.info("Loading model from disk")
		model_path = os.path.join(checkpoint_dir, "model.pt")
		self.model.load_state_dict(torch.load(model_path))
		self.model.eval()

		# Load model
		data = self.load_eval_multi_file(eval_cfg.data.name, eval_cfg.data.to_mask)
		inputs = data["inputs"]
		labels = data["labels"]
		sentences = data["sentences"]

		assert len(inputs) == len(labels), f"Inputs (size {len(inputs)}) must match labels (size {len(labels)}) in length"

		# Calculate performance on data
		with torch.no_grad():

			log.info("Evaluating model on testing data")

			outputs = []

			for i in inputs:
				output = self.model(**i)
				outputs.append(output)

			summary = self.get_entailed_summary(sentences, outputs, labels, eval_cfg)
			for (ratio_name, sentence_type), summary_slice in summary.groupby(['ratio_name', 'sentence_type']):
				odds_ratios = [round(float(o_r), 2) for o_r in list(summary_slice.odds_ratio.values)]
				role_position = summary_slice.role_position.unique()[0].replace('_' , ' ')
				print(f'\nLog odds of {ratio_name} in {role_position} in {sentence_type}s:\n\t{odds_ratios}')
			
			print('')
				
			dataset_name = eval_cfg.data.name.split('.')[0]
			summary.to_pickle(f"{dataset_name}-scores.pkl")
			
			summary_csv = summary.copy()
			summary_csv['odds_ratio'] = summary_csv['odds_ratio'].astype(float).copy()
			summary_csv.to_csv(f"{dataset_name}-scores.csv", index = False)

			self.graph_entailed_results(summary, eval_cfg)
	
	def eval(self, eval_cfg: DictConfig, checkpoint_dir: str):
		
		# Load model from disk
		log.info("Loading model from disk")
		model_path = os.path.join(checkpoint_dir, "model.pt")
		self.model.load_state_dict(torch.load(model_path))
		self.model.eval()
		
		# Load data
		inputs, labels, sentences = self.load_eval_data_file(eval_cfg.data.name, eval_cfg.data.to_mask)
		
		# Calculate results on given data
		with torch.no_grad():
			
			log.info("Evaluating model on testing data")
			outputs = self.model(**inputs)
			
			results = self.collect_results(inputs, eval_cfg.data.eval_groups, outputs)
			summary = self.summarize_results(results, labels)
			
			log.info("Creating graphs")
			self.graph_results(results, summary, eval_cfg)
	
	def tune(self):
		"""
		Fine-tunes the model on the provided tuning data. Saves model state to disk.
		"""
		if self.tokens_to_mask:
			# randomly initialize the embeddings of the novel tokens we care about
			# to provide some variablity in model tuning
			model_e_dim = getattr(
				self.model, 
				self.model_bert_name
			).embeddings.word_embeddings.embedding_dim
			num_new_tokens = len(list(self.tokens_to_mask.keys()))
			new_embeds = torch.nn.Embedding(
				num_new_tokens, 
				model_e_dim
			)
			
			with torch.no_grad():
			
				if self.model_bert_name != 'roberta':
					unused_embedding_weights = getattr(
						self.model, 
						self.model_bert_name
					).embeddings.word_embeddings.weight[range(0,999), :]
				else:
					unused_embedding_weights = getattr(
						self.model,
						self.model_bert_name
					).embeddings.word_embeddings.weight[range(50261,50263), :]
				
				std, mean = torch.std_mean(unused_embedding_weights)
				log.info(f"Initializing unused tokens with random data drawn from N({mean:.2f}, {std:.2f})")
				
				# These are experimentally determined values to match the
				# default embedding weights of BERT's unused vocab items
				torch.nn.init.normal_(new_embeds.weight, mean=mean, std=std)
					
				for i, key in enumerate(self.tokens_to_mask):
					tok = self.tokens_to_mask[key]
					tok_id = self.tokenizer(tok, return_tensors="pt")["input_ids"][:,1]
					
				getattr(
					self.model, 
					self.model_bert_name
				).embeddings.word_embeddings.weight[tok_id, :] = new_embeds.weight[i,:]
				
			self.old_embeddings = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.clone()

			log.info(f"Freezing model parameters")
			# Freeze parameters
			for name, param in self.model.named_parameters():
				if 'word_embeddings' not in name:
					param.requires_grad = False
					
			for name, param in self.model.named_parameters():
				if param.requires_grad:
					assert 'word_embeddings' in name, f"{name} is not frozen!"

		if not self.tuning_data:
			log.info("Saving model state dictionary.")
			torch.save(self.model.state_dict(), "model.pt")
			return
		
		log.info(f"Training model @ '{os.getcwd()}'")

		# Collect Hyperparameters
		lr = self.cfg.hyperparameters.lr
		epochs = self.cfg.hyperparameters.epochs
		optimizer = torch.optim.AdamW(
			self.model.parameters(), 
			lr=lr,
			weight_decay=0
		)

		writer = SummaryWriter()

		# Construct inputs, labels
		# If masked and using simple tuning, or not masking, construct the inputs only once to save time
		# Otherwise, we do it in the loop since it should be random each time
		if self.cfg.hyperparameters.masked and self.masked_tuning_style == 'always':
			inputs = self.tokenizer(self.masked_tuning_data, return_tensors="pt", padding=True)
		elif not self.cfg.hyperparameters.masked:
			inputs = self.tokenizer(self.tuning_data, return_tensors="pt", padding=True)

		labels = self.tokenizer(self.tuning_data, return_tensors="pt", padding=True)["input_ids"]

		self.model.train()

		set_seed(42)

		log.info("Fine-tuning model")

		with trange(epochs) as t:
			for epoch in t:

				optimizer.zero_grad()

				if self.cfg.hyperparameters.masked and self.masked_tuning_style == 'bert':
					inputs = self.tokenizer(
						self.mixed_tuning_data, 
						return_tensors="pt", padding=True
					)

				# Compute loss
				outputs = self.model(**inputs, labels=labels)
				loss = outputs.loss
				t.set_postfix(loss=loss.item())
				loss.backward()
				
				# Log results
				writer.add_scalar(f"loss/{self.model_bert_name}", loss, epoch)
				masked_input = self.tokenizer(
					self.masked_tuning_data, 
					return_tensors="pt", 
					padding=True
				)
				results = self.collect_results(masked_input, self.tokens_to_mask, outputs)
				
				sent_key = list(results.keys())[0]
				pos_key = list(results[sent_key].keys())[0]
				spec_results = results[sent_key][pos_key]["mean grouped log_probability"]

				for key in spec_results:
					writer.add_scalar(f"{key} LogProb/{self.model_bert_name}", spec_results[key], epoch)

				# GRADIENT ADJUSTMENT
				# 
				# The word_embedding remains unfrozen, but we only want to update
				# the embeddings of the novel tokens. To do this, we zero-out
				# all gradients except for those at these token indices.

				# Copy gradients of the relevant tokens
				nz_grad = {}
				for key in self.tokens_to_mask:
					
					tok = self.tokens_to_mask[key]
					tok_id = self.tokenizer(tok, return_tensors="pt")["input_ids"][:,1]
					grad = getattr(
						self.model, 
						self.model_bert_name
					).embeddings.word_embeddings.weight.grad[tok_id, :].clone()
					nz_grad[tok_id] = grad
				
				
				# Zero out all gradients of word_embeddings in-place
				getattr(
					self.model, 
					self.model_bert_name
				).embeddings.word_embeddings.weight.grad.data.fill_(0)

				# print(optimizer)
				# raise SystemExit

				# Replace the original gradients at the relevant token indices
				for key in nz_grad:
					getattr(
						self.model, 
						self.model_bert_name
					).embeddings.word_embeddings.weight.grad[key, :] = nz_grad[key]

				optimizer.step()
				
				# Check that we changed the correct number of parameters
				new_embeddings = getattr(
					self.model, 
					self.model_bert_name
				).embeddings.word_embeddings.weight.clone()
				sim = torch.eq(self.old_embeddings, new_embeddings)
				changed_params = int(list(sim.all(dim=1).size())[0]) - sim.all(dim=1).sum().item()
				
				exp_ch = len(list(self.tokens_to_mask.keys()))
				assert changed_params == exp_ch, f"Exactly {exp_ch} embeddings should have been updated, but {changed_params} were!"
				
		log.info("Saving model state dictionary.")
		torch.save(self.model.state_dict(), "model.pt")

		writer.flush()
		writer.close()
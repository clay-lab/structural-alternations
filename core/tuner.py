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
import random
import logging
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns

from tqdm import trange
from typing import Dict, List
from omegaconf import DictConfig, OmegaConf
from PyPDF2 import PdfFileMerger, PdfFileReader
from transformers import logging as lg
from transformers import DistilBertForMaskedLM, DistilBertTokenizer, RobertaForMaskedLM, RobertaTokenizer, BertForMaskedLM, BertTokenizer

lg.set_verbosity_error()

log = logging.getLogger(__name__)

def set_seed(seed):
	seed = int(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	
def strip_punct(sentence):
	return re.sub(r'[^\[\]\<\>\w\s,]', '', sentence)

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
	def mask_tok(self):
		return self.tokenizer.mask_token
	
	@property
	def mask_tok_id(self):
		return self.tokenizer(self.mask_tok, return_tensors="pt")["input_ids"][:,1]
	
	@property
	def string_id(self) -> str:
		return self.cfg.model.string_id
	
	@property
	def reference_sentence_type(self):
		return self.cfg.tuning.reference_sentence_type
	
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
			if self.cfg.hyperparameters.strip_punct:
				s = strip_punct(s)
				
			for key in self.tokens_to_mask:
				s = s.replace(key, self.tokens_to_mask[key])
			
			data.append(s)
		
		return [d.lower() for d in data]
	
	@property
	def mixed_tuning_data(self) -> List[str]:
		to_mix = self.verb_tuning_data['data'] if self.cfg.tuning.new_verb else self.tuning_data
		
		data = []
		for s in to_mix:
			if self.cfg.hyperparameters.strip_punct:
				s = strip_punct(s)
			
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
		to_mask = self.verb_tuning_data['data'] if self.cfg.tuning.new_verb else self.tuning_data
		
		data = []
		for s in to_mask:
			if self.cfg.hyperparameters.strip_punct:
				s = strip_punct(s)
			for val in list(self.tokens_to_mask.values()):
				s = s.replace(val.lower(), self.mask_tok)
			
			data.append(s)
		
		return data
	
	@property
	def verb_tuning_data(self):
		to_replace = self.cfg.tuning.args
		
		manual_to_replace = {
			arg : value
			for arg, value in self.cfg.tuning.args.items()
				if not isinstance(value, int)
		}
		
		auto_to_replace = {
			arg : value
			for arg, value in self.cfg.tuning.args.items()
				if isinstance(value, int)
		}
		
		# Get all combinations of replacement values so we can generate all combinations with manual replacements
		args, values = zip(*manual_to_replace.items())
		replacement_combinations = itertools.product(*list(manual_to_replace.values()))
		manual_to_replace_dicts = [dict(zip(args, t)) for t in replacement_combinations]
		
		data = []
		for d in manual_to_replace_dicts:
			for sentence in self.tuning_data:
				if self.cfg.hyperparameters.strip_punct:
					s = strip_punct(s)
				
				for arg, value in d.items():
					sentence = sentence.replace(arg, value)
				
				data.append(sentence)
		
		if auto_to_replace:
			if len(auto_to_replace) > 1:
				raise ValueError(f"Error: only one token can be set automatically, but you are trying to set {len(to_replace)} tokens!")
				sys.exit(1)
			
			token_to_replace = list(auto_to_replace.keys())[0]
			num_to_replace = list(auto_to_replace.values())[0]
			
			fill_data = [sentence.replace(token_to_replace, self.mask_tok) for sentence in data]
			
			from transformers import pipeline
			
			filler = pipeline('fill-mask', model = self.string_id, top_k = num_to_replace)
			
			# we currently choose the top n distinct nouns that achieve the highest scores in individual cases
			# we could also choose the n most frequent nouns regardless of probability
			# not sure what the best option is
			# most frequent seems to give the highest overall in the current test case, but its close
			#### it turns out that this doesn't give us the desired behavior: the models learn A LOT about
			#### the subject nouns, so we need to rethink what to do with this
			best_nouns = [filler(sentence) for sentence in fill_data]
			best_nouns = [noun for sublist in best_nouns for noun in sublist]
			
			# Filter out instances where we've chosen a word we're already using for replacement
			# this could be adjusted to be a word not already in the sentence with a bit of work
			best_nouns = [
				best_noun
				for best_noun in best_nouns 
					if not best_noun['token_str'] in [
						item for sublist in list(manual_to_replace.values()) for item in sublist
					]
			]
			
			#best_nouns = sorted(best_nouns, key = lambda noun: -noun['score'])
			best_nouns = [noun['token_str'].strip() for noun in best_nouns]
			best_nouns = { noun : best_nouns.count(noun) for noun in best_nouns }#
			best_nouns = { noun : count for noun, count in sorted(best_nouns.items(), key = lambda noun: -noun[1])}#
			best_nouns = list(dict.fromkeys(best_nouns))[:num_to_replace]
			
			to_replace[token_to_replace] = best_nouns
			
			# we can do this in one pass since there can only be one automatically set token
			data = [sentence.replace(token_to_replace, noun) for noun in best_nouns for sentence in data]
		
		sentences = [d.lower() for d in data]
		
		# Return the args as well as the sentences, since we need to save them when they were generated automatically
		return {
			'args' : to_replace,
			'data' : sentences
		}
	
	@property
	def tokens_to_mask(self) -> Dict[str,str]:
		if self.model_bert_name != 'roberta':
			return self.cfg.tuning.to_mask
		else:
			if len(self.cfg.tuning.to_mask) > 3:
				raise ValueError("Insufficient unused tokens in RoBERTa vocabulary to train model on more than three novel tokens.")
				sys.exit(1)
			else:
				def atoi(text):
					return int(text) if text.isdigit() else text
				
				def natural_keys(text):
					return[atoi(c) for c in re.split(r'(\d+)', text)]
				
				orig_tokens = list(self.cfg.tuning.to_mask.values())
				orig_tokens.sort(key = natural_keys)
				
				bert_roberta_mapping = dict(zip(
					orig_tokens,
					('madeupword0000', 'madeupword0001', 'madeupword0002')
				))
				
				for token in self.cfg.tuning.to_mask:
					self.cfg.tuning.to_mask[token] = bert_roberta_mapping[self.cfg.tuning.to_mask[token]]
				
				return self.cfg.tuning.to_mask
	
	# END Computed Properties
	
	def __init__(self, cfg: DictConfig, eval: bool = False) -> None:
		
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
			
			log.info(f"Initializing Model: {self.cfg.model.base_class}")
			
			self.model = self.model_class.from_pretrained(
				self.string_id,
				local_files_only=True
			)
			
			if self.tokens_to_mask:
				# randomly initialize the embeddings of the novel tokens we care about
				# to provide some variablity in model tuning
				model_e_dim = getattr(
					self.model, 
					self.model_bert_name
				).embeddings.word_embeddings.embedding_dim
				
				num_new_tokens = len(self.tokens_to_mask)
				
				new_embeds = torch.nn.Embedding(
					num_new_tokens, 
					model_e_dim
				)				
				
				# Add special tokens to roberta tokenizer
				if self.tokenizer.name_or_path == 'roberta-base':
					
					from transformers.tokenization_utils import AddedToken
					
					added_tokens = []
					
					sorted_tokens = { key : value for key, value in sorted(self.tokens_to_mask.items(), key = lambda item: item[1])}
					
					for i, (key, value) in enumerate(sorted_tokens.items()):
						added_token = 'madeupword000' + str(i)
						added_token = AddedToken(added_token, lstrip = True, rstrip = False) if isinstance(added_token, str) else added_token
						added_tokens.append(added_token)
					
					setattr(self.tokenizer, 'additional_special_tokens', added_tokens)
					
					self.tokenizer.add_tokens(
						added_tokens,
						special_tokens = True
					)
				
				# If we are evaluating, there's no need to reinitialize the weights of the unused tokens,
				# since we'll just be loading the saved weight(s)
				if not eval:
					with torch.no_grad():
						# These are experimentally determined values to match the
						# default embedding weights of the models' unused vocab items
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
						
						seed = int(torch.randint(2**32-1, (1,)))
						set_seed(seed)
						log.info(f"Seed set to {seed}")
						
						torch.nn.init.normal_(new_embeds.weight, mean=mean, std=std)
						#log.info(f"First 3 values of randomly initialized weights: {[round(x, 3) for x in new_embeds.weight[:,:3].data[0].numpy().tolist()]}")
							
						for i, key in enumerate(self.tokens_to_mask):
							tok = self.tokens_to_mask[key]
							tok_id = self.tokenizer(tok, return_tensors="pt")["input_ids"][:,1]
							
							getattr(
								self.model, 
								self.model_bert_name
							).embeddings.word_embeddings.weight[tok_id, :] = new_embeds.weight[i,:]
					
					self.old_embeddings = getattr(
						self.model, 
						self.model_bert_name
					).embeddings.word_embeddings.weight.clone()
					
					# Freeze parameters
					log.info(f"Freezing model parameters")
					for name, param in self.model.named_parameters():
						if 'word_embeddings' not in name:
							param.requires_grad = False
					
					for name, param in self.model.named_parameters():
						if param.requires_grad:
							assert 'word_embeddings' in name, f"{name} is not frozen!"
	
	def tune(self):
		"""
		Fine-tunes the model on the provided tuning data. Saves model state to disk.
		"""
		
		# function to return the weight updates so we can save them every epoch
		def get_updated_weights():
			updated_weights = {}
			for key in self.tokens_to_mask:
				tok = self.tokens_to_mask[key]
				tok_id = self.tokenizer(tok, return_tensors='pt')['input_ids'][:,1]
				
				updated_weights[int(tok_id)] = getattr(
					self.model,
					self.model_bert_name	
				).embeddings.word_embeddings.weight[tok_id, :].clone()
				
			return updated_weights
		
		if not self.tuning_data:
			log.info(f'Saving randomly initialized weights')
			with open('weights.pkl', 'wb') as f:
				pkl.dump({ 0 : get_updated_weights()}, f)
			return
		
		# Collect Hyperparameters
		lr = self.cfg.hyperparameters.lr
		epochs = self.cfg.hyperparameters.epochs
		optimizer = torch.optim.AdamW(
			self.model.parameters(), 
			lr=lr,
			weight_decay=0
		)
		
		writer = SummaryWriter()
		
		# Determine what data to use based on the experiment
		if self.cfg.tuning.new_verb:
			inputs_data = self.verb_tuning_data['data'] if not self.cfg.hyperparameters.masked else self.masked_tuning_data
			labels_data = self.verb_tuning_data['data']
			args = { 'args' : self.verb_tuning_data['args'] }
			log.info("Final noun replacements:")
			for arg in args['args']:
				log.info("" + arg + ":\t[" + ", ".join(args['args'][arg]) + "]")
			
			with open('args.yaml', 'w') as outfile:
				outfile.write(OmegaConf.to_yaml(args['args']))
		elif self.cfg.hyperparameters.masked and self.masked_tuning_style == 'always':
			inputs_data = self.masked_tuning_data
			labels_data = self.tuning_data
		elif not self.cfg.hyperparameters.masked:
			inputs_data = self.tuning_data
			labels_data = self.tuning_data
		
		# Construct inputs, labels
		inputs = self.tokenizer(inputs_data, return_tensors="pt", padding=True)
		labels = self.tokenizer(labels_data, return_tensors="pt", padding=True)["input_ids"]
		
		log.info(f"Training model @ '{os.getcwd()}'")
		
		self.model.train()
		
		# Store weights before fine-tuning
		saved_weights = {}
		saved_weights[0] = get_updated_weights()
		
		with trange(epochs) as t:
			current_epoch = 0
			for epoch in t:
				
				optimizer.zero_grad()
				
				# If we are using bert-style masking, get new randomly changed inputs each epoch
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
				
				# store weights of the relevant tokens
				current_epoch += 1
				saved_weights[current_epoch] = get_updated_weights()
				
				# GRADIENT ADJUSTMENT
				# 
				# The word_embedding remains unfrozen, but we only want to update
				# the embeddings of the novel tokens. To do this, we zero-out
				# all gradients except for those at these token indices.
				nz_grad = {}
				for key in self.tokens_to_mask:
					tok = self.tokens_to_mask[key]
					tok_id = self.tokenizer(tok, return_tensors='pt')['input_ids'][:,1]
					grad = getattr(
						self.model,
						self.model_bert_name
					).embeddings.word_embeddings.weight.grad[tok_id,:].clone()
					nz_grad[tok_id] = grad
				
				# Zero out all gradients of word_embeddings in-place
				getattr(
					self.model, 
					self.model_bert_name
				).embeddings.word_embeddings.weight.grad.data.fill_(0)				
				
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
				
		log.info(f"Saving weights for each of {epochs} epochs")
		with open('weights.pkl', 'wb') as f:
			pkl.dump(saved_weights, f)
		
		writer.flush()
		writer.close()
	
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
	
	def restore_weights(self, checkpoint_dir, epoch: int = None):
		weights_path = os.path.join(checkpoint_dir, 'weights.pkl')
		
		with open(weights_path, 'rb') as f:
			weights = pkl.load(f)
		
		if epoch == None:
			epoch = max(weights.keys())
		
		log.info(f'Restoring saved weights from epoch {epoch}/{max(weights.keys())}')
		
		with torch.no_grad():
			for tok_id in weights[epoch]:
				getattr(
					self.model, 
					self.model_bert_name
				).embeddings.word_embeddings.weight[tok_id,:] = weights[epoch][tok_id]
		
		# return the epoch to help if we didn't specify it
		return epoch
	
	
	def eval(self, eval_cfg: DictConfig, checkpoint_dir: str, epoch: int = None):
		
		_ = self.restore_weights(checkpoint_dir, epoch)
		
		self.model.eval()
		
		# Load data
		inputs, labels, sentences = self.load_eval_file(eval_cfg.data.name, eval_cfg.data.to_mask)
		
		# Calculate results on given data
		with torch.no_grad():
			
			log.info("Evaluating model on testing data")
			outputs = self.model(**inputs)
			
			results = self.collect_results(inputs, eval_cfg.data.eval_groups, outputs)
			summary = self.summarize_results(results, labels)
			
			log.info("Creating graphs")
			self.graph_results(results, summary, eval_cfg)
	
	def load_eval_file(self, data_path: str, replacing: Dict[str, str]):
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
	
	
	def eval_entailments(self, eval_cfg: DictConfig, checkpoint_dir: str, epoch: int = None):
		"""
		Computes model performance on data consisting of 
			sentence 1 , sentence 2 , [...]
		where credit for a correct prediction on sentence 2[, 3, ...] is contingent on
		also correctly predicting sentence 1
		"""
		print(f"SAVING TO: {os.getcwd()}")
		
		epoch = self.restore_weights(checkpoint_dir, epoch)
		
		# Load model
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
			summary['eval_epoch'] = epoch
			"""for (ratio_name, sentence_type), summary_slice in summary.groupby(['ratio_name', 'sentence_type']):
				odds_ratios = [round(float(o_r), 2) for o_r in list(summary_slice.odds_ratio.values)]
				role_position = summary_slice.role_position.unique()[0].replace('_' , ' ')
				print(f'\nLog odds of {ratio_name} in {role_position} in {sentence_type}s:\n\t{odds_ratios}')
			
			print('')"""
			
			dataset_name = eval_cfg.data.name.split('.')[0]
			summary.to_pickle(f"{dataset_name}-{epoch}-scores.pkl")
			
			summary_csv = summary.copy()
			summary_csv['odds_ratio'] = summary_csv['odds_ratio'].astype(float).copy()
			summary_csv.to_csv(f"{dataset_name}-{epoch}-scores.csv", index = False)
			
			log.info('Creating plots')
			self.graph_entailed_results(summary, eval_cfg)
	
	def load_eval_entail_file(self, data_path: str, replacing: Dict[str, str]):
		
		resolved_path = os.path.join(
			hydra.utils.get_original_cwd(),
			"data",
			data_path
		)
		
		with open(resolved_path, "r") as f:
			
			raw_input = [line.strip().lower() for line in f]
			sentences = []
			
			if self.cfg.hyperparameters.strip_punct:
				raw_input = [strip_punct(line) for line in raw_input]
			
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
		
		inputs = [self.tokenizer(m, return_tensors="pt", padding=True) for m in masked_transposed]
		labels = [self.tokenizer(s, return_tensors="pt", padding=True)["input_ids"] for s in sentences_transposed]
		sentences = [[s.strip() for s in line.lower().split(',')] for line in raw_input]
		
		return {
			"inputs" : inputs,
			"labels" : labels,
			"sentences" : sentences
		}
	
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
					for row, idx in enumerate(token_summary['focus']):
						row_sentence_num = token_summary['sentence_num'][row]
						
						row_token = token_summary['token'][row]
						idx_row_token = tokens_indices[row_token]
						logits.append(sentence_type_logprobs[sentence_type][row_sentence_num,idx,idx_row_token])
						
						exp_row_token = token_summary['exp_token'][row]
						idx_exp_row_token = tokens_indices[exp_row_token]
						exp_logits.append(sentence_type_logprobs[sentence_type][row_sentence_num,idx,idx_exp_row_token])
					
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
		summary['strip_punct'] = self.cfg.hyperparameters.strip_punct
		
		return summary
	
	def graph_entailed_results(self, summary, eval_cfg: DictConfig, axis_size = 8, multi = False, pt_size = 24):
		
		if multi:
			summary['odds_ratio'] = summary['mean']
			summary = summary.drop('mean', axis = 1)
		
		# Get each unique pair of sentence types so we can create a separate plot for each pair
		sentence_types = summary['sentence_type'].unique()
		paired_sentence_types = list(itertools.combinations(sentence_types, 2))
		
		# Sort so that the trained cases are first
		paired_sentence_types = [
			sorted(pair, 
				   key = lambda x: '0' + x if x == self.reference_sentence_type else '1' + x) 
			for pair in paired_sentence_types
		]
		
		# Filter to only cases including the reference sentence type for ease of interpretation
		paired_sentence_types = [(s1, s2) for s1, s2 in paired_sentence_types if s1 == self.reference_sentence_type]
		
		# Set colors for every unique odds ratio we are plotting
		all_ratios = summary['ratio_name'].unique()
		colors = dict(zip(all_ratios, ['teal', 'r', 'forestgreen', 'darkorange', 'indigo', 'slategray']))
		
		acc_columns = ['s1', 's2', 'both_correct', 'ref_correct_gen_incorrect', 'both_incorrect', 'ref_incorrect_gen_correct', 'ref_correct', 'ref_incorrect', 'gen_correct', 'gen_incorrect', 'num_points']
		acc = pd.DataFrame(columns = acc_columns)
		
		dataset_name = eval_cfg.data.name.split('.')[0]

		# For each pair, we create a different plot
		for pair in paired_sentence_types:
			
			# Get x and y data. We plot the first member of each pair on x, and the second member on y
			x_data = summary[summary['sentence_type'] == pair[0]].reset_index(drop = True)
			y_data = summary[summary['sentence_type'] == pair[1]].reset_index(drop = True)
			
			# Filter data to only odds ratios that exist in both sentence types
			common_odds = set(x_data.ratio_name).intersection(y_data.ratio_name)
			x_data = x_data[x_data['ratio_name'].isin(common_odds)].reset_index(drop = True)
			y_data = y_data[y_data['ratio_name'].isin(common_odds)].reset_index(drop = True)
			
			# Get the number of points in each quadrant
			both_correct = len(x_data[(x_data.odds_ratio > 0) & (y_data.odds_ratio > 0)].odds_ratio)/len(x_data.odds_ratio) * 100
			ref_correct_gen_incorrect = len(x_data[(x_data.odds_ratio > 0) & (y_data.odds_ratio < 0)].odds_ratio)/len(x_data.odds_ratio) * 100
			both_incorrect = len(x_data[(x_data.odds_ratio < 0) & (y_data.odds_ratio < 0)].odds_ratio)/len(x_data.odds_ratio) * 100
			ref_incorrect_gen_correct = len(x_data[(x_data.odds_ratio < 0) & (y_data.odds_ratio > 0)].odds_ratio)/len(x_data.odds_ratio) * 100
			ref_correct = len(x_data[x_data.odds_ratio > 0].odds_ratio)/len(x_data.odds_ratio) * 100
			ref_incorrect = len(x_data[x_data.odds_ratio < 0].odds_ratio)/len(x_data.odds_ratio) * 100
			gen_correct = len(y_data[y_data.odds_ratio > 0].odds_ratio)/len(y_data.odds_ratio) * 100
			gen_incorrect = len(y_data[y_data.odds_ratio < 0].odds_ratio)/len(y_data.odds_ratio) * 100
			num_points = len(x_data)
			
			acc = acc.append(pd.DataFrame(
				[[pair[0], pair[1], 
				  both_correct, ref_correct_gen_incorrect, 
				  both_incorrect, ref_incorrect_gen_correct, 
				  ref_correct, ref_incorrect, 
				  gen_correct, gen_incorrect, 
				  num_points]],
				  columns = acc_columns
			))
			
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
				
			if len(ratio_names_positions) > 1 and not all(x_data.position_num == y_data.position_num):
				fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
				fig.set_size_inches(8, 10)
			else:
				fig, (ax1, ax2) = plt.subplots(1, 2)
				fig.set_size_inches(8, 6)
			
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
			
			# Construct plots by linear position if they'll be different
			if len(ratio_names_positions) > 1 and not all(x_data.position_num == y_data.position_num):
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
			title = re.sub(r"\'\s(.*?)", f"' {', '.join(pair)} ", eval_cfg.data.description.replace('tuples', 'pairs')) + \
				    (' @ epoch ' + str(np.unique(summary.eval_epoch)[0]) if len(np.unique(summary.eval_epoch)) == 1 else  '')
			
			model_name = np.unique(summary.model_name)[0] if len(np.unique(summary.model_name)) == 1 else 'multiple'
			masked_str = 'masked' if all(summary.masked) else 'unmasked' if all(1 - summary.masked) else 'multiple'
			masked_tuning_str = ', Masking type: ' + np.unique(summary.masked_tuning_style[summary.masked])[0] if len(np.unique(summary.masked_tuning_style[summary.masked])) == 1 else ', Masked tuning style: multiple' if any(summary.masked) else ''
			subtitle = f'Model: {model_name} {masked_str}{masked_tuning_str}'
			
			tuning_data_str = np.unique(summary.tuning)[0] if len(np.unique(summary.tuning)) == 1 else 'multiple'
			subtitle += '\nTuning data: ' + tuning_data_str
			
			strip_punct_str = 'No punctuation' if all(summary.strip_punct) else "Punctuation" if all(~summary.strip_punct) else 'Multiple punctuation'
			subtitle += ', ' + strip_punct_str
			
			perc_correct_str = '\nBoth: ' + str(both_correct) + ', Neither: ' + str(both_incorrect) + ', X only: ' + str(ref_correct_gen_incorrect) + ', Y only: ' + str(ref_incorrect_gen_correct)
			subtitle += perc_correct_str
			
			fig.suptitle(title + '\n' + subtitle)
			
			fig.tight_layout()
			plt.savefig(f"{dataset_name}-{pair[0]}-{pair[1]}-paired.pdf")
			plt.close('all')
			del fig
		
		acc['eval_epoch'] = np.unique(summary.eval_epoch)[0] if len(np.unique(summary.eval_epoch)) == 1 else 'multi'
		acc['model_id'] = np.unique(summary.model_id)[0] if len(np.unique(summary.model_id)) == 1 else 'multi'
		acc['eval_data'] = np.unique(summary.eval_data)[0] if len(np.unique(summary.eval_data)) == 1 else 'multi'
		acc['model_name'] = np.unique(summary.model_name)[0] if len(np.unique(summary.model_name)) == 1 else 'multi'
		acc['tuning'] = np.unique(summary.tuning)[0] if len(np.unique(summary.tuning)) == 1 else 'multi'
		acc['masked'] = np.unique(summary.masked)[0] if len(np.unique(summary.masked)) == 1 else 'multi'
		acc['masked_tuning_style'] = np.unique(summary.masked_tuning_style)[0] if len(np.unique(summary.masked_tuning_style)) == 1 else 'multi'
		acc['strip_punct'] = np.unique(summary.strip_punct)[0] if len(np.unique(summary.strip_punct)) == 1 else 'multi'
		
		all_epochs = '-'.join([str(x) for x in sorted(np.unique(summary.eval_epoch).tolist(), key = lambda x: -x)])
		
		acc.to_csv(f'{dataset_name}-{all_epochs}-accuracy.csv', index = False)
		
		# Combine the plots into a single pdf
		pdfs = []
		for sentence_type in sentence_types:
			pdfs.append([pdf for pdf in os.listdir(os.getcwd()) if pdf.endswith(f'{sentence_type}-paired.pdf')])
		
		pdfs = [pdf for sublist in pdfs for pdf in sublist]
		
		# Filter out duplicate file names
		pdfs = list(set(pdfs))
		keydict = eval_cfg.data.sentence_types
		keydict = {k : v for v, k in enumerate(keydict)}
		
		pdfs = sorted(pdfs, key = lambda pdf: keydict[pdf.replace(dataset_name + '-' + self.reference_sentence_type + '-', '').replace('.pdf', '').replace('-paired', '')])
		merged_plots = PdfFileMerger()
		
		for pdf in pdfs:
			with open(pdf, 'rb') as f:
				merged_plots.append(PdfFileReader(f))
				
		merged_plots.write(f'{dataset_name}-{all_epochs}-plots.pdf')
		
		# Clean up
		for pdf in pdfs:
			os.remove(pdf)
	
	
	def eval_new_verb(self, eval_cfg: DictConfig, args_cfg: DictConfig, checkpoint_dir: str, epoch: int = None):
		"""
		Computes model performance on data with new verbs
		where this is determined as the difference in the probabilities associated
		with each argument to be predicted before and after training.
		To do this, we check predictions for each arg, word pair in args_cfg on a fresh model, 
		and then check them on the fine-tuned model.
		"""
		from transformers import pipeline
		
		data = self.load_eval_verb_file(args_cfg, eval_cfg.data.name, eval_cfg.data.to_mask)
		
		# Define a local function to evaluate the models
		def get_probs(epoch: int):
			epoch = self.restore_weights(checkpoint_dir, epoch)
			filler = pipeline('fill-mask', model = self.model, tokenizer = self.tokenizer)
			
			log.info(f'Evaluating model @ epoch {epoch} on testing data')
			results = {}
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
								#if not self.model_bert_name == 'roberta':
								s_dict['results'][arg2] = filler(s, targets = args_cfg[arg2])
								"""else:
									targets = args_cfg[arg2]
									# if we're using roberta and the mask token is not at the beginning of the string,
									# add the weird "space before me" character to the targets
									if not re.findall(rf'^{self.mask_tok}', s):
										targets = ['\u0120' + arg for arg in targets]
									
									###########################################################
									# roberta's filler doesn't deal with new words correctly, #
									# it blanks them out ######################################
									###########################################################
									# need to figure out how to fix this ######################
									###########################################################
									s_dict['results'][arg2] = filler(s, targets = targets)
									for i, d in s_dict['results'][arg2].iteritems():
										# Remove the preceding blanks so we can match them up later
										s_dict['results'][arg2][i]['token_str'] = s_dict['results'][arg2][i]['token_str'].strip()"""
								
								s_dict['results'][arg2] = sorted(
									s_dict['results'][arg2],
									key = lambda x: args_cfg[arg2].index(x['token_str'].replace(' ', ''))
								)
								
						results[arg][eval_cfg.data.sentence_types[i]].append(s_dict)
			
			return { epoch : results }
		
		results = {**get_probs(epoch = 0), **get_probs(epoch = epoch)}
		
		print(f"SAVING TO: {os.getcwd()}")
		
		summary = self.get_new_verb_summary(results, args_cfg, eval_cfg)
		
		# Print the summary
		# Disabled for now, as this is not actually super useful
		"""for (eval_epoch, target_position_name, predicted_token_type, sentence_type), summary_slice \
			in summary.groupby(['eval_epoch', 'predicted_token_type', 'target_position_name', 'sentence_type'], sort = False):
			surprisals = [round(float(sur), 2) for sur in list(summary_slice.surprisal.values)]
			predicted_token_type = predicted_token_type.replace('_', ' ')
			target_position_name = target_position_name.replace('_', ' ')
			print(f'\n{eval_epoch[0].upper() + eval_epoch[1:]} surprisals of {predicted_token_type} arguments in {target_position_name} position in {sentence_type}s:\n\t{surprisals}')
			print('')"""
		
		# Save the summary
		dataset_name = eval_cfg.data.friendly_name
		epoch = max(results.keys())
		
		summary.to_pickle(f"{dataset_name}-0-{epoch}-scores.pkl")
		summary.to_csv(f"{dataset_name}-0-{epoch}-scores.csv", index = False)

		# Create graphs
		self.graph_new_verb_results(summary, eval_cfg)
	
	def load_eval_verb_file(self, args_cfg, data_path: str, replacing: Dict[str, str]):
		
		resolved_path = os.path.join(
			hydra.utils.get_original_cwd(),
			"data",
			data_path
		)
		
		with open(resolved_path, "r") as f:
			raw_input = [line.strip().lower() for line in f]
		
		if self.cfg.hyperparameters.strip_punct:
			raw_input = [strip_punct(line) for line in raw_input]
		
		sentences = []
		for r in raw_input:
			line = []
			s_splits = r.split(',')
			for s in s_splits:
				for key in replacing:
					s = s.replace(key, self.tokens_to_mask[replacing[key]])
				
				line.append(s.strip())
			
			sentences.append(line)
		
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
		
		return filled_sentences
	
	def get_new_verb_summary(self, results, args_cfg, eval_cfg: DictConfig):
		"""
		Convert the pre- and post-tuning results into a pandas.DataFrame
		"""
		
		# Define a local function to convert each set of results to a data frame
		def convert_results(results: dict, args_cfg) -> pd.DataFrame:
			summary = pd.DataFrame()
			
			for eval_epoch in results:
				for target_position in results[eval_epoch]:
					for sentence_type in results[eval_epoch][target_position]:
						for i, sentence in enumerate(results[eval_epoch][target_position][sentence_type]):
							if 'results' in sentence:
								for predicted_token_type in sentence['results']:
									for prediction in sentence['results'][predicted_token_type]:
										summary_ = pd.DataFrame()
										pred_seq = prediction['sequence']
										mask_seq = sentence['sentence']
										# replace the internal tokens with the visible one
										for eval_group in eval_cfg.data.eval_groups:
											pred_seq = pred_seq.replace(
												eval_cfg.data.eval_groups[eval_group],
												eval_cfg.data.to_mask['[' + eval_group + ']']
											)
											
											mask_seq = mask_seq.replace(
												eval_cfg.data.eval_groups[eval_group],
												eval_cfg.data.to_mask['[' + eval_group + ']']
											)
										
										summary_['filled_sentence'] = [pred_seq]
										summary_['vocab_token_index'] = [prediction['token']]
										summary_['predicted_token'] = [prediction['token_str'].replace(' ', '')]
										summary_['p'] = [prediction['score']]
										summary_['surprisal'] = -np.log2(summary_['p'])
										
										summary_['predicted_token_type'] = [re.sub(r'\[|\]', '', predicted_token_type)]
										summary_['masked_sentence'] = [mask_seq]
										summary_['sentence_type'] = [sentence_type]
										summary_['sentence_num'] = [i]
										summary_['target_position_name'] = [re.sub(r'\[|\]', '', target_position)]
										summary_['eval_epoch'] = eval_epoch
										
										# Get the position number of the masked token
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
												
										summary_['target_position_num'] = ['position_' + str(position_num)]
										
										summary = summary.append(summary_, ignore_index = True)
				
			summary['model_id'] = os.path.normpath(os.getcwd()).split(os.sep)[-2]
			summary['eval_data'] = eval_cfg.data.friendly_name
			summary['model_name'] = self.model_bert_name
			summary['masked'] = self.masked
			summary['masked_tuning_style'] = self.masked_tuning_style
			summary['tuning'] = self.cfg.tuning.name
			summary['strip_punct'] = self.cfg.hyperparameters.strip_punct
			
			return summary
		
		summary = convert_results(results, args_cfg)
		
		# Reorder the columns
		columns = [
			'model_id', 'model_name', 'eval_epoch', 'tuning', 'strip_punct', 'masked', 'masked_tuning_style', # model properties
			'eval_data', # eval properties
			'sentence_type', 'target_position_name', 'target_position_num', 'predicted_token_type', 'masked_sentence', 'sentence_num', # sentence properties
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
	
	def graph_new_verb_results(self, summary, eval_cfg: DictConfig, axis_size = 10, multi = False, pt_size = 24):
		
		if multi:
			summary['surprisal'] = summary['mean']
			summary = summary.drop('mean', axis = 1)
		
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
		for sentence_type in sentence_types:
			
			# Get x and y data. We plot the first member of each pair on x, and the second member on y
			x_data = summary.loc[(summary.eval_epoch == x_epoch) & (summary['sentence_type'] == sentence_type)].reset_index(drop = True)
			y_data = summary.loc[(summary.eval_epoch == y_epoch) & (summary['sentence_type'] == sentence_type)].reset_index(drop = True)
			
			if not multi:
				lim = np.max(np.abs([*x_data['surprisal'].values, *y_data['surprisal'].values])) + 0.5
			else:
				x_sur = np.abs(x_data['surprisal'].values) + x_data['sem'].values
				y_sur = np.abs(y_data['surprisal'].values) + y_data['sem'].values
				lim = np.max([*x_sur, *y_sur]) + 0.5
			
			# Get number of linear positions (if there's only one position, we can't make plots by linear position)
			sur_names_positions = x_data[['surprisal_gf_label', 'target_position_num']].drop_duplicates().reset_index(drop = True)
			sur_names_positions = list(sur_names_positions.to_records(index = False))
			sur_names_positions = sorted(sur_names_positions, key = lambda x: int(x[1].replace('position_', ' ')))
			
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
			
			if multi:
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
			if not multi:
				ylim_diffs = np.max(np.abs([*x_data.surprisal.values, *(y_data.surprisal - x_data.surprisal).values])) + 1
			else:
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
			
			if multi:
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
		pdfs = []
		for sentence_type in sentence_types:
			pdfs.append([pdf for pdf in os.listdir(os.getcwd()) if pdf.endswith(f'{sentence_type}.pdf')])
		
		pdfs = [pdf for sublist in pdfs for pdf in sublist]
		
		# Filter out duplicate file names in case some sentence types are contained within others
		pdfs = list(set(pdfs))
		keydict = eval_cfg.data.sentence_types
		keydict = {k : v for v, k in enumerate(keydict)}
		
		pdfs = sorted(pdfs, key = lambda pdf: keydict[pdf.replace(dataset_name + '-', '').replace('.pdf', '')])
		merged_plots = PdfFileMerger()
		
		for pdf in pdfs:
			with open(pdf, 'rb') as f:
				merged_plots.append(PdfFileReader(f))
		
		all_epochs = '-'.join([str(x) for x in sorted(np.unique(summary.eval_epoch).tolist(), key = lambda x: x)])
		
		merged_plots.write(f'{dataset_name}-{all_epochs}-plots.pdf')
		
		# Clean up
		for pdf in pdfs:
			os.remove(pdf)
	
	# no longer used
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
	
	# no longer used
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
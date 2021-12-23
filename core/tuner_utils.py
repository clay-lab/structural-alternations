# tuner_utils.py
#
# utility functions for tuner.py and related
import os
import re
import json
import torch
import random
import logging
import requests

import numpy as np
import pandas as pd

from typing import List, Type
from PyPDF2 import PdfFileMerger, PdfFileReader
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer
from statsmodels.nonparametric.smoothers_lowess import lowess

model_max_length = 512

log = logging.getLogger(__name__)

def z_transform(x: np.ndarray) -> np.ndarray:
	z = (x - np.mean(x))/np.std(x)
	return z

def set_seed(seed: int) -> None:
	seed = int(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	
def strip_punct(sentence: str) -> str:
	return re.sub(r'[^\[\]\<\>\w\s,]', '', sentence)
	
def merge_pdfs(pdfs: List[str], filename: str) -> None:
	if not pdfs:
		return
	
	merged_pdfs = PdfFileMerger()
	
	for pdf in pdfs:
		with open(pdf, 'rb') as f:
			merged_pdfs.append(PdfFileReader(f))
	
	merged_pdfs.write(filename)
	
	# Clean up (if the os doesn't happen to lock the file)
	for pdf in pdfs:
		try:
			os.remove(pdf)
		except Exception:
			pass

def create_tokenizer_with_added_tokens(model_id: str, tokenizer_class: Type['PreTrainedTokenizer'], tokens_to_mask: List[str], delete_tmp_vocab_files: bool = True, **kwargs) -> 'PreTrainedTokenizer':
	if 'uncased' in model_id:
		tokens_to_mask = [t.lower() for t in tokens_to_mask]
	
	# if we are doing bert/distilbert, the answer is easy: just add the tokens to the end of the vocab.txt file
	# which we generate by creating a pretrained tokenizer and extracting the vocabulary
	if re.search(r'(^bert-)|(^distilbert-)', model_id):
		bert_tokenizer = tokenizer_class.from_pretrained(model_id, **kwargs)
		vocab = bert_tokenizer.get_vocab()
		for token in tokens_to_mask:
			if len(bert_tokenizer.tokenize(token)) == 1 and not bert_tokenizer.tokenize(token) == [bert_tokenizer.unk_token]:
				raise ValueError(f'New token {token} already exists in {model_id} tokenizer!')
			
			vocab.update({token: len(vocab)})
		
		with open('vocab.tmp', 'w', encoding = 'utf-8') as tmp_vocab_file:
			tmp_vocab_file.write('\n'.join(vocab))
		
		tokenizer = tokenizer_class(name_or_path = model_id, vocab_file = 'vocab.tmp', **kwargs)
		tokenizer.model_max_length = model_max_length
		
		# for some reason, we have to re-add the [MASK] token to bert to get this to work, otherwise
		# it breaks it apart into separate tokens '[', 'mask', and ']' when loading the vocab locally (???)
		# I have verified that this does not affect the embedding or the model's ability to recognize it
		# as a mask token (e.g., if you create a filler object with this tokenizer, 
		# it will identify the mask token position correctly)
		tokenizer.add_tokens(tokenizer.mask_token, special_tokens=True)
		tokenizer.model_max_length = model_max_length
		
		if delete_tmp_vocab_files:
			os.remove('vocab.tmp')
		
		if verify_tokens_exist(tokenizer, tokens_to_mask):
			return tokenizer
		else:
			raise Exception('New tokens were not added correctly!')
	
	# for roberta, we need to modify both the merges.txt file and the vocab.json files
	elif re.search(r'^roberta-', model_id):
		roberta_tokenizer = tokenizer_class.from_pretrained(model_id, **kwargs)
		vocab = roberta_tokenizer.get_vocab()
		
		# strip any preceding spaces out since they'll be added specially in the functions called here
		tokens_to_mask = [token.strip() for token in tokens_to_mask]
		
		# we need to get the tokens from the cfg here instead of the class property 
		# so we do not get the ones with spaces before,
		# which have to be added in a special way in the functions called here
		for token in tokens_to_mask:
			if ( # verify that the token or the version of it with a preceding space does not already exist in the vocabulary
					(len(roberta_tokenizer.tokenize(token)) == 1 or len(roberta_tokenizer.tokenize(' ' + token)) == 1) 
					and not roberta_tokenizer.tokenize(token) == roberta_tokenizer.unk_token
					and not roberta_tokenizer.tokenize(' ' + token) == roberta_tokenizer.unk_token
				):
				raise ValueError(f'New token {token} already exists in {model_id} tokenizer!')
			# roberta treats words with spaces in front of them differently,
			# so we add a version of the token with that special character to the model as well
			vocab.update({token : len(vocab)})
			vocab.update({chr(288) + token : len(vocab)})
		
		with open('vocab.tmp', 'w', encoding = 'utf8') as tmp_vocab_file:
			json.dump(vocab, tmp_vocab_file, ensure_ascii=False)
		
		try:
			url = f'https://huggingface.co/{model_id}/resolve/main/merges.txt'
			merges = requests.get(url).content.decode().split('\n')
		except Exception:
			raise FileNotFoundError(f'Unable to access {model_id} merges.txt file from huggingface. Are you connected to the Internet?')
		
		merges = merges[:1] + get_roberta_merges_for_new_tokens(tokens_to_mask) + merges[1:]
		merges = list(dict.fromkeys(merges)) # drop the duplicates while preserving order
		with open('merges.tmp', 'w', encoding = 'utf-8') as tmp_merges_file:
			tmp_merges_file.write('\n'.join(merges))
		
		tokenizer = tokenizer_class(name_or_path = model_id, vocab_file = 'vocab.tmp', merges_file = 'merges.tmp', **kwargs)
		
		# for some reason, we have to re-add the <mask> token to roberta to get this to work, otherwise
		# it breaks it apart into separate tokens '<', 'mask', and '>' when loading the vocab and merges locally (???)
		# I have verified that this does not affect the embeddings or the model's ability to recognize it
		# as a mask token (e.g., if you create a filler object with this tokenizer, 
		# it will identify the mask token position correctly)
		tokenizer.add_tokens(tokenizer.mask_token, special_tokens=True)
		tokenizer.model_max_length = model_max_length
		
		if delete_tmp_vocab_files:
			os.remove('vocab.tmp')
			os.remove('merges.tmp')
		
		roberta_tokens = list(tokens_to_mask) + [' ' + token for token in list(tokens_to_mask)]
		if verify_tokens_exist(tokenizer, roberta_tokens):
			return tokenizer
		else:
			raise Exception('New tokens were not added correctly!')
	
	else:
		raise ValueError('Only BERT, DistilBERT, and RoBERTa tokenizers are supported.')

def get_roberta_merges_for_new_tokens(new_tokens: List[str]) -> List[str]:
	roberta_merges = [gen_roberta_merges_pairs(new_token) for new_token in new_tokens]
	roberta_merges = [pair for token in roberta_merges for pair in token]
	return roberta_merges
		
def gen_roberta_merges_pairs(new_token: str, highest: bool = True) -> List[str]:
	chrs = [c for c in new_token]
	if len(chrs) == 2:
		if not highest:
			return tuple([chrs[0], chrs[1]])
		else:
			return [' '.join([chrs[0], chrs[1]])]
			
	if len(chrs) == 3:
		if not highest:
			return tuple([chrs[0], ''.join(chrs[1:])])
		else:
			return gen_roberta_merges_pairs(chrs[1:]) + [' '.join([chrs[0], ''.join(chrs[1:])])]
	
	if len(chrs) % 2 == 0:
		pairs = gen_roberta_merges_pairs(''.join(chrs[:-2]), highest = False)
		pairs += gen_roberta_merges_pairs(''.join(chrs[-2:]), highest = False)
		pairs += tuple([''.join(chrs[:-2]), ''.join(chrs[-2:])])
		if not highest:
			return pairs
	else:
		pairs = gen_roberta_merges_pairs(''.join(chrs[:-3]), highest = False)
		pairs += gen_roberta_merges_pairs(''.join(chrs[-2:]), highest = False)
		pairs += gen_roberta_merges_pairs(''.join(chrs[-3:]), highest = False)
		pairs += tuple([''.join(chrs[:-3]), ''.join(chrs[-3:])])
		if not highest:		
			return pairs
	
	pairs = tuple(zip(pairs[::2], pairs[1::2]))
	pairs = [' '.join(pair) for pair in pairs]
	sp = chr(288)
	# # make a copy so we can loop through the original list while updating it
	# old_pairs = pairs.copy()
	# pairs.append(f'{sp} {new_token[0]}')
	# for pair in old_pairs:
	# 	#if pair.startswith(new_token[0]):
	# 	if re.search(r'^' + ''.join(pair.split(' ')), new_token):
	# 		pairs.append(sp + pair)
	
	# pairs with the preceding special token
	g_pairs = []
	for pair in pairs:
		if re.search(r'^' + ''.join(pair.split(' ')), new_token):
			g_pairs.append(chr(288) + pair)
	
	pairs = g_pairs + pairs
	pairs = [f'{sp} {new_token[0]}'] + pairs
	
	pairs = list(dict.fromkeys(pairs)) # remove any duplicates
	
	return pairs

def verify_tokens_exist(tokenizer: 'PreTrainedTokenizer', tokens: List[str]) -> bool:
	for token in tokens:
		if len(tokenizer.tokenize(token)) != 1:
			return False
		elif tokenizer.tokenize(token) == [tokenizer.unk_token]:
			return False
	
	return True

def verify_tokenization_of_sentences(tokenizer: 'PreTrainedTokenizer', sentences: List[str], tokens_to_mask: List[str] = None, **kwargs) -> bool:
	"""
	verify that a custom tokenizer and one created using from_pretrained behave identically except on the tokens to mask
	"""
	tokenizer_id = tokenizer.name_or_path
	tokenizer_one = tokenizer
	tokenizer_two = eval(tokenizer_one.__class__.__name__).from_pretrained(tokenizer_id, **kwargs)
	
	# flatten a list of lists of strings without breaking string into characters
	# from https://stackoverflow.com/questions/5286541/how-can-i-flatten-lists-without-splitting-strings
	flatten = lambda y: [k for j in ([i] if not isinstance(i,list) else flatten(i) for i in y) for k in j]
	sentences = flatten(sentences)
	
	if tokens_to_mask:
		masked_sentences = []
		for sentence in sentences:
			for token in tokens_to_mask:
				sentence = sentence.replace(token, tokenizer.mask_token)
			
			masked_sentences.append(sentence)
		
		sentences = masked_sentences
	
	tokenizations_one = tokenizer_one(sentences)['input_ids']
	tokenizations_two = tokenizer_two(sentences)['input_ids']
	
	comparisons = [True if tokenization_one == tokenization_two else False for tokenization_one, tokenization_two in zip(tokenizations_one, tokenizations_two)]
	if not all(comparisons):
		mismatches = [i for i, comparison in enumerate(comparisons) if comparison != True]
		mismatches_pairs = [[[mismatch] + tokenizer_one.convert_ids_to_tokens(tokenizations_one[mismatch]), [mismatch] + tokenizer_two.convert_ids_to_tokens(tokenizations_two[mismatch])] for mismatch in mismatches]
		log.warning(f'The following sentences did not match for {tokenizer_id}!\n')
		for i, mismatch_pair in enumerate(mismatches_pairs):
			log.warning(f'(evaluation) (s{mismatch_pair[0][0]}): ' + ', '.join(mismatch_pair[0][1:]))
			log.warning(f'(pretrained) (s{mismatch_pair[1][0]}): ' + ', '.join(mismatch_pair[1][1:]) + '\n')
		
		return False
	
	return True

def get_best_epoch(loss_df: pd.DataFrame, method: str = 'mean', frac: float = 0.1) -> int:
	loss_df = loss_df.copy().sort_values(['dataset', 'epoch']).reset_index(drop=True)
	
	datasets = loss_df.dataset.unique()
	
	# replace the losses with the lowess
	# this smooths the irregular losses we see in various circumstances
	# no longer needed to due change in how we do the dev sets
	# for dataset in datasets:
	# 	loss_df.loc[loss_df.dataset == dataset, 'value'] = lowess(loss_df[loss_df.dataset == dataset].value.values, loss_df[loss_df.dataset == dataset].epoch.values, frac = frac)[:,1]
	
	if method == 'sumsq':
		best_losses = loss_df.loc[loss_df.groupby('dataset').value.idxmin()].reset_index(drop=True)
		
		epoch_sumsqs = []
		for dataset in datasets:
			dataset_epoch = best_losses[best_losses.dataset == dataset].epoch.values[0]
			epoch_losses = loss_df.loc[loss_df.epoch == dataset_epoch].reset_index(drop=True).value.values
			epoch_avg_loss = np.mean(epoch_losses)
			sumsq = sum((epoch_avg_loss - epoch_losses)**2)
			epoch_sumsq = tuple([dataset_epoch, sumsq])
			epoch_sumsqs.append(epoch_sumsq)
		
		epoch_sumsqs = sorted(epoch_sumsqs, key = lambda epoch_sumsq: epoch_sumsq[1])
		best_epoch = epoch_sumsqs[0][0]
		log.info(f'Best epoch is {best_epoch} (sumsq loss = ' + '{:.2f}'.format(epoch_sumsqs[0][1]) + f', minimum for {", ".join(best_losses[best_losses.epoch == best_epoch].dataset.values)}).')
	elif method == 'mean':
		mean_losses = loss_df.groupby('epoch').value.agg('mean')
		lowest_loss = mean_losses.min()
		best_epoch = loss_df.loc[loss_df.epoch == mean_losses.idxmin()].epoch.unique()[0]
		log.info(f'Best epoch is {best_epoch} (mean loss = ' + '{:.2f}'.format(lowest_loss) + ').')
	else:
		best_epoch = loss_df.epoch.max()
		log.warning(f'No method for determining best epoch provided (use "sumsq", "mean"). The highest epoch {best_epoch} will be used. This may not be the best epoch!')
	
	if best_epoch == max(loss_df.epoch):
		log.warning('Note that the best epoch is the final epoch. This may indicate underfitting.')
	
	return best_epoch
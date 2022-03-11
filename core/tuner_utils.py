# tuner_utils.py
#
# utility functions for tuner.py and related
import os
import re
import json
import gzip
import torch
import random
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob
from typing import *
from shutil import copyfileobj
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

def z_transform(x: np.ndarray) -> np.ndarray:
	return (x - np.mean(x))/np.std(x)

def set_seed(seed: int) -> None:
	seed = int(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	
def strip_punct(sentence: str) -> str:
	return re.sub(r'[^\[\]\<\>\w\s,]', '', sentence)

def create_tokenizer_with_added_tokens(model_id: str, tokens_to_mask: List[str], delete_tmp_vocab_files: bool = True, **kwargs) -> 'PreTrainedTokenizer':
	kwargs.update(dict(use_fast=False))
	
	if 'uncased' in model_id:
		tokens_to_mask = [t.lower() for t in tokens_to_mask]
	
	# if we are doing bert/distilbert, the answer is easy: just add the tokens to the end of the vocab.txt file
	# which we generate by creating a pretrained tokenizer and extracting the vocabulary
	if re.search(r'(^bert-)|(^distilbert-)', model_id):
		bert_tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
		vocab = bert_tokenizer.get_vocab()
		for token in tokens_to_mask:
			if len(bert_tokenizer.tokenize(token)) == 1 and not bert_tokenizer.tokenize(token) == [bert_tokenizer.unk_token]:
				raise ValueError(f'New token {token} already exists in {model_id} tokenizer!')
			
			vocab.update({token: len(vocab)})
		
		with open('vocab.tmp', 'w', encoding = 'utf-8') as tmp_vocab_file:
			tmp_vocab_file.write('\n'.join(vocab))
		
		exec(f'from transformers import {bert_tokenizer.__class__.__name__}')
		
		tokenizer = eval(bert_tokenizer.__class__.__name__)(name_or_path = model_id, vocab_file = 'vocab.tmp', **kwargs)
		
		# for some reason, we have to re-add the [MASK] token to bert to get this to work, otherwise
		# it breaks it apart into separate tokens '[', 'mask', and ']' when loading the vocab locally (???)
		# I have verified that this does not affect the embedding or the model's ability to recognize it
		# as a mask token (e.g., if you create a filler object with this tokenizer, 
		# it will identify the mask token position correctly)
		tokenizer.add_tokens(tokenizer.mask_token, special_tokens=True)
		tokenizer.model_max_length = bert_tokenizer.model_max_length
		
		if delete_tmp_vocab_files:
			os.remove('vocab.tmp')
		
		if verify_tokens_exist(tokenizer, tokens_to_mask):
			return tokenizer
		else:
			raise Exception('New tokens were not added correctly!')
	
	# for roberta, we need to modify both the merges.txt file and the vocab.json file
	elif re.search(r'^roberta-', model_id):
		roberta_tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
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
		
		merges = [' '.join(key) for key in roberta_tokenizer.bpe_ranks.keys()]
		# we have to add a newline at the beginning of the file
		# since it's expecting it to be a comment, so we add a blank string here
		# that will get joined with a newline
		merges = [''] + get_roberta_merges_for_new_tokens(tokens_to_mask) + merges
		merges = list(dict.fromkeys(merges)) # drop any duplicates we may have happened to add while preserving order
		with open('merges.tmp', 'w', encoding = 'utf-8') as tmp_merges_file:
			tmp_merges_file.write('\n'.join(merges))
		
		exec(f'from transformers import {roberta_tokenizer.__class__.__name__}')
		
		tokenizer = eval(roberta_tokenizer.__class__.__name__)(name_or_path = model_id, vocab_file = 'vocab.tmp', merges_file = 'merges.tmp', **kwargs)
		
		# for some reason, we have to re-add the <mask> token to roberta to get this to work, otherwise
		# it breaks it apart into separate tokens '<', 'mask', and '>' when loading the vocab and merges locally (???)
		# I have verified that this does not affect the embeddings or the model's ability to recognize it
		# as a mask token (e.g., if you create a filler object with this tokenizer, 
		# it will identify the mask token position correctly)
		tokenizer.add_tokens(tokenizer.mask_token, special_tokens=True)
		tokenizer.model_max_length = roberta_tokenizer.model_max_length
		
		if delete_tmp_vocab_files:
			os.remove('vocab.tmp')
			os.remove('merges.tmp')
		
		roberta_tokens = list(tokens_to_mask) + [' ' + token for token in list(tokens_to_mask)]
		if verify_tokens_exist(tokenizer, roberta_tokens):
			return tokenizer
		else:
			raise Exception('New tokens were not added correctly!')
	else:
		raise ValueError('Only BERT, DistilBERT, and RoBERTa tokenizers are currently supported.')

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
		if len(tokenizer.tokenize(token)) != 1 or tokenizer.tokenize(token) == [tokenizer.unk_token]:
			return False
	
	return True

def verify_tokenization_of_sentences(tokenizer: 'PreTrainedTokenizer', sentences: List[str], tokens_to_mask: List[str] = None, **kwargs) -> bool:
	"""
	verify that a custom tokenizer and one created using from_pretrained behave identically except on the tokens to mask
	"""
	kwargs.update(dict(use_fast=False))
	
	tokenizer_id = tokenizer.name_or_path
	tokenizer_one = tokenizer
	tokenizer_two = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
	
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

def get_best_epoch(loss_df: pd.DataFrame, method: str = 'mean') -> int:
	loss_df = loss_df.copy().sort_values(['dataset', 'epoch']).reset_index(drop=True)
	
	datasets = loss_df.dataset.unique()

	if method in ['best_sumsq', 'sumsq']:
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
	elif method in ['best_mean', 'mean']:
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

# Used with the R analysis script since it's much quicker to do this in Python
# went back to the old way of doing this because of https://github.com/rstudio/reticulate/issues/1166
# def load_csv_gzs(csv_gzs: List[str]) -> pd.DataFrame:
# 	csv_gzs = [csv_gzs] if isinstance(csv_gzs, str) else csv_gzs
# 	return pd.concat([pd.read_csv(f) for f in tqdm(csv_gzs)], ignore_index=True)

# Used with the R analysis script since it's much quicker to do this in Python
def unzip_csv_gzs(csv_gzs: List[str]) -> None:
	dests = [f.replace('.gz', '') for f in csv_gzs]
	fs_dests = tuple(zip(csv_gzs, dests))
	for f, dest in tqdm(fs_dests):
		with gzip.open(f, "rb") as f_in, open(dest, "wb") as f_out:
			copyfileobj(f_in, f_out)

# Used with the R analysis script since it's much quicker to do this in Python
def delete_files(files: List[str]) -> None:
	for f in tqdm(files):
		try: 
			os.remove(f)
		except:
			print(f'Unable to remove {f}.')
			continue

class gen_summary():
	"""
	Generates summary statistics for saved dataframes
	"""
	def __init__(self, 
		df: Union[pd.DataFrame,str,'TextIOWrapper','StringIO'], 
		columns: List = ['gen_given_ref', 'correct', 'odds_ratio_pre_post_diff', 'odds_ratio', 'cossim'],
		funs: List = ['mean', 'sem']
	) -> Type['gen_summary']:
		"""
		Create a gen_summary instance
		df: a pandas dataframe, str, or file-like object
		columns: a list of columns to generate summary statistics for that are in df.columns
		funs: a list of summary statistics to generate using pandas' agg function
		"""
		if not isinstance(df, pd.DataFrame):
			df = pd.read_csv(df)
			
		self.df = df
		self.columns = [c for c in columns if c in self.df.columns]
		self.funs = funs
		self.prep_quasiquotation()
	
	def __call__(self, *columns: Tuple) -> pd.DataFrame.groupby:
		"""
		Generate a summary using the columns specified, print and return it
		"""
		agg_dict = {c : self.funs for c in self.columns if c in self.df.columns}
		results = self.df.groupby(list(columns)).agg(agg_dict).sort_values(list(columns))
		print(results)
		return results
	
	def prep_quasiquotation(self) -> None:
		"""
		Add column names to global variables to facilitate R-like quasiquotation usage.
		As a failsafe, if a variable is already defined with a conflicting definition,
		we exit. In this case, strings may be provided to generate a summary.
		"""
		for c in self.df.columns:
			if c in globals() and not globals()[c] == c:
				print(f"{c} already has a conflicting definition: '{globals()[c]}'. Exiting.")
				return
			
			globals()[c] = c
	
	def set_funs(self, *funs: Tuple) -> None:
		"""
		Set the summary functions to use for this gen_summary
		funs: a list of summary statistics to generate using pandas' agg function
		"""
		self.funs = list(funs)
	
	def set_columns(*columns: Tuple) -> None:
		"""
		Set the columns to get summary statistics for
		columns: a list of columns to generate summary statistics for that are in self.df.columns
		"""
		self.columns = [c for c in list(columns) if c in self.df.columns]
	
	def set_df(self, df: Union[pd.DataFrame,str,'TextIOWrapper','StringIO']) -> None:
		"""
		Set the df to a different one, and reinitialize all variables
		df: a pd.DataFrame, str, or file-like object
		"""
		if not isinstance(df, pd.DataFrame):
			df = pd.read_csv(df)
		
		self.df = df
		
		# reset the columns to remove any that aren't present in the new df
		set_columns(self.columns)
		
		# reinitialize quasiquotation variables using the columns from the new df
		self.prep_quasiquotation()
		
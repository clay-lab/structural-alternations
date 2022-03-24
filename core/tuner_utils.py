# tuner_utils.py
#
# utility functions for tuner.py
import os
import re
import json
import gzip
import torch
import random
import inspect
import logging
import itertools

import numpy as np
import pandas as pd

from math import sqrt
from tqdm import tqdm
from glob import glob
from typing import *
from copy import deepcopy
from shutil import copyfileobj
from omegaconf import OmegaConf, DictConfig, ListConfig
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

# grammatical functions in order of prominence for sorting things
GF_ORDER = [
	'[subj]', 
	'[obj]', 
	'[2obj]', 
	'[iobj]', 
	'[obl]',
	'[pobj]', 
	'[adj]'
]


# short useful functions
def flatten(l: List) -> List:
	# flatten a list of lists or np.ndarrays without breaking strings into characters
	# from https://stackoverflow.com/questions/5286541/how-can-i-flatten-lists-without-splitting-strings
	l = apply_to_all_of_type(l, ListConfig, OmegaConf.to_container)
	if l is not None:
		return [k for j in ([i] if not isinstance(i,(list,tuple,np.ndarray)) else flatten(i) for i in l) for k in j]

def listify(l: 'any') -> List:
	if isinstance(l,list):
		return l
	
	if isinstance(l,ListConfig):
		return OmegaConf.to_container(l)
	
	if isinstance(l,tuple):
		return list(l)
	
	if isinstance(l,(np.ndarray,pd.Series)):
		return l.tolist()
	
	return [l]

def unlistify(l: 'any') -> 'any':
	if isinstance(l,dict):
		return l[list(l.keys())[0]] if len(l) == 1 else l
	
	return l[0] if isinstance(l,list) or isinstance(l,tuple) and len(l) == 1 else l

def none(iterator: 'Iterator') -> bool:
	
	return not any(iterator)

def multiplator(
	series: Union[List,pd.Series], 
	multstr = 'multiple'
) -> 'any':
	if np.unique(series).size == 1:
		return np.unique(series)[0]
	elif np.unique(series).size > 1:
		return multstr
	else:
		return np.nan

def z_transform(x: np.ndarray) -> np.ndarray:
	diffs = x - np.mean(x)
	return diffs/np.std(x)

def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def strip_punct(sentence: str) -> str:
	
	return re.sub(r'[^\[\]\<\>\w\s,]', '', sentence)
	
def apply_to_all_of_type(
	data: 'any', 
	t: Type, 
	fun: Callable, 
	*args: List, 
	**kwargs: Dict
) -> 'any':
	if isinstance(data,(DictConfig,ListConfig)):
		# we need the primitive versions of these so we can modify them
		data = OmegaConf.to_container(data)
	
	data = deepcopy(data)
	
	if isinstance(data,t):
		returns = filter_none(fun(data, *args, **kwargs))
	elif isinstance(data,dict):
		returns = filter_none({apply_to_all_of_type(k, t, fun, *args, **kwargs): apply_to_all_of_type(v, t, fun, *args, **kwargs) for k, v in data.items()})
	elif isinstance(data,(list,tuple,set)):
		returns = filter_none(type(data)(apply_to_all_of_type(i, t, fun, *args, **kwargs) for i in data))
	elif isinstance(data,(torch.Tensor,pd.Series)):
		returns = filter_none(type(data)([apply_to_all_of_type(i, t, fun, *args, **kwargs) for i in data]))
	elif isinstance(data,np.ndarray):
		returns = filter_none(np.array([apply_to_all_of_type(i, t, fun, *args, **kwargs) for i in data]))
	else:
		returns = filter_none(data)
	
	if isinstance(data,(pd.Series,np.ndarray)):
		return returns if returns.any() else None
	else:
		return returns if returns else None

def filter_none(data: 'any') -> 'any':
	# from https://stackoverflow.com/questions/20558699/python-how-recursively-remove-none-values-from-a-nested-data-structure-lists-a
	data = deepcopy(data)
	
	if isinstance(data,(list,tuple,set)):
		return type(data)(filter_none(x) for x in data if x is not None)
	elif isinstance(data,dict):
		return type(data)(
			(filter_none(k), filter_none(v))
				for k, v in data.items() if k is not None and v is not None
			)
	else:
		if isinstance(data,(pd.Series,np.ndarray)):
			return data if data.any() else None
		else:
			return data if data else None

# decorator to create recursive functions
def recursor(t: 'type', *args, **kwargs) -> Callable:
	return lambda fun: \
		lambda data, *args, **kwargs: \
			apply_to_all_of_type(data=data, t=t, fun=fun, *args, **kwargs)


# summary and file related
def get_file_prefix(summary: pd.DataFrame) -> str:
	dataset_name = summary.eval_data.unique()[0]
	epoch_label = multiplator(summary.epoch_criteria, multstr = '-')
	if summary.model_id.unique().size == 1:
		epoch_label = '-' + epoch_label
		magnitude = len(str(summary.total_epochs.unique()[0]))
		epoch_label = f'{str(summary.eval_epoch.unique()[0]).zfill(magnitude)}{epoch_label}'
		
	file_prefix = f'{dataset_name}-{epoch_label}'
	
	return file_prefix

def get_sentence_label(data: pd.DataFrame) -> str:
	first_rows = data[data.sentence == data.loc[0].sentence][['ratio_name', 'position_ratio_name', 'sentence']].drop_duplicates().reset_index(drop=True)
	position_map = {}
	for row in first_rows.index:
		position_map.update({gf : position for gf, position in tuple(zip(first_rows.loc[row].ratio_name.split('/'), [int(p) for p in first_rows.loc[row].position_ratio_name.replace('position ', '').split('/')]))})
	
	position_map = dict(sorted(position_map.items(), key=lambda item: item[1]))
	sentence_ex = first_rows.sentence[0]
	for gf in position_map:
		# this is bit hacky, but it's to ensure it'll work in cases with multiple models' data
		# where we can't rely on the mask token for each model being available.
		sentence_ex = re.sub(r'^(.*?)(\[MASK\]|\<mask\>)', f'\\1{gf}', sentence_ex)
	
	return sentence_ex

def get_data_for_pairwise_comparisons(
	summary: pd.DataFrame,
	eval_cfg: DictConfig = None, 
	cossims: bool = False, 
	diffs: bool = False
) -> Tuple:
	
	def get_pairs(
		summary: pd.DataFrame, 
		eval_cfg: DictConfig = None, 
		cossims: bool = False
	) -> List[Tuple[str]]:
		if not cossims:
			# Get each unique pair of sentence types so we can create a separate plot for each pair
			types = summary.sentence_type.unique()
			types = sorted(types, key=lambda s_t: eval_cfg.data.sentence_types.index(s_t))
		else:
			types = summary.predicted_arg.unique()
		
		pairs = [pair for pair in list(itertools.combinations(types, 2)) if not pair[0] == pair[1]]
		
		if not cossims:
			# Sort so that the trained cases are first
			reference_sentence_type = multiplator(summary.reference_sentence_type)
			pairs = [sorted(pair, key=lambda x: str(-int(x == reference_sentence_type)) + x) for pair in pairs]
			
			# Filter to only cases including the reference sentence type for ease of interpretation
			pairs = [(s1, s2) for s1, s2 in pairs if s1 == reference_sentence_type] if reference_sentence_type != 'none' else pairs
		else:
			pairs = list(set(tuple(sorted(pair)) for pair in pairs))
		
		return pairs
	
	def format_summary_for_comparisons(
		summary: pd.DataFrame, 
		exp_type: str = None, 
		cossims: bool = False, 
		diffs: bool = False
	) -> Tuple[pd.DataFrame,Tuple[str]]:
		summary = summary.copy()
				
		if exp_type == 'newverb' and not cossims:
			# Sort by grammatical function prominence for newverb exps (we do this because 'subj' alphabetically follows 'obj'), but it's more natural for it to go first
			summary['ratio_order'] = [GF_ORDER.index(gf_ratio_name.split('/')[0]) for gf_ratio_name in summary.gf_ratio_name]
			summary = summary.sort_values(['model_id', 'ratio_order'])
			summary = summary.drop('ratio_order', axis=1)
			summary['ratio_name'] = [re.sub(r'\[|\]', '', ratio_name) for ratio_name in summary.ratio_name]
		
		colnames 	= get_eval_metrics_colnames(exp_type, cossims, diffs)
		metric 		= colnames[-2]
		
		if summary.model_id.unique().size > 1:
			metric = f'{metric}_mean'
		else:
			summary[f'{metric}_sem'] = 0
		
		return summary, colnames
	
	def get_eval_metrics_colnames(
		exp_type: str, 
		cossims: bool = False, 
		diffs: bool = False
	) -> Tuple[str]:
		if not cossims:
			metric = 'odds_ratio' if exp_type == 'newarg' or not diffs else 'odds_ratio_pre_post_difference' if exp_type == 'newverb' else None
		elif cossims:
			metric = 'cossim'
		
		semmetric = f'{metric}_sem'
		
		return metric, semmetric
	
	exp_type 			= eval_cfg.data.exp_type if eval_cfg is not None else None
	summary, colnames 	= format_summary_for_comparisons(summary, exp_type, cossims=cossims, diffs=diffs)
	pairs 				= get_pairs(summary, eval_cfg, cossims)
	
	return summary, tuple(colnames), pairs
	
def transfer_hyperparameters_to_df(
	source: pd.DataFrame, 
	target: pd.DataFrame
) -> pd.DataFrame:
	hp_cols = [
		c for c in source.columns if not c in [
			'odds_ratio', 'ratio_name', 'position_ratio_name',
			'role_position', 'token_type', 'token_id', 'token', 
			'sentence', 'sentence_type', 'sentence_num', 'odds_ratio_pre_post_difference',
			'both_correct', 'both_incorrect', 'gen_correct', 'gen_incorrect', 
			'ref_correct', 'ref_incorrect', 'ref_correct_gen_incorrect',
			'ref_incorrect_gen_correct', 'specificity', 'specificity_se',
			'gen_given_ref', 's1', 's2', 's1_ex', 's2_ex', 'arg_type', 'args_group',
			'predicted_arg', 'predicted_role',
		] 
		and not c.endswith('_ref') and not c.endswith('_gen')
	]
	
	for c in hp_cols:
		target[c] = multiplator(source[c])
	
	return target

def get_single_pair_data(
	summary: pd.DataFrame, pair: Tuple[str], 
	group: str, pair_col: str = 'sentence_type'
) -> Tuple:
	x_data = summary[summary[pair_col] == pair[0]].reset_index(drop=True)
	y_data = summary[summary[pair_col] == pair[1]].reset_index(drop=True)
	
	# Filter data to groups that only exist in both sets
	common_groups = set(x_data[group]).intersection(y_data[group])
	x_data = x_data[x_data[group].isin(common_groups)].reset_index(drop=True)
	y_data = y_data[y_data[group].isin(common_groups)].reset_index(drop=True)
	
	return x_data, y_data

def get_accuracy_measures(
	refs: pd.DataFrame, gens: pd.DataFrame, 
	colname: str
) -> Dict:
	refs_correct 				= refs[colname] > 0
	gens_correct 				= gens[colname] > 0
	num_points 					= len(refs.index)
	
	gen_given_ref 				= sum(gens_correct[refs_correct.index])/len(refs_correct) * 100 if not refs_correct.empty else np.nan
	both_correct 				= sum(refs_correct * gens_correct)/num_points * 100
	both_incorrect				= sum(-refs_correct * -gens_correct)/num_points * 100
	ref_correct 				= sum(refs_correct)/num_points * 100
	ref_incorrect 				= 100. - ref_correct
	gen_correct 				= sum(gens_correct)/num_points * 100
	gen_incorrect 				= 100. - gen_correct
	ref_correct_gen_incorrect 	= sum( refs_correct * -gens_correct)/num_points * 100
	ref_incorrect_gen_correct 	= sum(-refs_correct *  gens_correct)/num_points * 100
	
	sq_err						= (gens[colname] - refs[colname])**2
	specificity 				= np.mean(sq_err)
	specificity_se 				= np.std(sq_err)/sqrt(num_points)
	
	return {
		'gen_given_ref'				: gen_given_ref,
		'both_correct'				: both_correct,
		'both_incorrect'			: both_incorrect,
		'ref_correct'				: ref_correct,
		'ref_incorrect'				: ref_incorrect,
		'gen_correct'				: gen_correct,
		'gen_incorrect'				: gen_incorrect,
		'ref_correct_gen_incorrect'	: ref_correct_gen_incorrect,
		'ref_incorrect_gen_correct'	: ref_incorrect_gen_correct,
		'specificity_(MSE)'			: specificity,
		'specificity_se'			: specificity_se
	}

def get_odds_ratios_accuracies(
	summary: pd.DataFrame, 
	eval_cfg: DictConfig, 
	get_diffs_accuracies: bool = False
) -> pd.DataFrame:
	
	def update_acc(acc: List[Dict], refs: pd.DataFrame, gens: pd.DataFrame, colname: str, **addl_columns) -> None:
		acc_data = get_accuracy_measures(refs=refs, gens=gens, colname=colname)
		
		cols = {
			**addl_columns,
			**acc_data,
			'token'		: multiplator(refs.token, multstr='any'),
			'token_id'	: multiplator(refs.token_id),
		}
		
		if 'token_type' in refs.columns:
			cols = {**cols, 'token_type': multiplator(refs.token_type)}
		
		acc.append(cols)
	
	summary, (odds_ratio, odds_ratio_sem), paired_sentence_types = get_data_for_pairwise_comparisons(summary, eval_cfg=eval_cfg, diffs=get_diffs_accuracies)
	
	acc = []
	
	for pair in paired_sentence_types:
		x_data, y_data = get_single_pair_data(summary, pair, 'ratio_name')
		
		s1_ex = get_sentence_label(x_data)
		s2_ex = get_sentence_label(y_data)
		
		common_args = {
			 'acc'						: acc,
			 'colname'					: odds_ratio,
			 's1'						: pair[0], 
			 's2'						: pair[1], 
			 's1_ex'					: s1_ex, 
			 's2_ex'					: s2_ex,
			 'ratio_name'				: multiplator(x_data.ratio_name),
			 'predicted_arg'			: multiplator(x_data.ratio_name.str.split('/')[0], multstr='any'),
			f'position_ratio_name_ref'	: multiplator(x_data.position_ratio_name),
			f'position_ratio_name_gen'	: multiplator(y_data.position_ratio_name),
		}

		if eval_cfg.data.exp_type == 'newverb':
			common_args.update({'args_group': multiplator(x_data.args_group, multstr='any')})
		elif eval_cfg.data.exp_type == 'newarg':
			common_args.update({'predicted_role': multiplator(x_data.role_position, multstr='any')})
		
		update_acc(refs=x_data, gens=y_data, **common_args)
		
		if x_data.ratio_name.unique().size > 1:
			for name, x_group in x_data.groupby('ratio_name'):
				y_group = y_data[y_data.ratio_name == name]
				
				common_args.update({
					 'ratio_name'				: name,
					 'predicted_arg'			: name.split('/')[0],
					f'position_ratio_name_ref'	: multiplator(x_group.position_ratio_name),
					f'position_ratio_name_gen'	: multiplator(y_group.position_ratio_name),
				})
				
				if eval_cfg.data.exp_type == 'newarg':
					common_args.update({'predicted_role': x_group.role_position.unique()[0].split()[0]})
				
				update_acc(refs=x_group, gens=y_group, **common_args)
				
				if x_group.token.unique().size > 1:
					for token, x_token_group in x_group.groupby('token'):
						y_token_group = y_data[y_data.token == token]
						update_acc(refs=x_token_group, gens=y_token_group, **common_args)
	
	acc = pd.DataFrame(acc)
	
	return acc


# Tokenizer utilities
def create_tokenizer_with_added_tokens(
	model_id: str, tokens_to_mask: List[str], 
	delete_tmp_vocab_files: bool = True, **kwargs
) -> 'PreTrainedTokenizer':
	kwargs.update(dict(use_fast=False))
	
	if re.search(r'(^bert-)|(^distilbert-)', model_id):
		return create_bert_tokenizer_with_added_tokens(model_id, tokens_to_mask, delete_tmp_vocab_files, **kwargs)
	elif re.search(r'^roberta-', model_id):
		return create_roberta_tokenizer_with_added_tokens(model_id, tokens_to_mask, delete_tmp_vocab_files, **kwargs)	
	else:
		raise ValueError('Only BERT, DistilBERT, and RoBERTa tokenizers are currently supported.')

def create_bert_tokenizer_with_added_tokens(
	model_id: str, tokens_to_mask: List[str], 
	delete_tmp_vocab_files: bool = True, **kwargs
) -> Union['BertTokenizer', 'DistilBertTokenizer']:
	if 'uncased' in model_id:
		tokens_to_mask = [t.lower() for t in tokens_to_mask]
	
	bert_tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
	vocab = bert_tokenizer.get_vocab()
	assert not verify_tokens_exist(bert_tokenizer, tokens_to_mask), f'New token(s) already exist(s) in {model_id} tokenizer!'
	
	for token in tokens_to_mask:
		vocab.update({token: len(vocab)})
	
	with open('vocab.tmp', 'w', encoding='utf-8') as tmp_vocab_file:
		tmp_vocab_file.write('\n'.join(vocab))
	
	exec(f'from transformers import {bert_tokenizer.__class__.__name__}')
	
	tokenizer = eval(bert_tokenizer.__class__.__name__)(name_or_path=model_id, vocab_file='vocab.tmp', **kwargs)
	
	# for some reason, we have to re-add the [MASK] token to bert to get this to work, otherwise
	# it breaks it apart into separate tokens '[', 'mask', and ']' when loading the vocab locally (???)
	# this does not affect the embedding or the model's ability to recognize it
	# as a mask token (e.g., if you create a filler object with this tokenizer, 
	# it will identify the mask token position correctly)
	tokenizer.add_tokens(tokenizer.mask_token, special_tokens=True)
	tokenizer.model_max_length = bert_tokenizer.model_max_length
	
	if delete_tmp_vocab_files:
		os.remove('vocab.tmp')
	
	assert verify_tokens_exist(tokenizer, tokens_to_mask)
	
	return tokenizer

def create_roberta_tokenizer_with_added_tokens(
	model_id: str, tokens_to_mask: List[str], 
	delete_tmp_vocab_files: bool = True, **kwargs
) -> 'RobertaTokenizer':
	if 'uncased' in model_id:
		tokens_to_mask = [t.lower() for t in tokens_to_mask]
	
	roberta_tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
	assert not verify_tokens_exist(roberta_tokenizer, tokens_to_mask), f'New token {token} already exists in {model_id} tokenizer!'
	
	vocab = roberta_tokenizer.get_vocab()
	
	for token in tokens_to_mask:
		vocab.update({token: len(vocab)})
	
	with open('vocab.tmp', 'w', encoding = 'utf8') as tmp_vocab_file:
		json.dump(vocab, tmp_vocab_file, ensure_ascii=False)
	
	merges = [' '.join(key) for key in roberta_tokenizer.bpe_ranks.keys()]
	# we have to add a newline at the beginning of the file
	# since it's expecting it to be a comment, so we add a blank string here
	# that will get joined with a newline
	merges = [''] + get_roberta_bpes_for_new_tokens(tokens_to_mask) + merges
	merges = list(dict.fromkeys(merges)) # drops any duplicates we may have happened to add while preserving order
	with open('merges.tmp', 'w', encoding = 'utf-8') as tmp_merges_file:
		tmp_merges_file.write('\n'.join(merges))
	
	exec(f'from transformers import {roberta_tokenizer.__class__.__name__}')
	
	tokenizer = eval(roberta_tokenizer.__class__.__name__)(name_or_path=model_id, vocab_file='vocab.tmp', merges_file='merges.tmp', **kwargs)
	
	# for some reason, we have to re-add the <mask> token to roberta to get this to work, otherwise
	# it breaks it apart into separate tokens '<', 'mask', and '>' when loading the vocab and merges locally (???)
	# this does not affect the embeddings or the model's ability to recognize it
	# as a mask token (e.g., if you create a filler object with this tokenizer, 
	# it will identify the mask token position correctly)
	tokenizer.add_tokens(tokenizer.mask_token, special_tokens=True)
	tokenizer.model_max_length = roberta_tokenizer.model_max_length
	
	if delete_tmp_vocab_files:
		os.remove('vocab.tmp')
		os.remove('merges.tmp')
	
	assert verify_tokens_exist(tokenizer, tokens_to_mask), 'New tokens were not added correctly!'
	
	return tokenizer

def get_roberta_bpes_for_new_tokens(new_tokens: List[str]) -> List[str]:
	
	def gen_roberta_bpes(new_token: str, highest: bool = True) -> List[str]:
		# I do not completely understand how this works, but it does
		# a lot of trial and error is recorded here...
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
				return gen_roberta_bpes(chrs[1:]) + [' '.join([chrs[0], ''.join(chrs[1:])])]
		
		if len(chrs) % 2 == 0:
			pairs = gen_roberta_bpes(''.join(chrs[:-2]), highest = False)
			pairs += gen_roberta_bpes(''.join(chrs[-2:]), highest = False)
			pairs += tuple([''.join(chrs[:-2]), ''.join(chrs[-2:])])
			if not highest:
				return pairs
		else:
			pairs = gen_roberta_bpes(''.join(chrs[:-3]), highest = False)
			pairs += gen_roberta_bpes(''.join(chrs[-2:]), highest = False)
			pairs += gen_roberta_bpes(''.join(chrs[-3:]), highest = False)
			pairs += tuple([''.join(chrs[:-3]), ''.join(chrs[-3:])])
			if not highest:		
				return pairs
		
		pairs = tuple(zip(pairs[::2], pairs[1::2]))
		pairs = [' '.join(pair) for pair in pairs]
		# sp = chr(288)
		
		# pairs with the preceding special token
		# g_pairs = []
		# for pair in pairs:
		# 	if re.search(r'^' + ''.join(pair.split(' ')), new_token):
		# 		g_pairs.append(chr(288) + pair)
		
		# pairs = g_pairs + pairs
		# pairs = [f'{sp} {new_token[0]}'] + pairs
		
		pairs = list(dict.fromkeys(pairs)) # remove any duplicates
		
		return pairs
	
	roberta_bpes = [gen_roberta_bpes(new_token) for new_token in new_tokens]
	roberta_bpes = [pair for token in roberta_bpes for pair in token]
	return roberta_bpes

@recursor(str)
def format_roberta_tokens_for_tokenizer(data: str) -> str:
	if not data.startswith(chr(288)) and not data.startswith('^'):
		data = f'{chr(288)}{data}'
	elif data.startswith('^'):
		data = re.sub(r'^\^', '', data)
	
	return data

@recursor(str)
def format_roberta_tokens_for_display(token: str) -> str:
	if token.startswith(chr(288)):
		token = re.sub(rf'^{chr(288)}', '', token)
	elif not token.startswith('^'):
		token = f'^{token}'
	
	return token
	
@recursor(str)
def format_strings_with_tokens_for_display(
	data: str, 
	tokenizer_tokens: List[str],
	tokens_to_uppercase: List[str], 
	model_name: str, 
	string_id: str
):
	for token in listify(tokenizer_tokens):
		if model_name == 'roberta':
			if token.startswith(chr(288)) or token.startswith('^'):
				token = token[1:]
			
			data = re.sub(rf'(?<!{chr(288)}){token}', f'^{token}', data)
			data = re.sub(rf'{chr(288)}{token}', token, data)
		
		# this might need to be adjusted if we ever use an uncased roberta model,
		# since we'll need to check after modifying the token to token[1:] above
		elif 'uncased' in string_id and token in tokens_to_uppercase:
			data = re.sub(token, token.upper(), data)
	
	return data

@recursor(str)
def format_data_for_tokenizer(
	data: str, 
	mask_token: str, 
	string_id: str, 
	remove_punct: bool
) -> str:
	data = data.lower() if 'uncased' in string_id else data
	data = strip_punct(data) if remove_punct else data
	data = data.replace(mask_token.lower(), mask_token)
	return data

def verify_tokens_exist(
	tokenizer: 'PreTrainedTokenizer', 
	tokens: Union[List[str],List[int]]
) -> bool:
	token_ids = [tokenizer.convert_tokens_to_ids(token) if isinstance(token,str) else token if isinstance(token,int) else None for token in listify(tokens)]
	token_ids = [token_id for token_id in token_ids if token_id]
	return none(token_id == tokenizer.convert_tokens_to_ids(tokenizer.unk_token) for token_id in token_ids)

def verify_tokenization_of_sentences(
	tokenizer: 'PreTrainedTokenizer', 
	sentences: List[str], 
	tokens_to_mask: List[str] = None,
	**kwargs: Dict
) -> bool:
	"""
	verify that a custom tokenizer and one created using from_pretrained behave identically except on the tokens to mask
	"""
	tokens_to_mask = deepcopy(tokens_to_mask)
	assert verify_tokens_exist(tokenizer, tokens_to_mask), f'Tokens in {tokens_to_mask} were not correctly added to the tokenizer!'
	
	kwargs.update(dict(use_fast=False))
	
	tokenizer_id 	= tokenizer.name_or_path
	tokenizer_one 	= tokenizer
	tokenizer_two 	= AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
	sentences 		= flatten(sentences)
	
	if 'roberta' in tokenizer.name_or_path:
		# to replace the tokens in the sentences for comparison with roberta, we need to get the display versions
		# we can't do this on the tokenized sentences, because those will break apart the new tokens in different ways
		tokens_to_mask = format_roberta_tokens_for_display(tokens_to_mask)
		
	if tokens_to_mask:
		masked_sentences = []
		for sentence in sentences:
			for token in tokens_to_mask:
				sentence = sentence.replace(token, tokenizer.mask_token)
			
			masked_sentences.append(sentence)
	
	# we compare things by determining identity on the sentences with the new tokens replaced with mask tokens
	# we cannot do this by comparing the ids directly, since how surrounding material gets tokenized depends on whether there is a mask token present
	tokenizations_one = tokenizer_one(masked_sentences, return_tensors='pt', padding=True)['input_ids']
	tokenizations_two = tokenizer_two(masked_sentences, return_tensors='pt', padding=True)['input_ids']
	identical = torch.all(torch.eq(tokenizations_one, tokenizations_two))
	
	return identical


# analysis
def get_best_epoch(
	loss_df: pd.DataFrame, 
	method: str = 'mean', 
	log_message: bool = True
) -> int:
	
	loss_df = loss_df.copy().sort_values(['dataset', 'epoch']).reset_index(drop=True)
	
	if method in ['best_sumsq', 'sumsq']:
		best_losses = loss_df.loc[loss_df.groupby('dataset').value.idxmin()].reset_index(drop=True)
		
		epoch_sumsqs = []
		for dataset, df in best_losses.groupby('dataset'):
			dataset_epoch = df.epoch.unique()[0]
			epoch_losses = loss_df[loss_df.epoch == dataset_epoch].value
			epoch_avg_loss = epoch_losses.mean()
			sumsq = sum((epoch_avg_loss - epoch_losses)**2)
			epoch_sumsqs.append([dataset_epoch, sumsq])
		
		epoch_sumsqs = sorted(epoch_sumsqs, key=lambda epoch_sumsq: epoch_sumsq[1])
		best_epoch = epoch_sumsqs[0][0]
		if log_message:
			log.info(f'Best epoch is {best_epoch} (sumsq loss = ' + '{:.2f}'.format(epoch_sumsqs[0][1]) + f', minimum for {", ".join(best_losses[best_losses.epoch == best_epoch].dataset.values)}.')
		
	elif method in ['best_mean', 'mean']:
		mean_losses = loss_df.groupby('epoch').value.agg('mean')
		lowest_loss = mean_losses.min()
		best_epoch = mean_losses.idxmin()
		if log_message:
			log.info(f'Best epoch is {best_epoch} (mean loss = ' + '{:.2f}'.format(lowest_loss) + ').')
	else:
		best_epoch = loss_df.epoch.max()
		if log_message:
			log.warning(f'No method for determining best epoch provided (use "sumsq", "mean"). The highest epoch {best_epoch} will be used. This may not be the best epoch!')
			# this is so we don't print out the log message below if the best epoch isn't actually the highest one
			log_message = False
	
	if best_epoch == max(loss_df.epoch) and log_message:
		log.warning('Note that the best epoch is the final epoch. This may indicate underfitting.')
	
	return best_epoch

# Used with the R analysis script since it's much quicker to do this in Python
# went back to the old way of doing this because of https://github.com/rstudio/reticulate/issues/1166
# though now that we've implemented a class to do this here, it may not be necessary
def load_csvs(
	csvs: List[str], 
	**kwargs: Dict
) -> pd.DataFrame:
	csvs = [csvs] if isinstance(csvs, str) else csvs
	return pd.concat([pd.read_csv(f, **kwargs) for f in tqdm(csvs)], ignore_index=True)

def load_pkls(pkls: List[str]) -> pd.DataFrame:
	pkls = [pkls] if isinstance(pkls, str) else pkls
	return pd.concat([pd.read_pickle(f) for f in tqdm(pkls)], ignore_index=True)

# Used with the R analysis script since it's much quicker to do this in Python
def ungzip(files: List[str]) -> None:
	dests = [re.sub(r'\.gz$', '', f) for f in files]
	fs_dests = tuple(zip(files, dests))
	for f, dest in tqdm(fs_dests):
		with gzip.open(f, 'rb') as f_in, open(dest, 'wb') as f_out:
			copyfileobj(f_in, f_out)

# Used with the R analysis script since it's much quicker to do this in Python
def delete_files(files: List[str]) -> None:
	for f in tqdm(files):
		try: 
			os.remove(f)
		except:
			print(f'Unable to remove "{f}".')
			continue

class gen_summary():
	"""
	Generates summary statistics for saved dataframes, allowing R-style unquoted arguments to functions
	for convenience, via manipulation of globals()
	"""
	def __init__(self, 
		df: Union[pd.DataFrame,str,'TextIOWrapper','StringIO'], 
		columns: List = ['gen_given_ref', 'correct', 'odds_ratio_pre_post_diff', 'odds_ratio', 'cossim'],
		funs: List = ['mean', 'sem']
	) -> 'gen_summary':
		"""
		Create a gen_summary instance
		
			params:
				df 				: a pandas dataframe, str, or file-like object
				columns 		: a list of columns to generate summary statistics for that are in df.columns
				funs 			: a list of summary statistics to generate using pandas' agg function
			
			returns:
				gen_summary 	: a summary generator that can be called with (un)quoted column names
								  to generate a summary of columns in df
		"""
		# allow passing a filepath
		if not isinstance(df, pd.DataFrame):
			df = pd.read_csv(df)
			
		self.df 		= df
		self.columns 	= [c for c in columns if c in self.df.columns]
		self.funs 		= funs
		self.prep_quasiquotation()
	
	def __call__(self, *columns: Tuple) -> pd.DataFrame.groupby:
		"""
		Generate a summary using the columns specified, print and return it
			
			params:
				*columns (tuple)				: a list of unquote names of columns in summary to group summary statistics by
			
			returns:
				results (pd.DataFrame.groupby) 	: a summary consisting of self.funs applied to the columns in self.columns
												  grouped by the passed columns
		"""
		agg_dict = {c : self.funs for c in self.columns if c in self.df.columns}
		results = self.df.groupby(list(columns)).agg(agg_dict).sort_values(list(columns))
		print(results)
		return results
	
	def prep_quasiquotation(self) -> None:
		"""
		Adds column names to global variables to facilitate R-like quasiquotation usage.
		As a failsafe, if a variable is already defined with a conflicting definition,
		we return early without adding any names. In this case, quoted strings may be provided to generate a summary.
		"""
		for c in self.df.columns:
			if c in globals() and not globals()[c] == c:
				print(f"{c} already has a conflicting definition: '{globals()[c]}'. Exiting.")
				return
		
		for c in self.df.columns:
			globals()[c] = c
	
	def set_funs(self, *funs: Tuple) -> None:
		"""
		Set the summary functions to use for this gen_summary
			
			params: 
				funs (tuple): 	a tuple of functions known to pandas by string 
								name used to generate summary statistics
		"""
		self.funs = list(funs)
	
	def set_columns(*columns: Tuple) -> None:
		"""
		Set the numerical columns to get summary statistics for
			
			params:
				columns (tuple): a tuple of columns in self.df to generate summary statistics for
		"""
		self.columns = [c for c in list(columns) if c in self.df.columns]
	
	def set_df(self, df: Union[pd.DataFrame,str,'TextIOWrapper','StringIO']) -> None:
		"""
		Set the df to a different one, and attempt to reinitialize all variables
			
			params:
				df (pd.DataFrame,str,IO):	a dataframe object, filehandler, or path to a csv
		"""
		if not isinstance(df, pd.DataFrame):
			df = pd.read_csv(df)
		
		self.df = df
		
		# reset the columns to remove any that aren't present in the new df
		set_columns(self.df.columns)
		
		# reinitialize quasiquotation variables using the columns from the new df
		# note that if the new df uses column names already defined in the previous df
		# this will throw an error, but some unquoted string names may still work
		self.prep_quasiquotation()
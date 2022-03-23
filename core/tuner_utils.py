# tuner_utils.py
#
# utility functions for tuner.py
import os
import re
import json
import gzip
import torch
import random
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

# order of grammatical functions for sorting things in display order
GF_ORDER = ['[subj]', '[obj]', '[2obj]', '[iobj]', '[obl]', '[pobj]', '[adj]']

# short useful functions
def listify(l: 'any') -> List:
	
	return l if isinstance(l, list) else OmegaConf.to_container(l) if isinstance(l, ListConfig) else [l]

def unlistify(l: 'any') -> 'any':
	if isinstance(l, dict):
		return l[list(l.keys())[0]] if len(l) == 1 else l
	
	return l[0] if isinstance(l, list) and len(l) == 1 else l

def none(iterator: 'Iterator') -> bool:
	
	return not any(iterator)

def multiplator(series: Union[List,pd.Series], multstr = 'multiple'):
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
):
	if isinstance(data, DictConfig) or isinstance(data, ListConfig):
		# we want the primitive versions of these so we can modify them
		data = OmegaConf.to_container(data)
	
	data = deepcopy(data)		
	
	if isinstance(data,t):
		return fun(data, *args, **kwargs)
	elif isinstance(data,dict):
		return {apply_to_all_of_type(k, t, fun, *args, **kwargs): apply_to_all_of_type(v, t, fun, *args, **kwargs) for k, v in data.items()}
	elif isinstance(data,list):
		return [apply_to_all_of_type(i, t, fun, *args, **kwargs) for i in data]
	elif isinstance(data, pd.Series):
		return pd.Series([apply_to_all_of_type(i, t, fun, *args, **kwargs) for i in data])
	else:
		return data


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
	
	def get_pairs(summary: pd.DataFrame, eval_cfg: DictConfig = None, cossims: bool = False) -> List[Tuple[str]]:
		if not cossims:
			# Get each unique pair of sentence types so we can create a separate plot for each pair
			types = summary.sentence_type.unique()
			types = sorted(types, key = lambda s_t: eval_cfg.data.sentence_types.index(s_t))
		else:
			types = summary.predicted_arg.unique()
		
		pairs = [pair for pair in list(itertools.combinations(types, 2)) if not pair[0] == pair[1]]
		
		if not cossims:
			# Sort so that the trained cases are first
			reference_sentence_type = multiplator(summary.reference_sentence_type)
			pairs = [sorted(pair, key = lambda x: str(-int(x == reference_sentence_type)) + x) for pair in pairs]
			
			# Filter to only cases including the reference sentence type for ease of interpretation
			pairs = [(s1, s2) for s1, s2 in pairs if s1 == reference_sentence_type] if reference_sentence_type != 'none' else pairs
		else:
			pairs = list(set(tuple(sorted(pair)) for pair in pairs))
		
		return pairs
	
	def format_summary_for_comparisons(summary: pd.DataFrame, exp_type: str = None, cossims: bool = False, diffs: bool = False) -> Tuple[pd.DataFrame,Tuple[str]]:
		if not cossims:
			columns = ['token'] if exp_type == 'newverb' else ['ratio_name'] if exp_type == 'newarg' else None
		else:
			columns = ['predicted_arg', 'target_group']
		
		summary = summary.copy()
		
		for column in columns:
			# if we are dealing with bert/distilbert and roberta models, replace the strings with uppercase ones for comparison
			if 'roberta' in summary.model_name.unique() and summary.model_name.unique().size > 1 and exp_type == 'newarg':
				# if we are dealing with multiple models, we want to compare them by removing the idiosyncratic variation in how
				# tokenization works. bert and distilbert are uncased, which means the tokens are converted to lower case.
				# here, we convert them back to upper case so they can be plotted in the same group as the roberta tokens,
				# which remain uppercase
				summary.loc[summary.model_name != 'roberta', column] = \
					summary[summary.model_name != 'roberta'][column].str.upper()
				
			# for roberta, strings with spaces in front of them are tokenized differently from strings without spaces
			# in front of them. so we need to remove the special characters that signals that, and add a new character
			# signifying 'not a space in front' to the appropriate cases instead
			
			# first, check whether doing this will alter information
			if 'roberta' in summary.model_name.unique():
				roberta_summary = summary[summary.model_name == 'roberta'].copy()
				num_tokens_in_summary = len(set([i for i in [j.split('/') for j in roberta_summary[column].unique()]]))
				
				roberta_summary[column] = [re.sub(chr(288), '', ratio_name) for ratio_name in roberta_summary[column]]
				num_tokens_after_change = len(set([i for i in [j.split('/') for j in roberta_summary[column].unique()]]))
				if num_tokens_in_summary != num_tokens_after_change:
					# this isn't going to actually get rid of any info, but it's worth logging
					log.warning('RoBERTa tokens were used with and without preceding spaces. This may complicate comparing results to BERT models.')
				
				# first, replace the ones that don't start with spaces before with a preceding ^
				summary.loc[(summary.model_name == 'roberta') & ~(summary[column].str.startswith(chr(288))), column] = \
					summary[(summary.model_name == 'roberta') & ~(summary[column].str.startswith(chr(288)))][column].str.replace(r'((^.)|(?<=\/).)', r'^\1', regex=True)
				
				# then, replace the ones with the preceding special character (since we are mostly using them in the middle of sentences)
				summary.loc[(summary.model_name == 'roberta') & (summary[column].str.startswith(chr(288))), column] = \
					summary[(summary.model_name == 'roberta') & (summary[column].str.startswith(chr(288)))][column].str.replace(chr(288), '')
				
		if exp_type == 'newverb' and not cossims:
			# Sort by grammatical function prominence for newverb exps (we do this because 'subj' alphabetically follows 'obj'), but it's more natural for it to go first
			summary['gf_ratio_order'] = [GF_ORDER.index(gf_ratio_name.split('/')[0]) for gf_ratio_name in summary.gf_ratio_name]
			summary = summary.sort_values(['model_id', 'gf_ratio_order'])
			summary = summary.drop('gf_ratio_order', axis=1)
			summary['gf_ratio_name'] = [re.sub(r'\[|\]', '', gf_ratio_name) for gf_ratio_name in summary.gf_ratio_name]
		
		colnames = get_eval_metrics_colnames(exp_type, cossims, diffs)
		metric = colnames[-2]
		
		if summary.model_id.unique().size > 1:
			metric = f'{metric}_mean'
		else:
			summary[f'{metric}_sem'] = 0
		
		return summary, colnames
	
	def get_eval_metrics_colnames(exp_type: str, cossims: bool = False, diffs: bool = False) -> Tuple[str]:
		if not cossims:
			colnames = ['odds_ratio'] if exp_type == 'newarg' or not diffs else ['odds_ratio_pre_post_difference'] if exp_type == 'newverb' else [None]
		elif cossims:
			colnames = ['cossim']
		
		colnames.append(f'{colnames[-1]}_sem')
		
		return tuple(colnames)
	
	exp_type = eval_cfg.data.exp_type if eval_cfg is not None else None
	summary, colnames = format_summary_for_comparisons(summary, exp_type, cossims=cossims, diffs=diffs)
	
	pairs = get_pairs(summary, eval_cfg, cossims)
	
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
):
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
):
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

def format_roberta_tokens(tokens: Union[str,List[str]], dest: str = 'tokenizer') -> List[str]:
	if not dest in ['tokenizer', 't', 'display', 'd']:
		
		raise ValueError('Invalid format specified. Must be one of t(okenizer), d(isplay)')
	
	dest = dest[0]
	
	def format_single_roberta_token(token: str, dest: str):
		if dest == 't':
			if not token.startswith(chr(288)) and not token.startswith('^'):
				token = f'{chr(288)}{token}'
			elif token.startswith('^'):
				token = re.sub(r'^\^', '', token)
		elif dest == 'd':
			if token.startswith(chr(288)):
				token = re.sub(rf'^{chr(288)}', '', token)
			elif not token.startswith('^'):
				token = f'^{token}'
		
		return token
	
	return apply_to_all_of_type(data=tokens, t=str, fun=format_single_roberta_token, dest=dest)

def format_roberta_tokens_for_display(*args, **kwargs) -> List[str]:
	
	return format_roberta_tokens(*args, **kwargs, dest='display')

def format_roberta_tokens_for_tokenizer(*args, **kwargs) -> List[str]:
	
	return format_roberta_tokens(*args, **kwargs, dest='tokenizer')
	
def format_strings_with_tokens_for_display(
	s: List[str], 
	tokenizer_tokens: Union[str,List[str]], 
	model_name: str,
	string_id: str,
) -> List[str]:
	
	def format_single_string_with_tokens_for_display(s: str, tokens: List[str], model_name: str, string_id: str):
		for token in listify(tokens):
			if model_name == 'roberta':
				if token.startswith(chr(288)) or token.startswith('^'):
					token = token[1:]
				
				s = re.sub(rf'(?<!{chr(288)}){token}', f'^{token}', s)
				s = re.sub(rf'{chr(288)}{token}', token, s)
			
			if 'uncased' in string_id:
				s = re.sub(token, token.upper(), s)
		
		return s
		
	return apply_to_all_of_type(
		data=s, t=str, 
		fun=format_single_string_with_tokens_for_display, 
		tokens=tokenizer_tokens,
		model_name=model_name,
		string_id=string_id,
	)


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
	
	# flatten a list of lists of strings without breaking string into characters
	# from https://stackoverflow.com/questions/5286541/how-can-i-flatten-lists-without-splitting-strings
	flatten 	= lambda y: [k for j in ([i] if not isinstance(i,list) else flatten(i) for i in y) for k in j]
	sentences 	= flatten(sentences)
	
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

# evaluation
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


# analysis

# Used with the R analysis script since it's much quicker to do this in Python
# went back to the old way of doing this because of https://github.com/rstudio/reticulate/issues/1166
# though now that we've implemented a class to do this here, it may not be necessary
def load_csvs(csvs: List[str], converters: Dict) -> pd.DataFrame:
	csvs = [csvs] if isinstance(csvs, str) else csvs
	return pd.concat([pd.read_csv(f, converters=converters) for f in tqdm(csvs)], ignore_index=True)

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
		we return early. In this case, strings may be provided to generate a summary.
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
		
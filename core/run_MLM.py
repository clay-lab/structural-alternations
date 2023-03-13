########################################################################################################
# This script is (heavily) adapted from a script by HuggingFace Inc. for Seq2Seq agreement attraction. #
# It has been modified for use with a masked language modeling task by Michael Wilson (2023).		   #
########################################################################################################
#
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import json
import torch
import logging
import itertools
import transformers

import pandas as pd
import torch.nn.functional as F

from glob import glob
from tqdm import tqdm
from copy import deepcopy
from typing import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Dataset
from dataclasses import dataclass, field
from transformers import (
	AutoConfig,
	AutoTokenizer,
	HfArgumentParser,
	AutoModelForMaskedLM,
)

from tuner import Tuner

def model_load_functions(model_name_or_path: str) -> Callable:
	'''
	Returns the appropriate function for loading a model
	based on its name.
	'''
	return AutoModelForMaskedLM.from_pretrained

@dataclass
class ModelArguments:
	'''Arguments pertaining to which model/config/tokenizer we are going to evaluate.'''
	model_name_or_path: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
	)
	
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
	)
	
	model_revision: str = field(
		default="main",
		metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
	)
	
	use_auth_token: bool = field(
		default=False,
		metadata={
			"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
			"with private models)."
		},
	)
	
	use_gpu: bool = field(
		default=False,
		metadata={
			"help": "Whether to load the model on the GPU or not."
		}
	)

@dataclass
class DataArguments:
	'''Arguments pertaining to what data we are going to input our model for evaluation.'''
	output_dir: str = field(
		default=None,
		metadata={"help": "Where to store the results of the evaluation."}
	)
	
	test_file: str = field(
		default=None,
		metadata={"help": "The test data file to evaluate model predictions for."},
	)
	
	overwrite_cache: bool = field(
		default=False, 
		metadata={"help": "Overwrite the cached training and evaluation sets"}
	)
	
	preprocessing_num_workers: Optional[int] = field(
		default=None,
		metadata={"help": "The number of processes to use for the preprocessing."},
	)
	
	max_source_length: Optional[int] = field(
		default=1024,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	
	per_device_test_batch_size: int = field(
		default=16,
		metadata={
			"help": "The batch size for evaluation data."
		}
	)
	
	pad_to_max_length: bool = field(
		default=False,
		metadata={
			"help": "Whether to pad all samples to model maximum sentence length. "
			"If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
			"efficient on GPU but very bad for TPU."
		},
	)
	
	max_test_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes, truncate the number of test examples to this "
			"value if set."
		},
	)
	
	def __post_init__(self):
		if self.test_file is None:
			raise ValueError("Need a test file.")
		elif self.test_file is not None:
			extension = self.test_file.split('.')[-1]
			assert extension == 'data', "`test_file` should be a data file."
		
		if self.output_dir is None:
			self.output_dir = os.path.join('outputs', os.path.basename(self.test_file).replace('.data', ''))

def setup_logging() -> logging.Logger:
	'''Handles logging setup.'''
	logger = logging.getLogger(__name__)
	
	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)
	
	transformers.utils.logging.set_verbosity_error()
	
	return logger

def parse_cl_arguments(*args: Tuple) -> Tuple:
	'''
	Parse command line arguments into ModelArguments and DataArguments.
	See ModelArguments and DataArguments for details.
	'''
	parser = HfArgumentParser(args)
	model_args, data_args = parser.parse_args_into_dataclasses()
	
	return model_args, data_args

def load_tokenizer_model_and_added_args(model_args: ModelArguments) -> Tuple:
	'''Loads the tokenizer and model as specified in model_args.'''
	# If we're loading from the hub
	added_args = []
	
	if not os.path.exists(model_args.model_name_or_path):
		config = AutoConfig.from_pretrained(
			model_args.config_name if model_args.config_name else model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		
		tokenizer = AutoTokenizer.from_pretrained(
			model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			use_fast=model_args.use_fast_tokenizer,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		
		model = model_load_functions(model_args.model_name_or_path)(
			model_args.model_name_or_path,
			config=config,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
	else:
		tuner = Tuner(model_args.model_name_or_path, use_gpu=model_args.use_gpu)
		tuner._restore_original_random_seed()
		_, _ = tuner.restore_weights()
		
		if hasattr(tuner, 'args'):
			added_args = tuner._format_strings_with_tokens_for_display(list(tuner.args.values()))
		
		tokenizer = tuner.tokenizer
		model = tuner.model
	
	return tokenizer, model, added_args

def preprocess_dataset(
	dataset: 'Dataset', 
	data_args: DataArguments, 
	tokenizer: AutoTokenizer
) -> 'Dataset':
	'''
	Formats the dataset for use with a masked language model.
	
	params:
		dataset (Dataset)			: a huggingface dataset. Must contain a "test" split,
									  with examples in the "text" column.
		data_args (DataArguments)	: the arguments containing information about the data.
					  				  see the DataArguments class for more details.
		tokenizer (AutoTokenizer)	: the tokenizer to use to prepare the examples for the model.
	
	returns:
		Dataset 					: the dataset formatted for use with an AutoModelForMaskedLM.
	'''
	drop_cols 			= dataset['test'].column_names
	
	def preprocess_function(examples: 'Batch') -> Dict:
		'''Tokenizes a batch of string inputs.'''
		if tokenizer.mask_token != '[MASK]':
			examples['text'] = [text.replace('[MASK]', tokenizer.mask_token) for text in examples['text']]
		
		if 'uncased' in tokenizer.name_or_path:
			examples['text'] = [text.lower() for text in examples['text']]
			examples['text'] = [text.replace(tokenizer.mask_token.lower(), tokenizer.mask_token) for text in examples['text']]
		
		model_inputs 	= tokenizer(
							examples['text'], 
							max_length=data_args.max_source_length, 
							padding=True,
							truncation=True
						)
		
		return model_inputs
	
	test_dataset 		= dataset['test']
	
	if data_args.max_test_samples is not None:
		test_dataset 	= test_dataset.select(range(data_args.max_test_samples))
	
	test_dataset 		= test_dataset.map(
		preprocess_function,
		batched=True,
		num_proc=data_args.preprocessing_num_workers,
		remove_columns=drop_cols,
		load_from_cache_file=not data_args.overwrite_cache,
	)
	
	test_dataset.set_format(type='torch')
	
	return test_dataset

def check_ids(
	tokenizer: AutoTokenizer,
	eval_tokens: List[List[str]],
	eval_token_ids: List[List[int]],
) -> None:
	# check that eval tokens make sense
	for tokens, token_ids in zip(eval_tokens, eval_token_ids):
		if (
			any(token_id == tokenizer.unk_token_id for token_ids in eval_token_ids for token_id in token_ids) or 
			any(len(token_id) > 1 for token_id in token_ids)
		):
			raise ValueError(
				f'Some tokens used for evaluation are not tokenized as single words!:\n\n' +
				"\n".join(
					[str(t) for t in zip(tokens, token_ids) if any(token_id == tokenizer.unk_token_id for token_id in t[-1]) or len(t[-1]) > 1]
				)
			)

def evaluate_model(
	model_args: ModelArguments,
	model: AutoModelForMaskedLM,
	tokenizer: AutoTokenizer,
	test_dataset: Dataset,
	data_args: DataArguments,
	added_args: List[str],
) -> None:
	'''
	Evaluates a model on the test dataset using masked language modeling.
	Saves results in data_args.output_dir as a csv.gz.
	
	params:
		model_args (ModelArguments)			: used to assign a unique identifier for the directory
											  the model is in
		model (AutoModelForMaskedLM)		: the model to evaluate
		tokenizer (AutoTokenizer)			: the tokenizer for the model
		test_dataset (Dataset)				: the dataset to evaluate on
											  Should consist of sentences where
											  the position to be evaluated is 
											  replaced with "<extra_id_0>". Currently
											  evaluation of more than one replaced span is
											  not supported.
		data_args (DataArguments)			: the arguments containing information about the data.
											  see the DataArguments class for more details.
		added_args (List[str])				: additional tokens to evaluate for each example
											  not specified in the metadata file.
											  this is used for newverb experiments to allow the addition
											  of the individually chosen subject+object tokens
											  for each model
	raises:
		ValueError 							: if eval_tokens for any sentence are not tokenized
											  as single tokens, predictions are hard to interpret,
											  so a ValueError is raised in this case.
	'''
	# this removes any slashes in the model name, which
	# causes problems
	basename = os.path.basename(model.name_or_path)
	output_pred_file = os.path.join(data_args.output_dir, 'mlm_results.csv.gz')
	
	# do not reevaluate if the output file already exists
	if os.path.exists(output_pred_file):
		# return
		pass
	
	def pad_tensor(t: torch.Tensor, pad: int, dim: int = -1) -> torch.Tensor:
		'''
		Pads a tensor to length pad in dim dim.
		From https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8?u=mawilson
		
			params:
				t (torch.Tensor): tensor to pad
				pad (int)		: the size to pad to
				dim (int)		: dimension to pad
			
			returns:
				a new torch.Tensor padded to 'pad' in dimension 'dim'
		'''
		pad_size = list(t.shape)
		pad_size[dim] = pad - t.size(dim)
		return torch.cat([t, torch.zeros(*pad_size, dtype=t.dtype, device=t.device)], dim=dim)
	
	def pad_batch(batch: Tuple) -> Tuple:
		'''Pads examples in a batch to the same length.'''
		max_len = max(map(lambda ex: ex['input_ids'].size(-1), batch))
		batch 	= list(map(lambda ex: {k: pad_tensor(ex[k], pad=max_len, dim=-1) for k in ex}, batch))
		batch 	= {k: torch.stack([ex[k] for ex in batch], dim=0) for k in batch[0].keys()}
		return batch

	with open(data_args.test_file.replace('.data', '_metadata.json'), 'rt', encoding='utf-8') as in_file:
		metadata = [json.loads(l) for l in in_file.readlines()]
	
	warned = False
	for d in metadata:
		d["eval_tokens"] += list(itertools.chain(*added_args))
		if not "added_tokens" in d:
			d["added_tokens"] = ','.join(list(itertools.chain(*added_args)))
		elif not warned:
			warned = True
			print('Warning: additional arguments were added, but the key "added_args" already exists in the metadata.')
			print('Which arguments were added programmatically will not be recorded in the results.')
	
	os.makedirs(data_args.output_dir, exist_ok=True)
	
	dataloader = DataLoader(
					test_dataset,
					batch_size=data_args.per_device_test_batch_size,
					collate_fn=pad_batch
				)
	
	model.eval()
	
	n_observed_examples = 0
	metrics = []
	for inputs in tqdm(dataloader):
		n_examples_in_batch = inputs['input_ids'].shape[0]
		
		# use this as a unique input identifier
		input_nums = range(n_observed_examples, n_observed_examples + n_examples_in_batch)
		n_observed_examples += n_examples_in_batch
		
		batch_metadata = metadata[(n_observed_examples - n_examples_in_batch):n_observed_examples]
		eval_tokens = [d['eval_tokens'] for d in batch_metadata]
		
		metrics.extend(evaluate_batch(
			model=model, 
			tokenizer=tokenizer,
			inputs=inputs, 
			input_nums=input_nums,
			eval_tokens=eval_tokens,
			batch_metadata=batch_metadata
		))
	
	metrics = pd.DataFrame(metrics)
	
	metrics = metrics.assign(
		model_name=re.sub('["\']', '', model.config.name_or_path),
		model_path=model_args.model_name_or_path,
		test_dataset=os.path.basename(data_args.test_file).replace('.txt.gz', ''),
	)
	
	move_to_beginning = ['model_name', 'test_dataset']
	metrics = metrics[move_to_beginning + [c for c in metrics.columns if not c in move_to_beginning]]
	
	metrics.to_csv(output_pred_file, index=False, na_rep='NaN')

def evaluate_batch(
	model: AutoModelForMaskedLM,
	tokenizer: AutoTokenizer,
	inputs: Dict[str,torch.Tensor],
	input_nums: List[int] = None,
	eval_tokens: List[List[str]] = None,
	batch_metadata: List[Dict] = None,
) -> List[Dict]:
	'''Evaluate a single batch of inputs on the eval tokens, depending on the model type.'''
	if input_nums is None:
		input_nums = range(len(inputs['input_ids'].shape[0]))
	
	if eval_tokens is None:
		raise ValueError(f'No tokens were provided for evaluation.')
	
	if len(eval_tokens) != len(inputs['input_ids']):
		raise ValueError(
			f'{len(eval_tokens)} sets of eval tokens were '
			f'provided for {len(inputs["input_ids"])} sentences.'
		)
		
	if batch_metadata is None:
		batch_metadata = {}
	
	pred_token_indices = [
		(sequence == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
		for sequence in inputs['input_ids']
	]
	
	eval_tokens, eval_token_ids = get_eval_token_ids(
		tokenizer=tokenizer, 
		pred_token_indices=pred_token_indices, 
		eval_tokens=eval_tokens
	)
	
	model_eval_function = get_model_eval_function(model_name_or_path=model.name_or_path)
	
	return model_eval_function(
		model=model, 
		tokenizer=tokenizer, 
		inputs=inputs, 
		input_nums=input_nums, 
		eval_tokens=eval_tokens, 
		eval_token_ids=eval_token_ids,
		pred_token_indices=pred_token_indices,
		batch_metadata=batch_metadata
	)

def get_model_eval_function(model_name_or_path: str) -> Callable:
	'''
	Returns the appropriate function for eval based on the kind of 
	model.
	'''
	return evaluate_MLM_batch

def get_eval_token_ids(
	tokenizer: AutoTokenizer,
	pred_token_indices: torch.Tensor, 
	eval_tokens: List[List[str]]
) -> List[List[int]]:
	'''
	Get the eval token ids depending on their position
	in the input sequence (beginning of sentence or not).
	'''
	bos_index = 1 if tokenizer.bos_token is not None or tokenizer.cls_token is not None else 0
	
	if 'uncased' in tokenizer.name_or_path:
		preprocess_function = lambda s: s.lower()
	else:
		preprocess_function = lambda s: s
	
	# we might need multiple version
	# of a requested eval token depending
	# on where in the sentence it goes
	# so we need to return the actual ones to be
	# able to line them up
	actual_eval_tokens = []
	eval_token_ids = []
	
	for i, (token_indices, tokens) in enumerate(zip(pred_token_indices, eval_tokens)):
		actual_eval_tokens.append([])
		eval_token_ids.append([])
		for token_index in token_indices:
			if token_index == bos_index:
				actual_words = [preprocess_function(t) for t in tokens]
			else:
				actual_words = [preprocess_function(f' {t}') for t in tokens]
				
			tokenized = tokenizer(
				actual_words,
				add_special_tokens=False,
				return_attention_mask=False,
			)['input_ids']
			
			actual_eval_tokens[i].extend(actual_words)
			eval_token_ids[i].extend(tokenized)
	
	no_duplicates = [dict(zip(words, token_ids)) for words, token_ids in zip(actual_eval_tokens, eval_token_ids)]
	actual_eval_tokens = [list(d.keys()) for d in no_duplicates]
	eval_token_ids = [list(d.values()) for d in no_duplicates]
	
	# check that the eval tokens are single tokens
	check_ids(tokenizer=tokenizer, eval_tokens=actual_eval_tokens, eval_token_ids=eval_token_ids)
	
	eval_token_ids = [[id for t in token_ids for id in t] for token_ids in eval_token_ids]
	
	return actual_eval_tokens, eval_token_ids

def evaluate_MLM_batch(
	model: AutoModelForMaskedLM,
	tokenizer: AutoTokenizer,
	inputs: Dict[str,torch.Tensor],
	input_nums: List[int],
	eval_tokens: List[List[str]],
	eval_token_ids: List[List[int]],
	pred_token_indices: List[int],
	batch_metadata: List[Dict],
) -> List[Dict]:
	'''
	Evaluates a batch of examples for a Masked Language Model.
	For each input, determines the log probability of each eval token
	as a prediction for the mask token.
	'''
	with torch.no_grad():
		batch_outputs = model(**inputs)
	
	# get the logprobs for the indices where the predictions are to be found
	batch_scores = [t[i][:len(tokenizer.get_vocab())] for t, i in zip(batch_outputs.logits, pred_token_indices)]
	batch_logprobs = [F.log_softmax(batch_score, dim=-1) for batch_score in batch_scores]
	
	# record metrics
	metrics = []
	records = zip(input_nums, inputs['input_ids'], batch_outputs.logits, eval_tokens, eval_token_ids, batch_logprobs, batch_metadata)
	for input_num, input_seq, pred_seq, tokens, token_ids, scores, example_metadata in records:
		if tokenizer.pad_token_id in input_seq.tolist():
			last_index_to_decode = input_seq.tolist().index(tokenizer.pad_token_id)-1
		else:
			last_index_to_decode = len(input_seq.tolist())
		
		for i, score in enumerate(scores):
			metrics.extend(
				[
					{
						'item': input_num,
						'input_text': tokenizer.decode(input_seq).replace(f'{tokenizer.sep_token}', '').replace(f'{tokenizer.pad_token}', '').replace(f'{tokenizer.cls_token}', '').strip().replace('  ', ' '),
						'pred_seq': tokenizer.decode(torch.argmax(pred_seq, dim=-1)[1:last_index_to_decode], skip_special_tokens=True),
						'token': token,
						'token_id': token_id,
						'eval_position_order': i,
						'logprob': score[token_id].item(),
						**{k: v for k, v in example_metadata.items() if not k == 'eval_tokens'}
					} for token, token_id in zip(tokens, token_ids)
				]
			)
	
	return metrics

def _run_MLM(dataset: Dataset, model_args: ModelArguments, data_args: DataArguments) -> None:
	tokenizer, model, added_args = load_tokenizer_model_and_added_args(model_args)
	test_dataset = preprocess_dataset(dataset, data_args, tokenizer)
	
	evaluate_model(model_args, model, tokenizer, test_dataset, data_args, added_args)

def run_MLM() -> None:
	logger = setup_logging()
	
	model_args, data_args = parse_cl_arguments(ModelArguments, DataArguments)
	logger.info(f'Evaluation parameters: {data_args}')
	
	dataset = load_dataset('text', data_files={'test': data_args.test_file})
	
	if model_args.model_name_or_path.startswith('glob('):
		# strip off the glob function to get the expression
		model_args.model_name_or_path = model_args.model_name_or_path[len('glob('):-len(')')]
		all_paths = sorted(glob(model_args.model_name_or_path, recursive=True))
		start_dir = os.getcwd()
		
		# convert relative path to absolute path since loading a tuner changes
		# the working directory. we've already loaded the dataset, but this
		# will need to be set for the metadata to be located correctly
		if not data_args.test_file.startswith('/'):
			data_args.test_file = os.path.join(start_dir, data_args.test_file)
		
		for path in all_paths:
			model_args_new = deepcopy(model_args)
			model_args_new.model_name_or_path = path
			
			_run_MLM(dataset, model_args_new, data_args)
			os.chdir(start_dir)
	else:
		_run_MLM(dataset, model_args, data_args)

if __name__ == '__main__':
	run_MLM()

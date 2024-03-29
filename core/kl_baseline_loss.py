# Implements a loss that combines loss on the new data
# with the KL divergence between the updated model's predictions
# and the pretrained model's predictions
import logging

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import KLDivLoss

from tqdm import tqdm
from typing import *

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import logging as lg
lg.set_verbosity_error()

from datasets import load_dataset, Dataset, DatasetDict
from datasets.utils import logging as dataset_utils_logging
from datasets.utils import disable_progress_bar
disable_progress_bar()
dataset_utils_logging.set_verbosity_error()

import tuner_utils

log = logging.getLogger(__name__)

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

class KLBaselineLoss(KLDivLoss):
	'''
	Calculates the loss on a dataset based on the KL divergence between the predictions
	of a model being fine-tuned and a pretrained version of that model. Higher KL divergences
	mean the model's predictions have deviated more from the baseline. This penalizes deviation
	from the baseline model to avoid catastrophic forgetting. You should add the KLBaselineLoss
	to another loss against the fine-tuning targets during fine-tuning. Otherwise, you would
	just be optimizing for the behavior of the original model and not striking the intended
	balance between performance on tuning data and the original model's behavior.
	
	Adapted from torch.nn.modules.loss.KLDivLoss
	'''
	def __init__(
		self,
		model: 'PreTrainedModel',
		tokenizer: 'PreTrainedTokenizer',
		dataset: Dataset,
		batch_size: int = 1,
		scaleby: float = 1.,
		n_examples_per_step: int = None,
		masking: str = 'none',
		model_kwargs: Dict = {},
		tokenizer_kwargs: Dict = {},
		size_average = None,
		reduce = None,
		reduction: str = 'batchmean',
	) -> None:
		'''
		Creates a KL divergence loss object that codes divergence of a fine-tuned
		model compared to a baseline model on a specified dataset.
		
			params:
				model (PreTrainedModel)				: a huggingface pretrained model
				tokenizer (PreTrainedTokenizer)		: a huggingface pretrained tokenizer (should match the model)
				dataset (Dataset)					: a dataset in huggingface's datasets format that
													  has been pretokenized for use with the same kind of tokenizer
													  as passed
				batch_size (int)					: the number of sentences to run through the models at a single time.
													  KL divergence is computed per sentence and averaged
				scaleby (float)						: returned loss is multiplied by this
				n_examples_per_step (int)			: it may be too time-consuming to calculate the KLBaselineLoss on
													  the basis of the entire dataset, if the dataset is large.
													  you can use this to set how many random samples to draw
													  from dataset to use when calculating loss. If not set,
													  all examples will be used each time.
				masking (str)						: which style of masking to use when calculating the loss
													  valid options are the following.
													  "always": choose 15% of tokens to randomly mask per sentence, and replace with mask tokens
													  "bert"  : choose 15% of tokens to randomly mask per sentence. replace 80% with mask token, 10% with original token, 10% with random word.
													  "none"  : don't mask any tokens. KL divergence is calculated on the full sentence instead of 15% of randomly masked tokens
				model_kwargs (dict)					: used to create a baseline version of the passed model
				tokenizer_kwargs (dict)				: used to create a baseline version of the passed tokenizer
				size_average						: passed to KLDivLoss
				reduce 								: passed to KLDivLoss
				reduction							: passed to KLDivLoss
		'''
		super(KLBaselineLoss, self).__init__(size_average, reduce, reduction)
		self.model 				= model
		self.tokenizer 			= tokenizer
		self.device 			= self.model.device if torch.cuda.is_available() else 'cpu'
		self.masking 			= masking
		self.batch_size 		= batch_size
		
		log.info(f'Initializing Baseline Model for KLBaselineLoss:\t{self.model.config.architectures[0]} ({self.model.name_or_path})')
		self.baseline_model		= AutoModelForMaskedLM.from_pretrained(self.model.name_or_path, **model_kwargs).to(self.device)
		
		# we're not evaluating this
		# _ = is to prevent printing
		_ = self.baseline_model.eval()
		
		log.info(f'Initializing Baseline Tokenizer for KLBaselineLoss:\t{self.tokenizer.__class__.__name__}   ({self.tokenizer.name_or_path})')
		self.baseline_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer.name_or_path, **tokenizer_kwargs)
		
		# set up the dataset
		self.dataset 			= dataset
		
		# can't use more examples than we've got
		self.n_examples 		= self.dataset.num_rows if n_examples_per_step is None else min(n_examples_per_step, self.dataset.num_rows)
		
		# in our experiments, we add new tokens to the models that don't exist in the baseline tokenizers
		# so we need to exclude them when calculating the KL divergence
		self.to_include			= torch.LongTensor(
									list(
										set(self.tokenizer.get_vocab().values())
										.intersection(
											set(self.baseline_tokenizer.get_vocab().values())
										)
									)
								).to(self.device)
		
		self.scaleby 			= scaleby
		
	def forward(
		self, 
		progress_bar: bool = False,
		return_all: bool = False,
	) -> torch.Tensor:
		'''
		Computes KLBaselineLoss between the predictions of the baseline model
		and the predictions of the fine-tuned model on the basis of self.n_examples
		from self.dataset. Samples are randomized with each call.
		
			params:
				progress_bar (bool)		: whether to display a progress bar while iterating through
										  the chosen examples
				return_all (bool)		: whether to return a list containing every individual KL divergence
										  in a list in addition to the mean
			
			returns:
				kl_div (torch.Tensor)	: the mean KL divergence between the model and the baseline model
										  across n_examples of the dataset, multiplied by the scaling factor
				kl_divs (torch.Tensor)	: the individual KL divergence for each example
										  returned if return_all=True.
		'''
		# construct a comparison dataset for this call with n random examples
		comp_dataset 					= tuner_utils.sample_from_dataset(self.dataset, self.n_examples, log_message=progress_bar)
		dataloader 						= torch.utils.data.DataLoader(comp_dataset, batch_size=self.batch_size, collate_fn=pad_batch)
		total_kl_div					= torch.tensor((0.)).to(self.device)
		
		# haven't figure out a way to actually do batches > 1 yet
		# the issue is setting up the padding of the inputs correctly per batch. As it turns out
		# one at a is not actually that slow, so we'll just do it 1 at a time for now
		if progress_bar:
			dataloader 					= tqdm(dataloader)
		
		if progress_bar or return_all:
			kl_divs 					= []
		
		if return_all:
			all_mask_indices 			= []
		
		# when using bert-style masking, short examples
		# don't always have masked tokens, so we're not
		# adding any loss from them. for this reason,
		# we keep track of how many rows we do include
		# to get the correct mean value
		n_included_rows = 0
		n_not_included_rows = 0
		try:
			for i, batch in enumerate(dataloader):
				batch_inputs			= {k: v.to(self.device) for k, v in batch.items() if isinstance(v,torch.Tensor)}
				
				mask_indices			= []
				for i, _ in enumerate(batch_inputs['input_ids']):
					batch_inputs['input_ids'][i], mask_input_indices \
										= tuner_utils.mask_input(
											inputs=batch_inputs['input_ids'][i],
											tokenizer=self.baseline_tokenizer,
											masking_style=self.masking,
											device=self.device,
										)
					
					mask_indices.append(mask_input_indices)
				
				if return_all:
					# we don't want to return mask indices for things we're not actually calculating
					# the loss on due to no mask token
					all_mask_indices.extend([mi.tolist() for mi in mask_indices])
				
				outputs 				= self.model(**batch_inputs).logits.index_select(-1, self.to_include)
				
				# we're not training the baseline model, so no need to get gradients for it
				with torch.no_grad():
					baseline_outputs 	= self.baseline_model(**batch_inputs).logits
				
				# we calculate D_KL for each example individually because we'd like to record the D_KL 
				# per sentence for inspection later. doing it per batch would just give us a mean for a batch
				for output, baseline_output, ex_mask_indices in zip(outputs, baseline_outputs, mask_indices):		
					# we just calculate the loss on the selected tokens
					# if the sentence is very short, sometimes no tokens were
					# masked (when using masking = 'bert'). So don't include those
					if torch.any(ex_mask_indices):
						output				= torch.unsqueeze(F.log_softmax(torch.cat([output.index_select(0, mask_locations) for mask_locations in ex_mask_indices], dim=0), dim=-1), dim=0)
						baseline_output		= torch.unsqueeze(F.softmax(torch.cat([baseline_output.index_select(0, mask_locations) for mask_locations in ex_mask_indices], dim=0), dim=-1), dim=0)
						
						# we want this to be a mean instead of a sum, so divide by the length of the dataset
						kl_div 				= super(KLBaselineLoss, self).forward(output, baseline_output)
						total_kl_div 		+= kl_div
						n_included_rows 	+= 1
						
						if progress_bar or return_all:
							kl_divs.append(kl_div.cpu())
					else:
						n_not_included_rows += 1
						if progress_bar or return_all:
							kl_divs.append(np.nan)
						
				if progress_bar:
					dataloader.set_postfix(kl_div_mean=f'{np.nanmean(kl_divs):.2f}', kl_div_se=f'{tuner_utils.sem(kl_divs):.2f}')
					
		except KeyboardInterrupt:
			log.warning('KLBaselineLoss computation halted manually')
		
		if n_not_included_rows != 0:
			log.info(f'{n_not_included_rows}/{comp_dataset.num_rows} ({((n_not_included_rows/comp_dataset.num_rows)*100):.2f}%) examples were excluded from KL loss due to a lack of mask indices.')
		
		if return_all:
			return (total_kl_div/n_included_rows) * self.scaleby, torch.tensor(kl_divs).to(self.device), all_mask_indices
		else:
			return (total_kl_div/n_included_rows) * self.scaleby

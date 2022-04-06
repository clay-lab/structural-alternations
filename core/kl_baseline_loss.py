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

from . import tuner_utils

log = logging.getLogger(__name__)

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
		dataloader 						= torch.utils.data.DataLoader(comp_dataset, batch_size=1)
		mean_kl_div						= torch.tensor((0.)).to(self.device)
		
		# haven't figure out a way to actually do batches > 1 yet
		# the issue is setting up the padding of the inputs correctly per batch. As it turns out
		# one at a is not actually that slow, so we'll just do it 1 at a time for now
		if progress_bar:
			dataloader 					= tqdm(dataloader)
		
		if progress_bar or return_all:
			kl_divs 					= []
		
		if return_all:
			all_mask_indices 			= []
		
		try:
			for i, batch in enumerate(dataloader):
				batch_inputs			= {k: v.to(self.device) for k, v in batch.items() if isinstance(v,torch.Tensor)}
				
				mask_indices			= []
				for i, _ in enumerate(batch_inputs['input_ids']):
					batch_inputs['input_ids'][i], mask_input_indices \
										= tuner_utils.mask_input(
											inputs=batch_inputs['input_ids'][i],
											tokenizer=self.tokenizer,
											masking_style=self.masking,
											device=self.device,
										)
					
					mask_indices.append(mask_input_indices)
				
				if return_all:
					all_mask_indices.append(mask_indices)
				
				outputs 				= self.model(**batch_inputs).logits.index_select(-1, self.to_include)
				outputs 				= F.log_softmax(outputs, dim=-1)
				
				# we're not training the baseline model, so no need to get gradients for it
				with torch.no_grad():
					baseline_outputs 	= F.softmax(self.baseline_model(**batch_inputs).logits, dim=-1)
				
				# we just calculate the loss on the selected tokens
				outputs 				= torch.cat([torch.unsqueeze(outputs[i].index_select(0, mask_locations), dim=0) for i, mask_locations in enumerate(mask_indices)], dim=0)
				baseline_outputs		= torch.cat([torch.unsqueeze(baseline_outputs[i].index_select(0, mask_locations), dim=0) for i, mask_locations in enumerate(mask_indices)], dim=0)
				
				# we want this to be a mean instead of a sum, so divide by the length of the dataset
				kl_div 					= super(KLBaselineLoss, self).forward(outputs, baseline_outputs)
				mean_kl_div 			+= kl_div/comp_dataset.num_rows
						
				if progress_bar or return_all:
					kl_divs.append(kl_div)
					
					if progress_bar:
						dataloader.set_postfix(kl_div_mean=f'{np.mean(kl_divs):.2f}', kl_div_se=f'{tuner_utils.sem(kl_divs):.2f}')
					
		except KeyboardInterrupt:
			log.warning('KLBaselineLoss computation halted manually')
			mean_kl_div = (mean_kl_div * comp_dataset.num_rows)/i
		
		if return_all:
			return mean_kl_div * self.scaleby, torch.tensor(kl_divs).to(self.device), all_mask_indices
		else:
			return mean_kl_div * self.scaleby
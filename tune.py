# tune.py
# 
# Application entry point for fine-tuning a masked language model.
import os
import re
import hydra
import logging

from omegaconf import DictConfig, OmegaConf

from core.tuner import Tuner

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
	'get_dir_name',
	# we have to do this via an intermediate lambda because omegaconf only allows lambda resolvers
	lambda model, tuning, hyperparameters, kl_loss_params, layerwise_loss_params: formatted_dir_name(model, tuning, hyperparameters, kl_loss_params, layerwise_loss_params)
)

def formatted_dir_name(
	model: DictConfig, tuning: DictConfig, 
	hyperparameters: DictConfig, kl_loss_params: DictConfig,
	layerwise_loss_params: DictConfig,
) -> str:
	dir_name 	=	tuning.name
	
	model_name 	= 	'bbert' if model.friendly_name == 'bert' \
					else 'dbert' if model.friendly_name == 'distilbert' \
					else 'rbert' if model.friendly_name == 'roberta' \
					else 'mbert' + model.friendly_name.split('_')[-1] if 'multiberts' in model.friendly_name \
					else model.friendly_name
	
	dir_name 	= 	os.path.join(dir_name, model_name)
	
	dir_name 	+= 	'-'
	dir_name 	+= 	hyperparameters.masked_tuning_style[0] + 'mask' \
				  	if hyperparameters.masked_tuning_style in ['bert', 'roberta', 'always', 'none'] \
					else hyperparameters.masked_tuning_style
				
	dir_name 	+= 	'-'
	dir_name 	+= 	'wpunc' if not hyperparameters.strip_punct else 'npunc'
	
	dir_name 	+= 	'-'
	if hyperparameters.unfreezing == 'all_hidden_layers':
		dir_name += 'ahunf'
	else:
		dir_name += str(hyperparameters.unfreezing)[:2].zfill(2) + 'unf'
	
	if 'gradual' in hyperparameters.unfreezing:
		dir_name += re.sub(r'.*([0-9]+)', '\\1', hyperparameters.unfreezing).zfill(2)
	elif 'mixout' in hyperparameters.unfreezing:
		mixout_prob = re.search(r'([0-9]+)?\.[0-9]+$', hyperparameters.unfreezing)[0]
		if mixout_prob.startswith('.'):
			mixout_prob = '0' + mixout_prob
		dir_name += mixout_prob
	
	dir_name 	+= 	'-'
	dir_name 	+= 	f'lr{hyperparameters.lr}'

	if hyperparameters.use_kl_baseline_loss and not hyperparameters.unfreezing == 'none':
		dir_name += f'-{kl_loss_params.scaleby:.2f}kl'
		dir_name += kl_loss_params.masking[0] + 'mask' \
					if kl_loss_params.masking in ['always', 'none', 'bert'] \
					else kl_loss_params.masking
	
	if hyperparameters.use_layerwise_baseline_loss and not hyperparameters.unfreezing == 'none':
		dir_name += f'-{layerwise_loss_params.l2_scaleby:.2f}lw'
		dir_name += f'-{layerwise_loss_params.kl_scaleby:.2f}kl'
		dir_name += layerwise_loss_params.masking[0] + 'mask' \
					if layerwise_loss_params.masking in ['always', 'none', 'bert'] \
					else layerwise_loss_params.masking
	
	if 'which_args' in tuning and tuning.exp_type == 'newverb':
		dir_name = 	os.path.join(dir_name, model.friendly_name) if tuning.which_args == 'model' else \
				   	os.path.join(dir_name, tuning.which_args)
		dir_name += '_args'
	
	if hyperparameters.mask_args == True and tuning.exp_type == 'newverb':
		dir_name += '-margs'
	
	if 'mask_added_tokens' in hyperparameters and hyperparameters.mask_added_tokens != True:
		dir_name += '-nmato'
	
	return dir_name

@hydra.main(config_path='conf', config_name='tune')
def tune(cfg: DictConfig) -> None:
	print(OmegaConf.to_yaml(cfg, resolve=True))
	if cfg.tuning.data:
		tuner = Tuner(cfg, use_gpu=cfg.use_gpu)
		tuner.tune()
	else:
		log.warning(
			"You asked to tune, but didn't provide any tuning data. "
			"I'm quitting now. "
			"Not sure what you were hoping for..."
		)

if __name__ == "__main__":
	
	tune()
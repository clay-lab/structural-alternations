# tune.py
# 
# Application entry point for fine-tuning a masked language model.
import os
import re
import hydra

from pprint import PrettyPrinter
from omegaconf import DictConfig, OmegaConf, open_dict

from core.tuner import Tuner

OmegaConf.register_new_resolver(
	'get_dir_name',
	# we have to do this via an intermediate lambda because omegaconf only allows lambda resolvers
	lambda model, tuning, hyperparameters: formatted_dir_name(model, tuning, hyperparameters)
)

def formatted_dir_name(model: DictConfig, tuning: DictConfig, hyperparameters: DictConfig) -> str:
	dir_name = tuning.name
	
	dir_name = os.path.join(dir_name, model.friendly_name)
	
	dir_name += '-'
	dir_name += hyperparameters.masked_tuning_style[0] + 'mask' \
				if hyperparameters.masked_tuning_style in ['bert', 'roberta', 'always', 'none'] \
				else hyperparameters.masked_tuning_style
				
	dir_name += '-'
	dir_name += 'wpunc' if not hyperparameters.strip_punct else 'npunc'
	
	dir_name += '-'
	dir_name += str(hyperparameters.unfreezing)[:2].zfill(2) + 'unf'
	
	if 'gradual' in hyperparameters.unfreezing:
		dir_name += re.sub(r'.*([0-9]+)', '\\1', hyperparameters.unfreezing).zfill(2)
	
	dir_name += '-'
	dir_name += f'lr{hyperparameters.lr}'
	
	if 'which_args' in tuning:
		dir_name = os.path.join(dir_name, model.friendly_name) if tuning.which_args == 'model' else \
				   os.path.join(dir_name, tuning.which_args)
		dir_name += '_args'
	
	if hyperparameters.mask_args == True and tuning.exp_type == 'newverb':
		dir_name += '-margs'
	
	return dir_name

@hydra.main(config_path="conf", config_name="tune")
def tune(cfg: DictConfig) -> None:
	print(OmegaConf.to_yaml(cfg))
	tuner = Tuner(cfg)
	tuner.tune()

if __name__ == "__main__":
	
	tune()
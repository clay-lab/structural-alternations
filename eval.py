# eval.py
# 
# Application entry point for evaluating a masked language model.
import hydra
import logging

from omegaconf import DictConfig, OmegaConf

from core.tuner import Tuner

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
	'get_dir_name', 
	lambda data, comparison_masking: get_dir_name(data, comparison_masking)
)

def get_dir_name(data: DictConfig, comparison_masking: str) -> str:
	'''
	Gets a formatted directory name for saving results.
	
		params:
			data (dictconfig)		: a dict config containing the name of the eval dataset
			comparison_masking (str): a str specifying how kl divergence is to be calculated 
									  (i.e., how to mask tokens)
		
		returns:
			dir_name (str)			: a directory name to store the results in
	'''
	dir_name = data.split('.')[0]
	
	if comparison_masking:
		dir_name += '-kl' + comparison_masking[0] + 'mask'
	
	return dir_name

@hydra.main(config_path='conf', config_name='eval')
def eval(cfg: DictConfig) -> None:
	print(OmegaConf.to_yaml(cfg, resolve=True))
	tuner = Tuner(cfg.checkpoint_dir, use_gpu=cfg.use_gpu if 'use_gpu' in cfg else None)
	tuner.evaluate(cfg)

if __name__ == '__main__':
	
	eval()
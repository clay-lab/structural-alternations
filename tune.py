# tune.py
# 
# Application entry point for fine-tuning a masked language model.

import hydra
from omegaconf import DictConfig, OmegaConf

from core.tuner import Tuner

@hydra.main(config_path="conf", config_name="tune")
def tune(cfg: DictConfig) -> None:
	
	print(OmegaConf.to_yaml(cfg))
	
	if cfg.hyperparameters.masked_tuning_style == 'bert' and cfg.model.friendly_name == 'roberta':
		print('Warning: BERT-style masked tuning does not work with RoBERTa for now. masked_tuning_style will be set to "always".\n\n\t!!!!!!!!THIS IS NOT REFLECTED IN DIRECTORY NAMES OR CONFIG FILES!!!!!!!!\n')
		cfg.hyperparameters.masked_tuning_style = 'always'
	
	# Tune model
	tuner = Tuner(cfg)
	tuner.tune()

if __name__ == "__main__":
	tune()
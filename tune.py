# tune.py
# 
# Application entry point for fine-tuning a masked language model.
import hydra
from omegaconf import DictConfig, OmegaConf

from core.tuner import Tuner

OmegaConf.register_new_resolver(
	'maskname', 
	lambda model, masked, masked_tuning_style: \
		'bert_masking' if masked and masked_tuning_style == 'bert' and model == 'bert' else \
		'always_masked' if masked and (
							masked_tuning_style == 'always' or 
							(masked_tuning_style == 'bert' and model == 'roberta')
						) else \
		'no_masking' if not masked else \
		masked_tuning_style # failsafe
)

OmegaConf.register_new_resolver(
	'spname',
	lambda strip_punct: 'with_punctuation' if not strip_punct else 'no_punctuation'	# we add 'with' to the first one to facilitate multieval criteria
)

@hydra.main(config_path="conf", config_name="tune")
def tune(cfg: DictConfig) -> None:
	
	print(OmegaConf.to_yaml(cfg))
	
	if not cfg.hyperparameters.masked:
		cfg.hyperparameters.masked_tuning_style = None
	
	if cfg.hyperparameters.masked_tuning_style == 'bert' and cfg.model.friendly_name == 'roberta':
		print('Warning: BERT-style masked tuning does not work with RoBERTa for now. masked_tuning_style set to "always".')
		cfg.hyperparameters.masked_tuning_style = 'always'
	
	# Tune model
	tuner = Tuner(cfg)
	tuner.tune()

if __name__ == "__main__":
	tune()
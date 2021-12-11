# tune.py
# 
# Application entry point for fine-tuning a masked language model.
import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from core.tuner import Tuner

OmegaConf.register_new_resolver(
	'maskname', 
	lambda masked_tuning_style: \
		'bert_masking' if masked_tuning_style.lower() == 'bert' else
		'roberta_masking' if masked_tuning_style.lower() == 'roberta' else
		'always_masked' if masked_tuning_style.lower() == 'always' else
		'no_masking' if masked_tuning_style.lower() == 'none' else
		masked_tuning_style # failsafe
)

OmegaConf.register_new_resolver(
	'spname',
	lambda strip_punct: 'with_punctuation' if not strip_punct else 'no_punctuation'	# we add 'with' to the first one to facilitate multieval criteria
)

@hydra.main(config_path="conf", config_name="tune")
def tune(cfg: DictConfig) -> None:
	
	dev_sets = [cfg.dev] if isinstance(cfg.dev, str) else cfg.dev
	
	cfg.dev = {}
	with open_dict(cfg):
		for dev_set in dev_sets:
			dev = OmegaConf.load(os.path.join(hydra.utils.get_original_cwd(), 'conf', 'tuning', dev_set + '.yaml'))
			cfg.dev.update({dev.name:dev})
	
	# doing it this way lets us use any tuning data as dev data and vice versa,
	# though we can also define datasets that are only used for one or the other
	print(OmegaConf.to_yaml(cfg))
	tuner = Tuner(cfg)
	tuner.tune()

if __name__ == "__main__":
	tune()
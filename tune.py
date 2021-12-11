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
	
	# doing it this way lets us use any tuning data as dev data and vice versa,
	# though we can also define datasets that are only used for one or the other
	cfg.dev = OmegaConf.load(os.path.join(hydra.utils.get_original_cwd(), 'conf', 'tuning', cfg.dev + '.yaml'))
	print(OmegaConf.to_yaml(cfg))
	tuner = Tuner(cfg)
	tuner.tune()

if __name__ == "__main__":
	tune()
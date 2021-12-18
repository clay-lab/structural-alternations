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
	
	if cfg.dev == 'best_matches':
		criteria = cfg.tuning.name.split('_')
		candidates = os.listdir(os.path.join(hydra.utils.get_original_cwd(), 'conf', 'tuning'))
		candidates = [candidate.replace('.yaml', '').split('_') for candidate in candidates]
		
		# Find all the tuning sets that differ from the current one by one parameter, and grab those as our best matches
		candidates = [candidate for candidate in candidates if len(set(criteria) - set(candidate)) == 1 and candidate[0] == criteria[0]]
		
		# additionally filter out any manually excluded best_matches
		if 'dev_exclude' in cfg:
			cfg.dev_exclude = [cfg.dev_exclude] if isinstance(cfg.dev_exclude, str) else cfg.dev_exclude
			for exclusion in cfg.dev_exclude:
				candidates = [candidate for candidate in candidates if not exclusion in candidate]
		
		# join them together
		candidates = ['_'.join(candidate) for candidate in candidates]
		cfg.dev = candidates
	
	# change the arguments used if this is a new verb experiment and we are using the ones specific to the model
	# do this before printing
	if cfg.tuning.new_verb:
		if cfg.tuning.which_args == 'model':
			with open_dict(cfg):
				cfg.tuning.args = cfg.tuning[cfg.model.friendly_name + '_args']
		else;
			with open_dict(cfg):
				cfg.tuning.args = cfg.tuning[cfg.tuning.which_args]
	
	# print this before adding the dev sets, since that will print a lot of stuff we don't necessarily need to see
	print(OmegaConf.to_yaml(cfg))
	
	dev_sets = [cfg.dev] if isinstance(cfg.dev, str) else cfg.dev
	cfg.dev = {}
	with open_dict(cfg):
		for dev_set in dev_sets:
			# doing it this way lets us use any tuning data as dev data and vice versa,
			# though we can also define datasets that are only used for one or the other
			dev = OmegaConf.load(os.path.join(hydra.utils.get_original_cwd(), 'conf', 'tuning', dev_set + '.yaml'))
			cfg.dev.update({dev.name:dev})
	
	tuner = Tuner(cfg)
	tuner.tune()

if __name__ == "__main__":
	tune()
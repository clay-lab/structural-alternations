# eval.py
# 
# Application entry point for evaluating a masked language model.
import hydra
import logging

from omegaconf import DictConfig, OmegaConf

from core.tuner import Tuner

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver('name', lambda data: data.split('.')[0])

@hydra.main(config_path='conf', config_name='eval')
def eval(cfg: DictConfig) -> None:
	print(OmegaConf.to_yaml(cfg, resolve=True))
	tuner = Tuner(cfg.checkpoint_dir)
	tuner.evaluate(cfg)

if __name__ == '__main__':
	
	eval()
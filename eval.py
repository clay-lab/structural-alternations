# eval.py
# 
# Application entry point for evaluating a masked language model.

import os
import hydra

from omegaconf import DictConfig, OmegaConf

from core.tuner import Tuner

@hydra.main(config_path="conf", config_name="eval")
def eval(cfg: DictConfig) -> None:
	print(OmegaConf.to_yaml(cfg))
	tuner = Tuner(cfg.checkpoint_dir)
	tuner.evaluate(cfg)

if __name__ == "__main__":
	eval()
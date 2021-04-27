# tune.py
# 
# Application entry point for fine-tuning a masked language model.

import hydra
from omegaconf import DictConfig, OmegaConf

from core.tuner import Tuner

@hydra.main(config_path="conf", config_name="tune")
def tune(cfg: DictConfig) -> None:
  
  print(OmegaConf.to_yaml(cfg))

  # Tune model
  tuner = Tuner(cfg)
  tuner.tune()

if __name__ == "__main__":
  tune()
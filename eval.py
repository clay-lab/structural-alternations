# tune.py
# 
# Application entry point for fine-tuning a masked language model.

import os
import hydra
from omegaconf import DictConfig, OmegaConf

from core.tuner import Tuner

@hydra.main(config_path="conf", config_name="eval")
def tune(cfg: DictConfig) -> None:

  print(OmegaConf.to_yaml(cfg))

  # Load checkpoint configuration
  chkpt_dir = hydra.utils.to_absolute_path(cfg.checkpoint_dir)
  chkpt_cfg_path = os.path.join(chkpt_dir, '.hydra', 'config.yaml')
  chkpt_cfg = OmegaConf.load(chkpt_cfg_path)

  # Tune model
  tuner = Tuner(chkpt_cfg)
  if cfg.data.entail:
    tuner.eval_entailments(
      eval_cfg = cfg,
      checkpoint_dir = chkpt_dir
    )
  else:
    tuner.eval(
      eval_cfg = cfg, 
      checkpoint_dir=chkpt_dir
    )

if __name__ == "__main__":
  tune()
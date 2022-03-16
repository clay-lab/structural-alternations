# eval.py
# 
# Application entry point for evaluating a masked language model.
from core.tuner_imports import *
from core.tuner import Tuner

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="eval")
def eval(cfg: DictConfig) -> None:
	print(OmegaConf.to_yaml(cfg))
	tuner = Tuner(cfg.checkpoint_dir)
	tuner.evaluate(cfg)

if __name__ == "__main__":
	
	eval()
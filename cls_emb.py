# tune.py
# 
# Application entry point for training an SVM on word embeddings from a pre-trained model.
from core.tuner_imports import *
from core.embedding_classifier import EmbeddingClassifier

OmegaConf.register_new_resolver(
	'get_dir_name',
	# we have to do this via an intermediate lambda because omegaconf only allows lambda resolvers
	lambda model, tuning: formatted_dir_name(model, tuning)
)

def formatted_dir_name(model: DictConfig, tuning: DictConfig) -> str:
	dir_name = tuning.name
	dir_name = os.path.join(dir_name, 'cls_emb', model.friendly_name)
	dir_name += '-' + (f'{model.friendly_name}' if tuning.which_args == 'model' else tuning.which_args) + '_args'
	
	return dir_name

@hydra.main(config_path='conf', config_name='cls_emb')
def classify_embeddings(cfg: DictConfig) -> None:
	args_to_keep = cfg.tuning.which_args if not cfg.tuning.which_args == 'model' else f'{cfg.model.friendly_name}'
	
	# remove stuff we don't need to print
	with open_dict(cfg):
		for k in cfg['tuning'].copy():
			if not k in ['name', 'which_args', args_to_keep]:
				del cfg['tuning'][k]
	
	print(OmegaConf.to_yaml(cfg))
	embedding_classifier = EmbeddingClassifier(cfg)
	embedding_classifier.fit()

if __name__ == "__main__":
	
	classify_embeddings()
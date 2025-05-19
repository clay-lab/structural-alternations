# tune.py
# 
# Application entry point for fine-tuning a masked language model.
import os
import re
import gzip
import hydra
import optuna
import logging

import numpy as np
import pickle as pkl

from omegaconf import DictConfig, OmegaConf

from core.tuner import Tuner

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
	'get_dir_name',
	# we have to do this via an intermediate lambda because omegaconf only allows lambda resolvers
	lambda model, tuning, hyperparameters, kl_loss_params, layerwise_loss_params, do_optimize, optimize_cfg: formatted_dir_name(
		model, tuning, hyperparameters, kl_loss_params, layerwise_loss_params, do_optimize, optimize_cfg
	)
)

OmegaConf.register_new_resolver(
	'get_study_name',
	lambda params, target_label: '-'.join(
		[f'objective_{target_label}'] + 
		['_'.join([k] + ['_'.join([str(v2) for v2 in params[k]['values']])]) for k, v in params.items()]
	)
)

def formatted_dir_name(
	model: DictConfig, tuning: DictConfig, 
	hyperparameters: DictConfig, kl_loss_params: DictConfig,
	layerwise_loss_params: DictConfig, do_optimize: bool,
	optimize_cfg: DictConfig,
) -> str:
	# these things should not be optimized.
	dir_name = tuning.name
	
	model_name = 'bbert' if model.friendly_name == 'bert' \
		else 'dbert' if model.friendly_name == 'distilbert' \
		else 'rbert' if model.friendly_name == 'roberta' \
		else 'mbert' + model.friendly_name.split('_')[-1] if 'multiberts' in model.friendly_name \
		else 'mobert' + model.friendly_name.split(' ')[-1][0] if 'ModernBERT' in model.friendly_name \
		else model.friendly_name
	
	dir_name = os.path.join(dir_name, model_name)
	
	dir_name +=	'-'
	
	# the following might all be optimized, so we need to get the right dir name for that case.
	if do_optimize:
		return get_optimization_dir_name(
			prefix=dir_name,
			tuning=tuning,
			hyperparameters=hyperparameters,
			kl_loss_params=kl_loss_params,
			layerwise_loss_params=layerwise_loss_params,
			optimize_cfg=optimize_cfg
		)
	
	dir_name += hyperparameters.masked_tuning_style[0] + 'mask' \
		if hyperparameters.masked_tuning_style in ['bert', 'roberta', 'modernbert', 'always', 'none'] \
		else hyperparameters.masked_tuning_style
		
	dir_name +=	'-'
	
	dir_name += 'wpunc' if not hyperparameters.strip_punct else 'npunc'
	
	dir_name += '-'
	
	if hyperparameters.unfreezing == 'all_hidden_layers':
		dir_name += 'ahunf'
	else:
		dir_name += str(hyperparameters.unfreezing)[:2].zfill(2) + 'unf'
	
	if 'gradual' in hyperparameters.unfreezing:
		dir_name += re.sub(r'.*([0-9]+)', '\\1', hyperparameters.unfreezing).zfill(2)
	elif 'mixout' in hyperparameters.unfreezing:
		mixout_prob = re.search(r'([0-9]+)?\.[0-9]+$', hyperparameters.unfreezing)[0]
		if mixout_prob.startswith('.'):
			mixout_prob = '0' + mixout_prob
		dir_name += mixout_prob
	
	dir_name += '-'
	
	dir_name += f'lr{hyperparameters.lr}'
	
	if hyperparameters.use_kl_baseline_loss and not hyperparameters.unfreezing == 'none':
		dir_name += f'-{kl_loss_params.scaleby:.2f}kl'
		dir_name += kl_loss_params.masking[0] + 'mask' \
			if kl_loss_params.masking in ['always', 'none', 'modernbert', 'roberta', 'bert'] \
			else kl_loss_params.masking
	
	if hyperparameters.use_layerwise_baseline_loss and not hyperparameters.unfreezing == 'none':
		dir_name += f'-{layerwise_loss_params.l2_scaleby:.2f}lw'
		dir_name += f'-{layerwise_loss_params.kl_scaleby:.2f}kl'
		dir_name += layerwise_loss_params.masking[0] + 'mask' \
			if layerwise_loss_params.masking in ['always', 'none', 'bert', 'modernbert', 'roberta'] \
			else layerwise_loss_params.masking
	
	# this should not be optimized, as it involves changing the dataset
	if 'which_args' in tuning and tuning.exp_type == 'newverb':
		dir_name = os.path.join(dir_name, model.friendly_name) if tuning.which_args == 'model' else \
			os.path.join(dir_name, tuning.which_args)
		dir_name += '_args'
	
	if hyperparameters.mask_args == True and tuning.exp_type == 'newverb':
		dir_name += '-margs'
	
	if 'mask_added_tokens' in hyperparameters and hyperparameters.mask_added_tokens != True:
		dir_name += '-nmato'
	
	return dir_name

def get_optimization_dir_name(
	prefix: str, tuning: DictConfig,
	hyperparameters: DictConfig, kl_loss_params: DictConfig,
	layerwise_loss_params: DictConfig, optimize_cfg: DictConfig,
) -> str:
	'''
	For optimization, we need to join together the values being optimized in a sensible way
	with the values that aren't being optimized. So let's do that here.
	'''
	
	# gather all the params in a dictionary that we want to use. prefer the params that exist
	# in the optimization config; otherwise, use the params from hyperparameters/kl_loss_params/
	# layerwise_loss_params
	params = {}
	for k in hyperparameters:
		if k in optimize_cfg.params:
			params[k] = optimize_cfg.params[k]['values']
			continue
		
		params[k] = [hyperparameters[k]]
	
	params['kl_loss_params'] = {}
	for k in kl_loss_params:
		if f'kl_loss_params.{k}' in optimize_cfg.params:
			params['kl_loss_params'][k] = optimize_cfg.params[k]['values']
			continue
		
		params['kl_loss_params'][k] = [kl_loss_params[k]]
	
	params['layerwise_loss_params'] = {}
	for k in layerwise_loss_params:
		if f'layerwise_loss_params.{k}' in optimize_cfg.params:
			params['layerwise_loss_params'][k] = optimize_cfg.params[k]['values']
			continue
		
		params['layerwise_loss_params'][k] = [layerwise_loss_params[k]]
	
	params = OmegaConf.create(params)
	
	dir_name = ''
	
	dir_name += '_'.join([
		v[0] + 'mask' 
		if v in ['bert', 'roberta', 'modernbert', 'always', 'none'] 
		else v
		for v in params.masked_tuning_style
	])
	dir_name +=	'-'
	
	dir_name += '_'.join([
		'wpunc' if not v else 'npunc'
		for v in params.strip_punct
	])
	dir_name += '-'
	
	unf = []
	for v in params.unfreezing:
		if v == 'all_hidden_layers':
			u = 'ahunf'
		else:
			u = f'{v[:2].zfill(2)}unf'
		
		if 'gradual' in v:
			u += re.sub(r'.*([0-9]+)', '\\1', v).zfill(2)
		elif 'mixout' in v:
			mixout_prob = re.search(r'([0-9]+)?\.[0-9]+$', v)[0]
			if mixout_prob.startswith('.'):
				mixout_prob = '0' + mixout_prob
			u += mixout_prob
		
		unf += [u]
	
	dir_name += '_'.join(unf)
	dir_name += '-'
	
	dir_name += '_'.join([f'lr{v}' for v in params.lr])
	
	if any(params.use_kl_baseline_loss) and not all(x == 'none' for x in params.unfreezing):
		dir_name += '_'.join([f'-{v}kl' for v in params.kl_loss_params.scaleby])
		
		dir_name += '_'.join([
			f'{v[0]}mask' if v in ['always', 'none', 'modernbert', 'roberta', 'bert']
			else v
			for v in params.kl_loss_params.masking
		])
	
	if any(params.use_layerwise_baseline_loss) and not all(x == 'none' for x in params.unfreezing):
		dir_name += '_'.join([
			f'-{v:.2f}lw' for v in params.layerwise_loss_params.l2_scaleby
		])
		
		dir_name += '_'.join([f'-{v:.2f}kl' for v in layerwise_loss_params.kl_scaleby])
		dir_name += '_'.join([
			f'{v[0]}mask' if v in ['always', 'none', 'bert', 'modernbert', 'roberta']
			else v
			for v in layerwise_loss_params.masking
		])
	
	# this should not be optimized, as it involves changing the dataset
	if 'which_args' in tuning and tuning.exp_type == 'newverb':
		dir_name = os.path.join(dir_name, model.friendly_name) if tuning.which_args == 'model' else \
			os.path.join(dir_name, tuning.which_args)
		dir_name += '_args'
	
	if tuning.exp_type == 'newverb':
		if any(params.mask_args):
			dir_name += '-'
			dir_name += '_'.join(['margs' if v else 'nmargs' for v in params.mask_args])
	
	if 'mask_added_tokens' in params and any(v != True for v in params.mask_added_tokens):
		dir_name += '-'
		dir_name += '_'.join(['ymato' if v else 'nmato' for v in params.mask_added_tokens])
	
	dir_name = 'opt-' + prefix + dir_name
	
	return dir_name

def save_optimization_results(study: optuna.study.study.Study, tuner: Tuner) -> None:
	'''
	Save results from optimization study.
	params: study: optuna.study.study.Study: the study whose results are to be saved
		    tuner: tuner.Tuner: the tuner that contains the params for the study.
	'''
	# First, just save the whole study object in case.
	with gzip.open(os.path.join(tuner.checkpoint_dir, 'optimization_study.pkl.gz'), 'wb') as out_file:
		pkl.dump(study, out_file)
	
	# Now, save a more usable dataframe with results.
	df = study.trials_dataframe()
	# this only gives us the values for the final run. That's fine since the values
	# will mostly be the same. If they've changed, we drop them.
	df = tuner._add_hyperparameters_to_summary_df(df)
	df = df[[c for c in df.columns if not f'params_{c}' in df.columns]]
	
	# The only other thing we need to do is gather up the random seeds, since those
	# aren't recorded. We can do that from the log file.
	with open(os.path.join(tuner.checkpoint_dir, 'tune.log'), 'rt') as in_file:
		l = in_file.read()
	
	seeds = re.findall(r'Seed set to ([0-9]+)\n', l)
	df.random_seed = [seeds[n] if n <= (len(seeds) - 1) else np.nan for n in df.number]
	
	# rename the 'value' column to the metric being optimized for clarity
	if 'value' in df.columns:
		# need to use the label to avoid a list as a column name in case
		# we're taking an average of multiple metrics as our optimization
		# goal
		df = df.rename(columns={'value': tuner.cfg.optimize_cfg.target_label})
	
	# remove any blank rows (those for which state is still RUNNING seem
	# to correspond to what the study object reports before it's been concluded)
	df = df[df.state != 'RUNNING']
	
	df.to_csv('optimization_trials.csv.gz', index=False)

def resolve_study_kwargs(study_kwargs: DictConfig) -> dict:
	'''
	Returns a dictionary containing the study kwargs for optuna.
	This loads the pruner and sampler classes from a string identifier
	if one is provided, as well as functions they may need.
	'''
	def import_class_from_string(path: str) -> 'any':
		# from Pat @ https://stackoverflow.com/questions/452969/does-python-have-an-equivalent-to-java-class-forname
		from importlib import import_module
		module_path, _, class_name = path.rpartition('.')
		mod = import_module(module_path)
		klass = getattr(mod, class_name)
		return klass
	
	st_kwargs = {}
	for k, v in study_kwargs.items():
		# these should just be strings, ints, or floats, so we're good
		if not any(k.startswith(x) for x in ['sampler', 'pruner']) or v is None:
			st_kwargs[k] = v
			continue
		
		if k in ['sampler', 'pruner'] and isinstance(v, str):
			sp_kwargs = {}
			for k2, v2 in study_kwargs.get(f'{k}_kwargs', {}).items():
				# these should just be strings, ints, or floats,
				# so we're good
				if not any(k2.startswith(x) for x in ['wrapped_pruner', 'gamma', 'weights']):
					sp_kwargs[k2] = v2
					continue
				
				# a wrapped pruner is an object, so we need to instantiate it
				# with the appropriate kwargs
				if k2 in ['wrapped_pruner', 'independent_sampler']:
					sp_kwargs[k2] = import_class_from_string(v2)(
						**study_kwargs
							.get(f'{k}_kwargs', {})
							.get(f'{k2}_kwargs', {})
					)
				
				# a gamma or weights kwarg is a function passed to a sampler,
				# so we need to convert the string identifier to the actual
				# Callable with the right parameters
				if k2 in ['gamma', 'weights', 'constraints_func']:
					from functools import partial
					callable_function = partial(
						globals()[v2],
						*study_kwargs
							.get(f'{k}_kwargs', {})
							.get(f'{k2}_args', []),
						**study_kwargs
							.get(f'{k}_kwargs', {})
							.get(f'{k2}_kwargs', {})
					)
					sp_kwargs[k2] = callable_function
			
			st_kwargs[k] = import_class_from_string(v)(**sp_kwargs)
	
	return st_kwargs

@hydra.main(config_path='conf', config_name='tune', version_base=None)
def tune(cfg: DictConfig) -> None:
	print(OmegaConf.to_yaml(cfg, resolve=True))
	if cfg.tuning.data:
		tuner = Tuner(cfg, use_gpu=cfg.use_gpu)
	else:
		log.warning(
			"You asked to tune, but didn't provide any tuning data. "
			"I'm quitting now. "
			"Not sure what you were hoping for..."
		)
	
	if tuner.cfg.do_optimize:
		log.info(
			f'Optimizing hyperparameters with settings: \n'
			f'{OmegaConf.to_yaml(tuner.cfg.optimize_cfg, resolve=True)}.'
		)
		study_kwargs = resolve_study_kwargs(tuner.cfg.optimize_cfg.study_kwargs)
		study = optuna.create_study(**study_kwargs)
		study.optimize(tuner.tune, **tuner.cfg.optimize_cfg.optimize_kwargs)
		log.info(f'Best parameters: {study.best_params}')
		log.info(f'Best {tuner.cfg.optimize_cfg.target_label}: {study.best_value}')
		save_optimization_results(study=study, tuner=tuner)
	else:
		tuner.tune()

if __name__ == "__main__":
	tune()
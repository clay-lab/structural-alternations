import re
import os
import hydra
import itertools

import sball

from glob import glob
from typing import List
from omegaconf import DictConfig

@hydra.main(config_path='.', config_name='ssplit')
def split_scripts(cfg: DictConfig) -> None:
	'''
	Automatically factor out sweeps into separate scripts
	
		params:	
				cfg (dictconfig): A dictconfig specifying which default options to use
	'''
	path, sweeps = cfg.sweep.split()[0], cfg.sweep.split()[1:]
	all_sweeps = []
	for sweep in sweeps:
		key = sweep.split('=', 1)[0]
		values = sweep.split('=', 1)[1]
		
		if values.startswith('glob(') and values.endswith(')'):
			values = parse_values_from_glob(values, os.path.join(cfg.hydra_glob_dirname, key))
		else:
			# this lets us not split arguments which have commas inside them
			values = list(set(values.split(','))) if not (re.match(r'^\[.*\]$', values) or re.match(r'^\'.*\'$', values) or re.match(r'^\".*\"$', values)) else [values]
		
		all_sweeps.append([key + '=' + value for value in values])
	
	all_sweeps = list(itertools.product(*all_sweeps))
	all_sweeps = [' \\\n\t'.join(sweep) for sweep in all_sweeps]
	
	header = '#!/bin/bash\n\n'
	
	for slurm_option in cfg.s:
		# we can't have dashes in hydra config options
		header += f'#SBATCH --{"job-name" if slurm_option == "jobname" else slurm_option}={cfg.s[slurm_option]}\n'
	
	header += '\n'
	
	for pre in cfg.header:
		header += pre + '\n'
	
	header += '\n'
	
	filenames = []
	for i, sweep in enumerate(all_sweeps):
		filename = sweep.split(' \\\n\t')
		filename = '-'.join([f.split('=')[0][0] + '=' + os.path.split(f.split('=')[-1])[-1][0] for f in filename])
		filename = filename.replace(os.path.sep, '-')
		
		n = 0
		if os.path.isfile(filename + '.sh'):
			while os.path.isfile(filename + str(n) + '.sh'):
				n += 1
			
			filename += str(n)
		
		filename += '.sh'
		
		file = header + f'echo Running script: ' + os.getcwd().replace(hydra.utils.get_original_cwd() + os.path.sep, '').replace('\\', '/') + f'/{filename}\n\n'
		file += cfg.command + ' ' + path + ' \\\n\t' + sweep
		
		filenames.append(filename)
		with open(filename, 'w') as out_file:
			out_file.write(file)
	
	if cfg.runafter:
		filenames = [os.path.join(os.getcwd().replace(hydra.utils.get_original_cwd() + os.path.sep, ''), f) for f in filenames]
		expr = ' '.join(filenames)
		os.chdir(hydra.utils.get_original_cwd())
		sball.sbatch_all(expr)

def parse_values_from_glob(values: str, hydra_glob_dirname: str) -> List[str]:
	'''
	Get values from a glob argument passed to hydra, in the way that hydra would resolve it.
	
		params:
			values (str)		: A str containing an expression starting with glob( and ending with ).
			hydra_glob_dirname (str): The directory path relative to the original cwd where the configs to be globbed reside.
	'''
	if not os.path.isdir(os.path.join(hydra.utils.get_original_cwd(), hydra_glob_dirname)):
		raise ValueError(f'Invalid directory {os.path.join(hydra.utils.get_original_cwd(), hydra_glob_dirname)}! Unable to use globs.')
	
	if ',exclude=' in values:
		excludes 	= (values.split(',exclude=')[-1]
						.replace(')', '')
						.replace('[', '[r"')
						.replace(']', '"]')
						.replace(',', '",r"')
					)
		excludes 	= '[r"' + excludes + '"]' if not excludes.startswith('[r"') and not excludes.endswith('"]') else excludes
		excludes 	= eval(excludes)
		excludes 	= [os.path.join(hydra.utils.get_original_cwd(), hydra_glob_dirname, g) for g in excludes]
		excluded 	= [f for g in excludes for f in glob(g, recursive=True)]
	else:
		excluded 	= []
		
	globs	 		= (values.split(',exclude=')[0]
						.replace('glob(', '')
						.replace(')', '')
						.replace('[', '[r"')
						.replace(']', '"]')
						.replace(',', '",r"')
					)
	globs 			= '[r"' + globs + '"]' if not globs.startswith('[r"') and not globs.endswith('"]') else globs
	globs 			= eval(globs)
	globs 			= [os.path.join(hydra.utils.get_original_cwd(), hydra_glob_dirname, g) for g in globs]
	included 		= [f for g in globs for f in glob(g, recursive=True)]
	
	return [os.path.split(f)[-1].split('.')[0] for f in included if not f in excluded]

if __name__ == '__main__':
	
	split_scripts()	

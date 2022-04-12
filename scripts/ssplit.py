import os
import hydra
import itertools
import sball

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
		key, values = sweep.split('=')[0], list(set(sweep.split('=')[1].split(',')))
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
		while os.path.isfile(filename + '.sh'):
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

if __name__ == '__main__':
	
	split_scripts()	

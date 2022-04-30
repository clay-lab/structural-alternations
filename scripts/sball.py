import os
import re
import sys
import time
from glob import glob
import subprocess

def sbatch_all(s):
	'''
	Submit all scripts matching glob expressions as sbatch jobs
	s consists of command line args except for 'sball'
	the final argument should be the glob pattern,
	any additional preceding arguments are passed to sbatch.
	'''
	scripts = s[-1].split()
	args 	= [arg for arg in s[:-1] if not arg.startswith('name=')]
	name 	= [arg.split('=')[1] for arg in s[:-1] if arg.startswith('name=')]
	name 	= name[0] if name else []
	
	globbed = []
	for script in scripts:
		globbed.append(glob(script, recursive=True))
	
	globbed = [script for l in globbed for script in l if script.endswith('.sh')]
	
	sbatch_options = {}
	submit_individually = False if name and not len(globbed) == 1 else True
	
	if not submit_individually:
		for script in globbed:
			with open(script, 'rt') as in_file:
				script = in_file.readlines()
			
			options 		= [line for line in script if line.startswith('#SBATCH') and not 'job-name' in line and not 'output' in line]
			option_keys 	= [re.sub('.*--(.*?)=.*\n', '\\1', option) for option in options]
			option_values	= [re.sub('.*--.*=(.*)\n', '\\1', option) for option in options]
			
			for key, value in zip(option_keys, option_values):
				sbatch_options[key] = sbatch_options.get(key, [])
				sbatch_options[key].append(value)
				sbatch_options[key] = list(set(sbatch_options[key]))
				if len(sbatch_options[key]) > 1:
					# if we have different options, we can't use a job array and submit them individually
					submit_individually = True
					break
			
			if submit_individually:
				break
		
	if not submit_individually:
		try:
			# create a joblist txt file
			joblist = []
			for script in globbed:
				with open(script, 'rt') as in_file: 
					script = in_file.readlines()
				
				script = [line.replace('\n', '') for line in script if not line.startswith('#') and not line == '\n']
				script = '; '.join(script)
				script = script.replace('\t', '').replace('\\; ', '') + '\n'
				joblist.append(script)
			
			joblist = ''.join(joblist)
			
			with open(os.path.join('scripts', name + '.txt'), 'wt') as out_file:
				out_file.write(joblist)
				
			sbatch_options = ['--' + k + ' ' + v[0] for k, v in sbatch_options.items()]
			sbatch_options = [i for sublist in [option.split(' ') for option in sbatch_options] for i in sublist]
			
			x = subprocess.Popen([
				'dsq', 
				'--job-file', os.path.join('scripts', name + '.txt'), 
				'--status-dir', 'joblogs' + os.path.sep, 
				'--job-name', name, 
				#'--submit', 
				*sbatch_options, 
				*args
			])
			time.sleep(1)
			x.kill()
			
		except Exception:
			print('Unable to submit jobs using dSQ. Submitting individually.')
			for script in globbed:
				x = subprocess.Popen(['sbatch', *args, script])
				time.sleep(1)
				x.kill()	
	else:
		print('Submitting job(s) individually. This may be due to differing options, or because there is only one matching script.')
		for script in globbed:
			x = subprocess.Popen(['sbatch', *args, script])
			time.sleep(1)
			x.kill()

if __name__ == '__main__':
	args = [arg for arg in sys.argv[1:] if not arg == 'sball.py']
	sbatch_all(args)

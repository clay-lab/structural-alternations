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
	args 	= s[:-1]
	
	globbed = []
	for script in scripts:
		globbed.append(glob(script))
	
	globbed = [script for l in globbed for script in l if script.endswith('.sh')]
	
	for script in globbed:
		x = subprocess.Popen(['sbatch', *args, script])
		time.sleep(0.5)
		x.kill()

if __name__ == '__main__':
	args = [arg for arg in sys.argv[1:] if not arg == 'sball.py']
	sbatch_all(args)

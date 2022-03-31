import sys
from glob import glob
import subprocess

def sbatch_all(s):
	'''
	Submit all scripts matching glob expressions as sbatch jobs
	'''
	scripts = s.split()
	
	# this isn't really necessary on bash, but just in case
	globbed = []
	for script in scripts:
		globbed.append(glob(script))
	
	globbed = [script for l in globbed for script in l]
	
	for script in globbed:
		subprocess.run(['sbatch', script])

if __name__ == '__main__':
	
	sbatch_all(sys.argv[-1])

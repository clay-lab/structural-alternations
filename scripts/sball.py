import sys
import time
from glob import glob
import subprocess

def sbatch_all(s):
	'''
	Submit all scripts matching glob expressions as sbatch jobs
	'''
	scripts = s.split()
	
	globbed = []
	for script in scripts:
		globbed.append(glob(script))
	
	globbed = [script for l in globbed for script in l if script.endswith('.sh')]
	
	for script in globbed:
		x = subprocess.Popen(['sbatch', script])
		time.sleep(0.5)
		x.kill()

if __name__ == '__main__':
	
	sbatch_all(sys.argv[-1])

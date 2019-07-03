import numpy as np
import os
import sys
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt

from main import plot_rdsd

if __name__ == '__main__':
	os.chdir(sys.argv[1])
	L = int(sys.argv[2])
	realfids = []
	succfids = []
	for r in range(72):
		f = np.load("R%d/Fidelity.npz"%r)
		realfids.append(f['real'])
		succfids.append(f['succ'])
		V = f['V']
	realfids = np.array(realfids)
	succfids = np.array(succfids)
	plot_rdsd(V,real=realfids,succ=succfids)
	plt.savefig('rdsd.pdf')
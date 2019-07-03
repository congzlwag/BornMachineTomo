# from mpi4py import MPI
import sys
sys.path.append('../')
from CS6 import ProjMeasureSet, MPS
import numpy as np
import os

measout_dir = './'

if __name__ == '__main__':
	typ = "Neel"
	space_size = int(sys.argv[1])
	if space_size == 8:
		ns = 0.01
	elif space_size == 14:
		ns = 0.11

	mxBatch = 200
	batch_size = 40

	sm = MPS(space_size, typ, noise=ns)
	sm.leftCano()
	for rk in range(8):
		print(rk)
		np.random.seed(rk)
		ds = ProjMeasureSet(space_size, mxBatch*batch_size, mps=sm, noise=ns)

		measout = './MeasOutcomes/%d/%g/'%(space_size, ns)
		try:
			os.makedirs(measout)
		except FileExistsError:
			pass
		ds.save(measout+"R%dSet"%rk)

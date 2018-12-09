from mpi4py import MPI
import sys
sys.path.append('../../')
from CS6 import ProjMeasureSet, MPS
import numpy as np
import os

measout_dir = './'

if __name__ == '__main__':
	typ = 'random'
	space_size = int(sys.argv[2])
	
	comm = MPI.COMM_WORLD
	rk = comm.Get_rank()
	dmax = int(sys.argv[1])
	if dmax == 1:
		dmin = 1
	else:
		dmin = 2
	np.random.seed(rk)

	mxBatch = 2000
	batch_size = 40
	sm = MPS(space_size, typ, randomInitMaxD=dmax+1, Dmin=dmin)
	sm.leftCano()
	ds = ProjMeasureSet(space_size, mxBatch*batch_size, mps=sm)

	measout = measout_dir+'/%d/%d/%d/'%(dmax, space_size, rk)
	try:
		os.makedirs(measout)
	except FileExistsError:
		pass
	ds.save(measout+"/R%dSet"%(rk))

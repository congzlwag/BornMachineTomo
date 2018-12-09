from mpi4py import MPI
import sys
sys.path.append('../')
from CS6 import ProjMeasureSet, MPS
import numpy as np
import os

measout_dir = './'

if __name__ == '__main__':
	typ = sys.argv[1]
	space_size = int(sys.argv[2])
	
	comm = MPI.COMM_WORLD
	rk = comm.Get_rank()
	if typ=='random':
		dmax = int(sys.argv[3])
		sd = int(sys.argv[4])
		np.random.seed(sd)
		sm = MPS(space_size, typ, randomInitMaxD=dmax+1)
	else:
		sm = MPS(space_size, typ)

	mxBatch = 1500
	batch_size = 40
	sm.leftCano()
	np.random.seed(rk)
	ds = ProjMeasureSet(space_size, mxBatch*batch_size, mps=sm)

	if typ=='random':
		measout = measout_dir+'/%s/%d/%d/%d/'%(typ, dmax, space_size, sd)
	else:
		measout = measout_dir+'/%s/%d/'%(typ, space_size)
	try:
		os.makedirs(measout)
	except FileExistsError:
		pass
	ds.save(measout+"/R%dSet"%(rk))

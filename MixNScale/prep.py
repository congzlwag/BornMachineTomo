from mpi4py import MPI
import sys
sys.path.append('../')
from CS6 import ProjMeasureSet, MPS
import numpy as np
import os

measout_dir = './'

if __name__ == '__main__':
	typ = 'random'
	space_size = int(sys.argv[1])
	nss = np.load('elist.npy')
	comm = MPI.COMM_WORLD
	rk = comm.Get_rank()
	np.random.seed(rk)

	mxBatch = 2000
	batch_size = 40

	sm = MPS(space_size, typ)
	sm.leftCano()
	for ns in nss:
		measout = './Random/%g/%d/%d/'%(ns, space_size, rk)
		if not os.path.exists(measout):
			os.makedirs(measout)
		
		ds = ProjMeasureSet(space_size, mxBatch*batch_size, mps=sm, noise=ns)
		ds.save(measout+"R%dSet"%rk)

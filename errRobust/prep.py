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
	nss = load('elist.npy')
	comm = MPI.COMM_WORLD
	rk = comm.Get_rank()
	np.random.seed(rk)

	mxBatch = 2000
	batch_size = 40

	sm = MPS(space_size, typ)
	sm.leftCano()
	for ns in nss:
		ds = ProjMeasureSet(space_size, mxBatch*batch_size, mps=sm, noise=ns)

		measout = './%s/%d/%g/'%(typ, space_size, ns)
		try:
			os.makedirs(measout)
		except FileExistsError:
			pass
		ds.save(measout+"R%dSet"%rk)

from mpi4py import MPI
import sys
sys.path.append('../')
from CS6 import ProjMeasureSet, MPS
import numpy as np
import os

measout_dir = './vir_measout/'

if __name__ == '__main__':
	typ = sys.argv[1]
	R = 9
	L = 249
	sd = 9
	if typ=='random':
		space_size = 10
		MPS_dir = '../trial9-randomTarget/%s/%d/%d/R%d/L%d'%(typ,space_size,sd,R,L)
	else:
		space_size = 30
		MPS_dir = '../trial8-Efficiency/%s/%d/R%d/L%d'%(typ,space_size,R,L)
	
	comm = MPI.COMM_WORLD
	rk = comm.Get_rank()

	sm = MPS(space_size,'random')
	sm.bondims = np.load(MPS_dir+'/Bondim.npy')
	sm.matrices= np.load(MPS_dir+'/Mats.npy').tolist()

	mxBatch = 500
	batch_size = 40
	sm.leftCano()
	np.random.seed(rk)
	ds = ProjMeasureSet(space_size, mxBatch*batch_size, mps=sm)

	if typ=='random':
		measout = measout_dir+'/%s/%d/%d/'%(typ, space_size, sd)
	else:
		measout = measout_dir+'/%s/%d/'%(typ, space_size)
	try:
		os.makedirs(measout)
	except FileExistsError:
		pass
	os.chdir(measout)
	ds.save("R%dSet"%(rk))
	with open('VirtualTarget.txt','w') as fp:
		print("These results are measured from\n"+MPS_dir,file=fp)


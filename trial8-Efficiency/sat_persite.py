import numpy as np 
from main import saturation_statis, saturation_analyz
import sys
import os

if __name__ == '__main__':
	nlst1 = np.asarray([5,6,8,10,12,14,17,20,23,26,30,34,38,42])
	nlst2 = np.asarray([6,8,10,12,14,16,18,20,22,26,30,34,38,42])
	typ = sys.argv[1]
	if typ == 'dimer':
		nlst = nlst2
	else:
		nlst = nlst1

	os.chdir(typ)
	mean_std = np.empty((nlst.size,2), 'd')
	nfails   = np.empty((nlst.size,),np.uint32)
	for i,n in enumerate(nlst):
		os.chdir(str(n))
		fids = np.load('fids.npz')
		satV = saturation_statis(fids['real'],fids['V'],0.9997**n)
		mn,sd,nfail = saturation_analyz(satV)
		mean_std[i,0] = mn
		mean_std[i,1] = sd
		nfails[i] = nfail
		os.chdir("..")

	print(nfails)
	assert (nfails==0).all
	np.savez('sat_persite.npz', mean_std=mean_std)

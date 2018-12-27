# -*- coding: utf-8 -*-
from numpy import *
from numpy.random import rand, seed, randn
from numpy.linalg import norm,eig,solve,svd

import pickle
import os
import sys
from copy import deepcopy

DEBUG = True

def normalize(tens):
	tens /= norm(tens)

def binDiscrep(x,y):
	"""
count the number of different bits in x and y
	"""
	z = x^y
	if isinstance(z,ndarray):
		res = zeros(z.shape)
		while (z>0).any():
			# print(z)
			tmp = z >> 1
			res += z - (tmp<<1)
			z = tmp
	else:
		# print(bin(z))
		res = 0
		while (z>0):
			res += z % 2
			z = z>>1
	return res

class MPS:
	"""
	Base MPS Class
	Parameter:
		initial_state_type:
			"GHZ"/"W"/"dimer"/"cluster" and you will obtain the designated MPS state.
			"random": each entry drawn from uniform distribution U[0,1]
	Attributes:
		id_uncanonical:
			pointer to the A-tensor
			None: neither right-canon nor left-canon, need canonicalization
			in range(space_size-1): mixed-canonical form
				with self.matrix[id_uncanonical] being the only one not being cannonical
	"""
	def __init__(self, space_size, initial_state_type, **kwarg):
		self._n = space_size
		self.cutoff = 0.05

		if initial_state_type == "random" or initial_state_type == "randomforTFIM":
			init_bond_dimension = 2
			self.bondims = [init_bond_dimension] * self._n ## bond[i] connect i i+1
			self.bondims[-1] = 1
			self.matrices = []
			for i in range(space_size):
				self.matrices.append(randn(self.bondims[i-1], 2, self.bondims[i]) + rand(self.bondims[i-1], 2, self.bondims[i]) * 1.j)
			if initial_state_type == "randomforTFIM":
				self.H = kwarg['H']

		elif initial_state_type == "GHZ":
			init_bond_dimension = 2
			self.bondims = [init_bond_dimension] * self._n ## bond[i] connect i i+1
			self.bondims[-1] = 1
			self.matrices = []
			eiphi = exp(0.1j)
			l = zeros((1,2,2), dtype=complex)
			l[0,0,0] = eiphi
			l[0,1,1] = 1.
			r = zeros((2,2,1), dtype=complex)
			r[0,0,0] = eiphi
			r[1,1,0] = 1.
			bulk = zeros((2,2,2), dtype=complex)
			bulk[0,0,0] = eiphi
			bulk[1,1,1] = 1.
			self.matrices.append(l.copy())
			for i in range(space_size - 2):
				self.matrices.append(bulk.copy())
			self.matrices.append(r.copy())

		elif initial_state_type == "dimer":
			init_bond_dimension = 3
			self.bondims = [init_bond_dimension] * self._n ## bond[i] connect i i+1
			self.bondims[-1] = 1
			self.matrices = []
			
			l = zeros((1,2,3), dtype=complex)
			l[0,1,1] = 1
			l[0,0,2] = 1.
			r = zeros((3,2,1), dtype=complex)
			r[1,0,0] = -1.
			r[2,1,0] = 1.
			bulk = zeros((3,2,3), dtype=complex)
			bulk[0,1,1] = 1.
			bulk[0,0,2] = 1.
			bulk[1,0,0] = -1.
			bulk[2,1,0] = 1.
			self.matrices.append(l.copy())
			for i in range(space_size - 2):
				self.matrices.append(bulk.copy())
			self.matrices.append(r.copy())

		elif initial_state_type == "W":
			init_bond_dimension = 2
			self.bondims = [init_bond_dimension] * self._n ## bond[i] connect i i+1
			self.bondims[-1] = 1
			self.matrices = []
		
			l = zeros((1,2,2), dtype=complex)
			l[0,0,0] = 1
			l[0,1,1] = 1.
			r = zeros((2,2,1), dtype=complex)
			r[1,0,0] = 1.
			r[0,1,0] = exp(0.1j * (self._n - 1))
			bulk = zeros((2,2,2), dtype=complex)
			bulk[0,0,0] = 1.
			bulk[1,0,1] = 1.
			
			self.matrices.append(l.copy())
			for i in range(1, space_size - 1):
				bulk[0,1,1] = exp(0.1j * i)
				self.matrices.append(bulk.copy())
			self.matrices.append(r.copy())
		elif initial_state_type == "Cluster":
			init_bond_dimension = 4
			self.bondims = [init_bond_dimension] * self._n ## bond[i] connect i i+1
			self.bondims[-1] = 1
			self.bondims[0] = 2
			self.bondims[-2] = 2
			
			self.matrices = []
		
			l1 = zeros((1,2,2)) + 0.j
			l1[0,0,0] = 1.
			l1[0,0,1] = -1.
			l1 /= 2 ** (1. / 3.)
			
			l2 = zeros((2,2,4)) + 0.j
			l2[0,0,0] = 1.
			l2[0,0,1] = -1.
			
			l2[1,1,2] = 1.
			l2[1,1,3] = 1.
			
			l2 /= 2 ** (2. / 3.)
			
			bulk = zeros((4,2,4)) + 0.j
			bulk[0,0,0] = 1.
			bulk[0,0,1] = -1.
			bulk[1,1,2] = 1.
			bulk[1,1,3] = 1.
			
			bulk[2,0,0] = -1.
			bulk[2,0,1] = 1.
			bulk[3,1,2] = -1.
			bulk[3,1,3] = -1.
						
			bulk /= 2 
			
			r2 = zeros((4,2,2)) + 0.j
			r2[0,0,0] = 1.
			r2[1,1,1] = 1.
			
			r2[2,0,0] = -1.
			r2[3,1,1] = -1.
			
			r2 /= 2 ** (2. / 3.)
						
			r1 = zeros((2,2,1)) + 0.j
			r1[0,0,0] = 1.
			r1[1,0,0] = -1.
			r1 /= 2 ** (1. / 3.)
			
			self.matrices.append(l1.copy())
			self.matrices.append(l2.copy())
			for i in range(2, space_size - 2):
				self.matrices.append(bulk.copy())
			self.matrices.append(r2.copy())
			self.matrices.append(r1.copy())

		elif initial_state_type == "cluster":
			self.bondims = [4] * self._n
			self.bondims[0] = 2
			self.bondims[-1] = 1
			self.bondims[-2] = 2

			S_x = zeros((2,2))
			S_x[0][1] = 1
			S_x[1][0] = 1
			# S_y = array([[0, -1.j], [1.j, 0]])
			S_z = identity(2)
			S_z[1][1] = -1

			leftop = zeros((1, 2, 2, 2))
			centop = zeros((2, 2, 2, 2))
			rightop = zeros((2, 2, 2, 1))
			leftop[0, :, :, 0] = identity(2) * (2 ** (-1/3.))
			centop[0, :, :, 0] = identity(2) * (2 ** (-1/3.))
			rightop[0, :, :, 0] = identity(2) * (2 ** (-1/3.))
			leftop[0, :, :, 1] = S_z * (2 ** (-1/3.))
			centop[1, :, :, 1] = S_x * (2 ** (-1/3.))
			rightop[1, :, :, 0] = S_z * (2 ** (-1/3.))

			down = array([0, 1])

			bulk = tensordot(down, rightop, axes = [0, 1])[:, :, 0]
			bulk = tensordot(bulk, centop, axes = [1, 1])
			bulk = tensordot(bulk, leftop, axes = [2, 1])[:, :, :, 0, :, :]
			bulk = bulk.swapaxes(2, 3)
			dim = bulk.shape
			bulk = bulk.reshape(dim[0] * dim[1], dim[2], dim[3] * dim[4])

			leftone = tensordot(down, leftop, axes = [0, 1])

			lefttwo = tensordot(down, centop, axes = [0, 1])
			lefttwo = tensordot(lefttwo, leftop, axes = [1, 1])[:, :, 0, :, :]
			lefttwo = lefttwo.swapaxes(1, 2)
			dim = lefttwo.shape
			lefttwo = lefttwo.reshape(dim[0], dim[1], dim[2] * dim[3])

			righttwo = tensordot(down, rightop, axes = [0, 1])[:, :, 0]
			righttwo = tensordot(righttwo, centop, axes = [1, 1])
			dim = righttwo.shape
			righttwo = righttwo.reshape(dim[0] * dim[1], dim[2], dim[3])

			rightone = tensordot(down, rightop, axes = [0, 1])
			self.matrices = []
			self.matrices.append(leftone)
			self.matrices.append(lefttwo)
			N = self._n
			for i in range(N - 4):
				self.matrices.append(bulk)
			self.matrices.append(righttwo)
			self.matrices.append(rightone)
		
		self.id_uncanonical = None
		"""pointer to the A-tensor
		None: neither right-canon nor left-canon, need canonicalization
		in range(space_size-1): mixed-canonical form
			with self.matrix[id_uncanonical] being the only one not being cannonical
		"""
		self.merged_matrix = None
		self.merged_bond = None
		
	def __len__(self):
		return self._n

	def calcPsi(self, state, spinor_setting):
		p = self._n - 1
		rvec = ones((1,),dtype=complex)
		if self.merged_matrix is None:
			while p >= 0:
				tmp = dot(spinor_setting[p,(state>>p)%2], self.matrices[p])
				rvec = dot(tmp, rvec)
				p -= 1
		else:
			mergedBdp1 = self.merged_bond + 1
			while p > mergedBdp1:
				tmp = dot(spinor_setting[p,(state>>p)%2], self.matrices[p])
				rvec = dot(tmp, rvec)
				p -= 1
			tmp = dot(spinor_setting[p,(state>>p)%2], self.merged_matrix)
			p -= 1
			tmp = dot(spinor_setting[p,(state>>p)%2], tmp)
			rvec = dot(tmp, rvec)
			p -= 1
			while p >= 0:
				tmp = dot(spinor_setting[p,(state>>p)%2], self.matrices[p])
				rvec = dot(tmp, rvec)
				p -= 1
		return rvec[0]

	def calcProb(self, state, spinor_setting): 
		return abs(self.calcPsi(state, spinor_setting)) ** 2.
	
	def mergeBond(self, bond):
		self.merged_bond = bond
		self.merged_matrix = tensordot(self.matrices[bond],self.matrices[bond + 1], ([2], [0]))
		
	def rebuildBond(self, going_right, keep_bondim=False):
		U, s, V = svd(self.merged_matrix.reshape((self.bondims[(self.merged_bond - 1)] * 2, 2 * self.bondims[(self.merged_bond + 1)])))
		minibond = 2
		bdm = minibond
		if keep_bondim:
			bdm = min(self.bondims[self.merged_bond], s.size)
		else:
			while bdm < s.size and s[bdm] >= s[0] * self.cutoff:
				bdm += 1
		s = diag(s[:bdm])
		U = U[:, :bdm]
		V = V[:bdm, :]
		if going_right:
			V = dot(s, V)
			normalize(V)
			if self.id_uncanonical is not None: self.id_uncanonical = (self.merged_bond + 1)
		else:
			U = dot(U, s)
			normalize(U)
			if self.id_uncanonical is not None: self.id_uncanonical = self.merged_bond
		
		self.bondims[self.merged_bond] = bdm
		self.matrices[self.merged_bond] = reshape(U, (self.bondims[(self.merged_bond - 1) % self._n], 2, bdm))
		self.matrices[(self.merged_bond + 1)] = reshape(V, (bdm, 2, self.bondims[(self.merged_bond + 1)]))
		self.merged_bond = None
		self.merged_matrix=None
		return diag(s)
		
	def leftCano(self):
		if self.merged_bond is not None:
			self.rebuildBond(True)
		start = self.id_uncanonical if self.id_uncanonical is not None else 0
		for bond in range(start, self._n - 1):
			self.mergeBond(bond)
			self.rebuildBond(True, keep_bondim=True)
		self.id_uncanonical = self._n - 1

	def rightCano(self):
		if self.merged_bond is not None:
			self.rebuildBond(False)
		start = self.id_uncanonical if self.id_uncanonical is not None else self._n-2
		for bond in range(start, -1, -1):
			self.mergeBond(bond)
			self.rebuildBond(True, keep_bondim=True)
		self.id_uncanonical = 0

	def genSample(self, spinor_setting):
		v = ones((1,),dtype=complex)
		restate = 0
		if self.id_uncanonical == self._n-1:
			pointer = self._n - 1
			while pointer >= 0:
				# restate *= 2
				nv = [dot(dot(self.matrices[pointer],v),spinor_setting[pointer,i]) for i in range(2)]
				p = [norm(nv[i]) ** 2 for i in range(2)]
				if rand() < p[1] / (p[0] + p[1]):
					restate += 1<<pointer
					v = nv[1]
				else:
					v = nv[0]
				pointer -= 1
				normalize(v)
		elif self.id_uncanonical==0: #right canoned
			pointer = 0
			while pointer < self._n:
				nv = [dot(v,dot(spinor_setting[pointer,i],self.matrices[pointer])) for i in range(2)]
				p = [norm(nv[i]) ** 2 for i in range(2)]
				if rand() < p[1] / (p[0] + p[1]):
					restate += 1<<pointer
					v = nv[1]
				else:
					v = nv[0]
				pointer += 1
				normalize(v)
		else:
			raise ValueError("Uncanonical MPS cannot genSample")
		return restate
		
	def giveFidelity(self, mats, persite=True):
		assert self._n == len(mats)
		assert self.merged_bond is None
		if isinstance(mats, MPS):
			mats = mats.matrices
		p = self._n - 1
		res = dot(self.matrices[p][:,:,0],mats[p].conj())[:,:,0]
		while p>0:
			p -= 1
			res = dot(self.matrices[p],res)
			res = tensordot(res, mats[p].conj(),([1,2],[1,2]))
		if persite:
			return abs(res[0,0]) ** (1. / self._n)
		else:
			return abs(res[0,0])

	# def Give_Entanglement(self, cut_bond):
		
	# 	for bond in range(self._n - 2, cut_bond - 1, -1):
	# 		self.merge_bond(bond)
	# 		Entanglement_Spectrum = self.rebuild_bond(bond, False)
	# 	for bond in range(cut_bond, self._n - 1):
	# 		self.merge_bond(bond)
	# 		self.rebuild_bond(bond, True)
		
	# 	Entanglement_Spectrum = sort(Entanglement_Spectrum)
	# 	Entanglement_Entropy = - sum(Entanglement_Spectrum * mylog(Entanglement_Spectrum))
		
	# 	return (Entanglement_Entropy, Entanglement_Spectrum.tolist())

	# def Give_Correlation_length(self, site, operator):
	# 	###assume left canonical 
	# 	if operator == "z":
	# 		op = asarray([[1,0],[0,-1]])
	# 	E = tensordot(self.matrices[site], conj(self.matrices[site]), ([1], [1]))
	# 	E = swapaxes(E, 1, 2)
	# 	E = reshape(E, (self.bondims[(site - 1) % self._n] ** 2, self.bondims[site] ** 2))
	# 	w, v = eig(E)
	# 	ind = argsort(abs(w))
		
	# 	v = reshape(v,(self.bondims[site],self.bondims[site],-1))
		
	# 	rho = v[:,:,ind[-1]]
		
	# 	Mz = tensordot(self.matrices[site], op, ([1], [0]))
	# 	Ezl = tensordot(Mz, conj(self.matrices[site]), ([0,2], [0,1]))
	# 	Ezr = tensordot(Mz, conj(self.matrices[site]), ([2], [1]))
	# 	Ezr = tensordot(Ezr, rho, ([1,3], [0,1]))
		
	# 	Aml = tensordot(Ezl, v, ([0,1], [0,1]))
		
	# 	Ezr = reshape(Ezr, (self.bondims[site]**2))
	# 	v = reshape(v,(self.bondims[site]**2,-1))
	# 	Amr = solve(v, Ezr)
		
	# 	Am = Aml * Amr
		
	# 	Am = Am[ind[:]]
	# 	w = w[ind[:]]
		
	# 	print(Am.tolist(),w.tolist())
		
	# 	xi = - 1 / log(sort(abs(w)))
	# 	return xi

class ProjMeasureSet:
	"""
Outcomes of Projective Measurements
You can either assign an MPS as generator or give the (spinor_settings, states) lists
If an MPS is given, available measuring modes: uniform/2n+1/fix
	"""
	def __init__(self, space_size, chunk_size=1, init_chunk_num=0, fn_mode='uniform', mps=None, **kwarg):
		self.__n = space_size
		self.__mps = mps
		if mps is not None:
			assert mps._n == self.__n

		self._chunk = chunk_size
		self.__states = []
		self.__spinor_settings = []
		# self.__chunk_num = 0
		self.__V_cumulat = [0] # self.__V_cumulat[k] = \sum_{j=0}^{k-1} chunk_j
		self.setMode(fn_mode)

		if init_chunk_num > 0:
			self.sampNMeasureUpTo(init_chunk_num)

	def getN(self):
		return self.__n

	def __len__(self):
		return len(self.__spinor_settings)

	def __getitem__(self, k):
		return (self.__spinor_settings[k], self.__states[k])

	def setMode(self, mod):
		assert self.__spinor_settings == []
		self.__mode = mod
		if mod=='uniform':
			self.nsamp = self.unifromNsamp
		elif mod=='fix':
			self.nsamp = self.zNsamp
			if 'fix_theta_phi' in kwarg.keys():
				theta,phi = kwarg['fix_theta_phi']
				c1 = cos(theta*0.5)
				s1 = sin(theta*0.5)
				phas= exp(1.0j*phi)
				self._fix_spin = array([[c1,s1*phas], [-s1*phas.conj(),c1]])
			else:
				raise ValueError("To set in mode 'fix', a direction fix_theta_phi = (theta,phi) must be designated")
		# elif mod=='dynamic':
		# 	self.nsamp = self.dynaMeas
		else:
			raise ValueError("Unknown measuring mode %s"%mod)

	def measure(self, spinor_settings):
		if isinstance(spinor_settings,ndarray) and spinor_settings.ndim==3:
			spinor_settings = [spinor_settings]
		for setting in  spinor_settings:
			for _ in range(self._chunk):
				states = asarray([self.__mps.genSample(setting) for c in range(self._chunk)])
				self.__spinor_settings.append(setting)
				self.__states.append(states)
				self.__V_cumulat.append(self.__V_cumulat[-1] + self._chunk)
	
	def sampNMeasureUpTo(self, chunk_num):
		nsettings = chunk_num - len(self.__spinor_settings)
		if nsettings > 0:
			self.measure(self.nsamp(nsettings))

	def unifromNsamp(self, nsettings):
		N = nsettings * self.__n
		c = rand(N)*2-1
		c1 = (0.5*(1+c))**0.5 #cos(theta/2)
		s1 = (0.5*(1-c))**0.5 #sin(theta/2)
		phi = rand(N) * pi
		phas= exp(1.0j*phi)

		settings = empty((N,2,2),dtype=complex)
		settings[:,0,0] = c1
		settings[:,0,1] = s1*phas
		settings[:,1,0] = -s1*phas.conj()
		settings[:,1,1] = c1
		return settings.reshape(nsettings, self.__n,2,2)

	def fixNsamp(self, nsettings):
		spins = empty((self.__n,2,2),dtype=complex)
		for i in range(2):
			for j in range(2):
				spins[:,i,j]=self._fix_spin[i,j]
		return [spins]*nsettings

	def __getstate__(self):
		""" Return a dictionary of state values to be pickled """
		exclusion=['_ProjMeasureSet__mps']
		mydict={}
		for key in self.__dict__.keys():
			if key not in exclusion:
				mydict[key] = self.__dict__[key]
		return mydict

	def getV(self, ichuk_end, ichuk_start):
		return self.__V_cumulat[ichuk_end] - self.__V_cumulat[ichuk_start]

	def getMats(self):
		# DON'T revise the returned!
		return (self.__mps.matrices)

	def save(self, name):
		# if name[-4:] == '.npz':
		# 	name = name[:-4]
		# np.savez_compressed(name+'.npz', spin=asarray(self.__spinor_settings), stat=asarray(self.__states), mode=self.__mode)
		mps_name = name.split('/')
		mps_name = '/'.join(mps_name[:-1])
		if len(mps_name) > 0:
			mps_name = mps_name+'/stdmps.pickle'
		else:
			mps_name = 'stdmps.pickle'
		if name[0]=='/':
			mps_name = '/'+mps_name
		try:
			fp = open(mps_name,'rb')
			fp.close()
		except FileNotFoundError:
			fp = open(mps_name,'wb')
			pickle.dump(self.__mps, fp)
			fp.close()

		with open(name+".pickle","wb") as fsav:
			pickle.dump(self, fsav)

	def load(self, name):
		fload = open(name+'.pickle','rb')
		im = pickle.load(fload)
		for ky in im.__dict__.keys():
			self.__dict__[ky] = im.__dict__[ky]
		fload.close()

		# info = load(name+'.npz')
		# self.__spinor_settings = list(info['spin'])
		# self.__states = list(info['stat'])
		# assert self.__mode == str(info['mode'])

		mps_name = name.split('/')
		mps_name = '/'.join(mps_name[:-1])
		if len(mps_name) > 0:
			mps_name = mps_name+'/stdmps.pickle'
		else:
			mps_name = 'stdmps.pickle'
		if name[0]=='/':
			mps_name = '/'+mps_name
		with open(mps_name,'rb') as fmps:
			self.__mps = pickle.load(fmps)

	# def designateMPS(self, mm):
	# 	if self.__states==[]:
	# 		self.__mps = mm
	# 		self.__mps.leftCano()
	# 	else:
	# 		print('Error: You cannot designate another MPS because there are measured data.')



class MMDTomoTrainer(MPS):
	"""
Tomograpy Trainer:
Attribute:
	batch_size:	the number of bases in a batch of measurement
	init_num_bases: the number of bases initially
	A_size: For a certain base, the number of (virtual) measurements on the model
	measdat:	the measdat containing the outcome batch
	dat_head, dat_rear:		indices defining where the cumulants concern -- dat[dat_head:dat_rear]
	_cumulantL, _cumulantR:	1) _cumulantL[0] == ones((n_sample, 1))
							2) _cumulantR[0] == ones((1, n_sample))
							3) if j>0: _cumulantL[j] = A(0)...A(j-1)
									   _cumulantR[j] = A(N-j)...A(N-1)

When saving, dat won't be saved along with a TomoTrainer instance
so when loading, dat need extra attaching, using attach_dat
	"""
	def __init__(self, measdat, batch_size=5, init_num_bases=10, A_size=20, add_mode=True, grad_mode='plain'):
		MPS.__init__(self, measdat.getN(), 'random')
		self.leftCano()
		self.idx_in_history = -1
		self.loss = []
		self.succ_fid = []
		self.real_fid = []
		self.train_history = []
		self.add_mode = add_mode
		self.batch_size = batch_size
		
		self.dat = measdat
		self.dat_head = 0
		self.dat_rear = None
		self.init_num_bases = init_num_bases

		self.learning_rate = 0.1
		self.descent_steps = 1
		self.penalty_rate = None
		self.setGradMode(grad_mode) # plain/gnorm/RMSProp/Adam
		self.setKernel([(self._n*0.5),1])

		self.A_size = A_size
		# self.loss.append(self.calcLoss(measdat))

	def setGradMode(self, gm):
		if gm=='plain':
			self._multiSteps = self.__plainSteps
		elif gm=='gnorm':
			self._multiSteps = self.__gnormSteps
		elif gm=='RMSProp' or gm=='RMSprop':
			self._multiSteps = self.__rmspropSteps
		elif gm=='RMSProp-momentum':
			self._multiSteps = self.__rmspmomSteps
		elif self.grad_mode == 'Adam' or self.grad_mode == 'adam':
			self._multiSteps = self.__adamSteps
		else:
			raise ValueError('Unrecognized gradient mode %s'%gm)

	def setKernel(self, sigmas):
		self._sigmas = asarray(sigmas).reshape(1,1,-1)
		
	def _kernel(self,x,y):
		bds = binDiscrep(x,y).reshape(x.size,y.size,1)
		return exp(-0.5/self._sigmas* bds).mean(axis=2)

	def train(self, loops):
		if self.dat_rear is not None:
			tmp_rear = self.batch_size + self.dat_rear
		else:
			tmp_rear = self.init_num_bases
		self.dat.sampNMeasureUpTo(tmp_rear)
		if DEBUG: print("Measured the target state on %d bases in total %d times"%(len(self.dat),self.dat.getV(-1,0)), flush=True)

		if not self.add_mode:
			self.dat_head = self.dat_rear
		self.dat_rear = tmp_rear
		late_mats = deepcopy(self.matrices)
		for lp in range(loops):
			if DEBUG: print("# %d"%lp, end='\t')
			for going_right, rag in [(False,range(self._n-2,0,-1)), (True,range(0,self._n-2))]:
				self._measureModel(going_right)
				if DEBUG: print('measured the model %d times'%(self.model_meas_dat.getV(-1,0)),end='\t',flush=True)
				self._initCumulant(going_right)
				if DEBUG: print('Cumulated',flush=True)
				for b in rag:
					self.mergeBond(b)
					self._multiSteps()
					self.rebuildBond(going_right)
					self._updateCumulant(going_right)
					# if DEBUG: print('bond: %d'%b, end='\r', flush=True)
				# self.loss.append(self._showLoss())
				# if DEBUG: print('Loss=%.3e'%self.loss[-1],self.bondims,flush=True)
				# print('')
		self.idx_in_history += 1
		self.train_history.append((self.cutoff, self.descent_steps, self.learning_rate, self.penalty_rate, self.dat_head, self.dat_rear, loops))	
		
		self.succ_fid.append(self.giveFidelity(late_mats))
		self.real_fid.append(self.giveFidelity(self.dat.getMats()))
		if DEBUG: print(self.real_fid[-1])

	def _measureModel(self, right_canoning):
		"""
On the same set of bases as the measurement on the target state
measure the tomographic state
		"""
		if right_canoning:
			self.rightCano()
		else:
			self.leftCano()
		self.model_meas_dat = ProjMeasureSet(self._n, self.A_size, mps=self)
		self.model_meas_dat.measure(self.dat[:][0])

	def _initCumulant(self, right_canoned):
		"""
Initialize self._cumulantL & self._cumulantR for a batch in TOMOGRAPHIC-state measurement outcomes
During the training process, it will be kept unchanged that:
1) _cumulantL[0] == ones((n_sample, 1))
2) _cumulantR[0] == ones((1, n_sample))
3) if j>0: _cumulantL[j] = A(0)...A(j-1)
		   _cumulantR[j] = A(N-j)...A(N-1)
		"""
		V = self.model_meas_dat.getV(self.dat_rear, self.dat_head)
		self._cumulantL = [ones((V, 1))]
		self._cumulantR = [ones((V, 1))]

		if not right_canoned:
			assert self.id_uncanonical == self._n-1 or self.id_uncanonical==self._n-2
			for n in range(0, self._n-2):
				tmp = empty((V,self.bondims[n]), dtype=complex)
				p = 0
				# try:
				for spin, states in zip(*(self.model_meas_dat[self.dat_head:self.dat_rear])):
					mid = tensordot(self._cumulantL[-1][p], self.matrices[n], ([0],[0]))
					for state in states:
						# print(p,end='\t',flush=True)
						tmp[p,:] = dot(spin[n,(state>>n)%2], mid)
						p += 1
				# except ValueError:
				# 	spin, states = self.model_meas_dat[self.dat_head:self.dat_rear]
				# 	print(len(spin))
				# 	print(len(states), len(states[0]))
				# 	raise ValueError
				self._cumulantL.append(asarray(tmp))
		else:
			assert self.id_uncanonical==0 or self.id_uncanonical==1
			for n in range(self._n-1, 1, -1):
				tmp = empty((V,self.bondims[n-1]), dtype=complex)
				p = 0
				for spin, states in zip(*(self.model_meas_dat[self.dat_head:self.dat_rear])):
					mid = dot(self.matrices[n], self._cumulantR[-1][p])
					for state in states:
						# try:
						tmp[p,:] = dot(mid,spin[n,(state>>n)%2])
						# except:
						# 	# print(mid.shape, spin[n,(state>>n)%2].shape)
						# 	# print(dot(mid,spin[n,(state>>n)%2]).shape)
						# 	print(n,self.bondims)
						# 	sys.exit(4)
						p += 1
				self._cumulantR.append(asarray(tmp))

	def _updateCumulant(self, going_right):
		k = len(self._cumulantL)-1
		V = self.model_meas_dat.getV(self.dat_rear, self.dat_head)
		tmp = empty((V,self.bondims[k]), dtype=complex)
		if going_right:
			p = 0
			for spin,states in zip(*(self.model_meas_dat[self.dat_head:self.dat_rear])):
				mid = tensordot(self._cumulantL[-1][p], self.matrices[k], ([0],[0]))
				for state in states:
					tmp[p,:] = dot(spin[k,(state>>k)%2], mid)
					p += 1
			self._cumulantL.append(asarray(tmp))
			self._cumulantR.pop()
		else:
			k += 1
			p = 0
			for spin,states in zip(*(self.model_meas_dat[self.dat_head:self.dat_rear])):
				mid = dot(self.matrices[k], self._cumulantR[-1][p])
				for state in states:
					tmp[p,:] = dot(mid, spin[k,(state>>k)%2])
					p += 1
			self._cumulantR.append(asarray(tmp))
			self._cumulantL.pop()

	def __plainSteps(self):
		for j in range(self.descent_steps):
			ngrad = self._neGrad()
			self.merged_matrix += ngrad * self.learning_rate
			if j < self.descent_steps-1:
				normalize(self.merged_matrix)

	def _neGrad(self):
		"""
 - \partial L/ \partial \theta*
		"""
		k = self.merged_bond
		negrad = zeros((self.bondims[(k-1) % self._n], 2, 2, self.bondims[(k+1) % self._n]), dtype=complex)
		ptc = 0
		for ispin, (spin, xs) in enumerate(zip(*(self.model_meas_dat[self.dat_head:self.dat_rear]))):
			tmp = zeros((self.bondims[(k-1) % self._n], 2, 2, self.bondims[(k+1) % self._n]), dtype=complex)
			target_zs = self.dat[ispin][1]
			xs1 = xs.reshape(-1,1)
			# try:
			kxz_mean_minus_kxy_mean = self._kernel(xs1, target_zs).mean(axis=1) - self._kernel(xs1, xs).mean(axis=1)
			# except:
			# print(xs1.shape, target_zs.shape)
			# print(self._kernel(xs1,target_zs).shape)
			# sys.exit(-2)
			for ix, x in enumerate(xs):
				tmp += self._gradLnProbs(spin,x,ptc) * kxz_mean_minus_kxy_mean[ix]
				ptc += 1
			negrad += tmp/xs.size
		ptc == self.model_meas_dat.getV(self.dat_rear, self.dat_head)
		# try:
		# 	assert  V
		# except:
		# 	print(V-ptc)
		# 	sys.exit(-3)
		negrad *= 2.0/(self.dat_rear-self.dat_head)
		if self.penalty_rate is not None and abs(self.penalty_rate) > 1e-10:
			S2_penalty = tensordot(conj(self.merged_matrix), self.merged_matrix, ([2,3], [2,3]))
			S2_penalty = tensordot(S2_penalty, conj(self.merged_matrix), ([2,3], [0,1]))
			expS2 = tensordot(S2_penalty, self.merged_matrix, ([0,1,2,3], [0,1,2,3]))
			negrad += conj(S2_penalty / expS2) * self.penalty_rate
		return negrad

	def _gradLnProbs(self, spin, state, pt_cumulant):
		"""
gradient of ln(p[rhoMPS, n](x)) wrt A^{(k,k+1)}
		"""
		k = self.merged_bond
		psi = self._showPsi(spin, state, pt_cumulant)
		if abs(psi) < 1e-15:
			print(self.calcPsi(state, spin))
			sys.exit(10)
		conjgrad = (self._cumulantL[-1][pt_cumulant].reshape(-1,1,1,1))\
					*(spin[k,(state>>k)%2].reshape(1,-1,1,1))\
					*(spin[k+1,(state>>(k+1))%2].reshape(1,1,-1,1))\
					*(self._cumulantR[-1][pt_cumulant].reshape(1,1,1,-1))\
					/psi
		return conj(conjgrad) - self.merged_matrix

	def _showPsi(self, spin, state, pt_cumulant):
		"""Evaluate with cumulant"""
		rvec = self._cumulantR[-1][pt_cumulant]
		k = self.merged_bond
		if k is not None:
			tmp = dot(self.merged_matrix, rvec)
			tmp = dot(tmp, spin[k+1,(state>>(k+1))%2])
			tmp = dot(tmp, spin[k, (state>>k)%2])
		else:
			k = len(self._cumulantL)-1
			tmp = dot(dot(self.matrices[k+1], rvec), spin[k+1,(state>>(k+1))%2])
			tmp = dot(dot(self.matrices[k], tmp), spin[k,(state>>(k))%2])
		return dot(self._cumulantL[-1][pt_cumulant], tmp)

	def _showLoss(self):
		"""Evaluate with cumulant"""
		res = 0
		for k in range(self.dat_head, self.dat_rear):
			spin, zs  = self.dat[k]
			_, xs = self.model_meas_dat[k]
			xs1 = xs.reshape(-1,1)
			res += self._kernel(xs1,xs).mean() -2*self._kernel(xs1,zs).mean() + self._kernel(zs.reshape(-1,1),zs).mean()
		res /= (self.dat_rear-self.dat_head)
		return res

	def save(self, stamp):
		try:
			mkdir('./'+stamp+'/')
		except:
			shutil.rmtree(stamp)
			mkdir('./'+stamp+'/')
		chdir('./'+stamp+'/')
		
		save('Bondim.npy',self.bondims)
		save('Mats.npy',self.matrices)
		save('idxInTrainHistory.npy',self.idx_in_history)

		chdir('..')
		save('Loss.npy',self.loss)
		savez('Fidelity.npz', succ=self.succ_fid, real=self.real_fid)
		with open('TrainHistory.pickle', 'wb') as thp:
			pickle.dump(self.train_history, thp)

	def load(self, srch_pwd=None):
		if srch_pwd is not None:
			oripwd = os.getcwd()
			os.chdir(srch_pwd)
		self.bondims = load('Bondim.npy').tolist()
		self.__n = len(self.bondims)
		try:
			self.Loss = load('Loss.npy').tolist()
		except FileNotFoundError:
			self.Loss = []

		self.idx_in_history = np.load('idxInTrainHistory.npy')
		with open('../TrainHistory.pickle', 'rb') as thp:
			self.train_history = pickle.load(thp)[:self.idx_in_history]
		self.cutoff,self.descent_steps,self.learning_rate,self.penalty_rate,\
			self.dat_head, self.dat_rear, _lp = self.train_history[-1]

		fids = np.load('../Fidelity.npz')[:self.idx_in_history]
		self.succ_fid = fids['succ'].tolist()
		self.real_fid = fids['real'].tolist()
		self.matrices = load('Mats.npy').tolist()
		
		self.merged_bond = None
		self.merged_matrix=None
		if srch_pwd is not None:
			os.chdir(oripwd)

if __name__ == '__main__':
	seed(1)
	sm = MPS(6,'GHZ')
	sm.leftCano()
	ds = ProjMeasureSet(6, 10, 0, mps=sm)
	mmdtomo = MMDTomoTrainer(ds, 5, 20)
	mmdtomo.cutoff = 0.5
	for j in range(5):
		mmdtomo.train(2)

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:37:55 2017

@author: lenovo
"""

## todo list fidelity basis generate

from numpy import *
import numpy as np
from numpy.random import rand, seed, randn, randint
from numpy.linalg import norm,eig,solve,svd
from copy import deepcopy
from os import mkdir,chdir
import shutil
import pickle
import os
import sys

def normalize(tens):
	tens /= norm(tens)

class MPS:
	"""
	Base MPS Class
	Parameter:
		initial_state_type:
			"W"/"dimer"/"cluster" and you will obtain the designated MPS state.
			"random": each entry drawn from uniform distribution U[0,1]
	Attributes:
		id_uncanonical:
			-1: neither right-canon nor left-canon, need canonicalization
			in range(space_size-1): self.matrix[id_uncanonical] is the only one not being cannonical
			space_size-1: left-canonicalized
	"""
	def __init__(self, space_size, initial_state_type, **kwarg):
		self._n = space_size
		self.cutoff = 0.01
		self.Dmin = 2
		if 'Dmin' in kwarg:
			self.Dmin = kwarg['Dmin']
		self.Dmax = None

		if initial_state_type == "random" or initial_state_type == "randomforTFIM":
			if "randomInitMaxD" in kwarg:
				self.bondims = randint(self.Dmin, kwarg['randomInitMaxD'],self._n)
			else:
				init_bond_dimension = 2
				self.bondims = [init_bond_dimension] * self._n ## bond[i] connect i i+1
			self.bondims[-1] = 1
			self.matrices = []
			for i in range(space_size):
				self.matrices.append(randn(self.bondims[i-1], 2, self.bondims[i]) + rand(self.bondims[i-1], 2, self.bondims[i]) * 1.j)
			if initial_state_type == "randomforTFIM":
				self.H = kwarg['H']

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

			S_x = np.zeros((2,2))
			S_x[0][1] = 1
			S_x[1][0] = 1
			# S_y = np.array([[0, -1.j], [1.j, 0]])
			S_z = np.identity(2)
			S_z[1][1] = -1

			leftop = np.zeros((1, 2, 2, 2))
			centop = np.zeros((2, 2, 2, 2))
			rightop = np.zeros((2, 2, 2, 1))
			leftop[0, :, :, 0] = np.identity(2) * (2 ** (-1/3.))
			centop[0, :, :, 0] = np.identity(2) * (2 ** (-1/3.))
			rightop[0, :, :, 0] = np.identity(2) * (2 ** (-1/3.))
			leftop[0, :, :, 1] = S_z * (2 ** (-1/3.))
			centop[1, :, :, 1] = S_x * (2 ** (-1/3.))
			rightop[1, :, :, 0] = S_z * (2 ** (-1/3.))

			down = np.array([0, 1])

			bulk = np.tensordot(down, rightop, axes = [0, 1])[:, :, 0]
			bulk = np.tensordot(bulk, centop, axes = [1, 1])
			bulk = np.tensordot(bulk, leftop, axes = [2, 1])[:, :, :, 0, :, :]
			bulk = bulk.swapaxes(2, 3)
			dim = bulk.shape
			bulk = bulk.reshape(dim[0] * dim[1], dim[2], dim[3] * dim[4])

			leftone = np.tensordot(down, leftop, axes = [0, 1])

			lefttwo = np.tensordot(down, centop, axes = [0, 1])
			lefttwo = np.tensordot(lefttwo, leftop, axes = [1, 1])[:, :, 0, :, :]
			lefttwo = lefttwo.swapaxes(1, 2)
			dim = lefttwo.shape
			lefttwo = lefttwo.reshape(dim[0], dim[1], dim[2] * dim[3])

			righttwo = np.tensordot(down, rightop, axes = [0, 1])[:, :, 0]
			righttwo = np.tensordot(righttwo, centop, axes = [1, 1])
			dim = righttwo.shape
			righttwo = righttwo.reshape(dim[0] * dim[1], dim[2], dim[3])

			rightone = np.tensordot(down, rightop, axes = [0, 1])
			self.matrices = []
			self.matrices.append(leftone)
			self.matrices.append(lefttwo)
			N = self._n
			for i in range(N - 4):
				self.matrices.append(bulk)
			self.matrices.append(righttwo)
			self.matrices.append(rightone)
		
		elif initial_state_type == 'Neel':
			self.bondims = [1]*self._n
			self.matrices = []
			odd = np.zeros((1,2,1))
			odd[0,1,0] = 1
			eve = np.zeros((1,2,1))
			eve[0,0,0] = 1
			for i in range(self._n):
				if i%2==0:
					self.matrices.append(eve)
				else:
					self.matrices.append(odd)
			
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

	def calcPsi(self, spinor_outcome):
		p = self._n - 1
		rvec = ones((1,),dtype=complex)
		if self.merged_matrix is None:
			while p >= 0:
				tmp = dot(spinor_outcome[p], self.matrices[p])
				rvec = dot(tmp, rvec)
				p -= 1
		else:
			mergedBdp1 = self.merged_bond + 1
			while p > mergedBdp1:
				tmp = dot(spinor_outcome[p], self.matrices[p])
				rvec = dot(tmp, rvec)
				p -= 1
			tmp = dot(spinor_outcome[p], self.merged_matrix)
			p -= 1
			tmp = dot(spinor_outcome[p], tmp)
			rvec = dot(tmp, rvec)
			p -= 1
			while p >= 0:
				tmp = dot(spinor_outcome[p], self.matrices[p])
				rvec = dot(tmp, rvec)
				p -= 1
		return rvec[0]

	def calcProb(self, spinor_outcome): 
		return np.abs(self.calcPsi(spinor_outcome)) ** 2.
	
	def mergeBond(self, bond):
		self.merged_bond = bond
		self.merged_matrix = tensordot(self.matrices[bond],self.matrices[bond + 1], ([2], [0]))
		
	def rebuildBond(self, going_right, keep_bondim=False):
		U, s, V = svd(reshape(self.merged_matrix, (self.bondims[(self.merged_bond - 1)] * 2, 2 * self.bondims[(self.merged_bond + 1)])))
		bdm = self.Dmin
		if keep_bondim:
			bdm = min(self.bondims[self.merged_bond], s.size)
		else:
			while bdm < s.size and s[bdm] >= s[0] * self.cutoff:
				bdm += 1
		if self.Dmax is not None and self.Dmax < bdm:
			bdm = self.Dmax
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
			self.rebuildBond(False, keep_bondim=True)
		self.id_uncanonical = 0

	# def genSample(self,spinor_setting):
	# 	v = ones((1,),dtype=complex)
	# 	restate = 0
	# 	pointer = self._n - 1
	# 	while pointer >= 0:
	# 		restate *= 2
	# 		nv = [dot(dot(self.matrices[pointer],v),spinor_setting[pointer,i]) for i in range(2)]
	# 		p = [norm(nv[i]) ** 2 for i in range(2)]
	# 		if rand() < p[1] / (p[0] + p[1]):
	# 			restate += 1
	# 			v = nv[1]
	# 		else:
	# 			v = nv[0]
	# 		pointer -= 1
	# 		normalize(v)
	# 	return restate
	def genSample(self, spinor_setting):
		respin = zeros((self._n,2), complex)
		restate = 0
		if self.id_uncanonical is not None:
			p_unnorm = self.id_uncanonical
			assert p_unnorm<self._n and p_unnorm>=0
			# sampling order: p_unnorm, p_unnorm-1,...,0, p_unnorm+1,p_unnorm+2,...,N-1
			mat = dot(spinor_setting[p_unnorm,1], self.matrices[p_unnorm])
			mat1_norm = norm(mat)
			if rand() < mat1_norm**2:
				respin[p_unnorm,:] = spinor_setting[p_unnorm,1,:]
				restate += 1<<p_unnorm
				mat /= mat1_norm
			else:
				respin[p_unnorm,:] = spinor_setting[p_unnorm,0,:]
				mat = dot(spinor_setting[p_unnorm,0], self.matrices[p_unnorm])
				mat /= norm(mat)
			for p in range(p_unnorm-1,-1,-1):
				nmat1 = dot(dot(spinor_setting[p,1],self.matrices[p]),mat)
				mat1_norm = norm(nmat1)
				if rand() < mat1_norm**2:
					respin[p,:] = spinor_setting[p,1,:]
					restate += 1<<p
					mat = nmat1/mat1_norm
				else:
					respin[p,:] = spinor_setting[p,0,:]
					mat = dot(dot(spinor_setting[p,0],self.matrices[p]),mat)
					mat /= norm(mat)
			for p in range(p_unnorm+1,self._n):
				nmat1 = dot(mat,dot(spinor_setting[p,1],self.matrices[p]))
				mat1_norm = norm(nmat1)
				if rand() < mat1_norm**2:
					respin[p,:] = spinor_setting[p,1,:]
					restate += 1<<p
					mat = nmat1/mat1_norm
				else:
					respin[p,:] = spinor_setting[p,0,:]
					mat = dot(mat,dot(spinor_setting[p,0],self.matrices[p]))
					mat /= norm(mat)
		else:
			raise ValueError("Uncanonical MPS cannot genSample")
		return restate, respin
		
	def giveFidelity(self, mats, persite=False):
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
			return np.abs(res[0,0]) ** (1. / self._n)
		else:
			return np.abs(res[0,0])

	# def Give_Entanglement(self, cut_bond):
		
	# 	for bond in range(self._n - 2, cut_bond - 1, -1):
	# 		self.merge_bond(bond)
	# 		Entanglement_Spectrum = self.rebuild_bond(bond, False)
	# 	for bond in range(cut_bond, self._n - 1):
	# 		self.merge_bond(bond)
	# 		self.rebuild_bond(bond, True)
		
	# 	Entanglement_Spectrum = sort(Entanglement_Spectrum)
	# 	Entanglement_Entropy = - np.sum(Entanglement_Spectrum * mylog(Entanglement_Spectrum))
		
	# 	return (Entanglement_Entropy, Entanglement_Spectrum.tolist())

	# def Give_Correlation_length(self, site, operator):
	# 	###assume left canonical 
	# 	if operator == "z":
	# 		op = asarray([[1,0],[0,-1]])
	# 	E = tensordot(self.matrices[site], conj(self.matrices[site]), ([1], [1]))
	# 	E = swapaxes(E, 1, 2)
	# 	E = reshape(E, (self.bondims[(site - 1) % self._n] ** 2, self.bondims[site] ** 2))
	# 	w, v = eig(E)
	# 	ind = argsort(np.abs(w))
		
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
		
	# 	xi = - 1 / log(sort(np.abs(w)))
	# 	return xi
	

class ProjMeasureSet:
	"""
Outcomes of Projective Measurements
You can either assign an MPS as generator or give the (spinor_settings, states) lists
If an MPS is given, available measuring modes: uniform/2n+1/onlyZ/dynamic
Attribute:
	noise: the noise level in the measurement. It's the probability that a random outcome is obtained
	"""
	def __init__(self, space_size, init_set_size=0, mode='uniform', mps=None, noise=0):
		self.__n = space_size
		self.__mps = mps
		# self.states = []
		self.spinor_outcomes = []
		self.setMode(mode)
		self.noise = noise
		if init_set_size > 0:
			self.measureUpTo(init_set_size)

	def getN(self):
		return self.__n

	def designateMPS(self, mm):
		if self.states==[]:
			self.__mps = mm
			self.__mps.leftCano()
		else:
			print('Error: You cannot designate another MPS because there are measured data.')

	def _singlemeas(self):
		setting = self.__gen_setting()
		if self.noise > 0 and rand() < self.noise:
			binoutcome = randint(0,2,self.__n)
			return asarray([setting[j,binoutcome[j]] for j in range(self.__n)])
		else:
			return self.__mps.genSample(setting)[1]

	def setMode(self, mod):
		# if self.states == []:
		self.__mode = mod
		if mod=='uniform':
			self.__gen_setting = self.uniforMeas
		elif mod=='onlyZ':
			self.__gen_setting = self.zzMeas
		else:
			raise ValueError("Unknown measuring mode %s"%mod)
	
	def uniforMeas(self):
		setting = empty((self.__n, 2,2),dtype=complex)
		c = rand(self.__n)*2-1
		c1 = (0.5*(1+c))**0.5 #cos(theta/2)
		s1 = (0.5*(1-c))**0.5 #sin(theta/2)
		phi = rand(self.__n) * pi
		phas= exp(1.0j*phi)
		setting[:,0,0] = c1
		setting[:,0,1] = s1*phas
		setting[:,1,0] = -s1*phas.conj()
		setting[:,1,1] = c1
		return setting
		# binstate, spinor_state = self.__mps.genSample(setting)
		# return spinor_state

	def zzMeas(self):
		setting = empty((self.__n, 2,2),dtype=int8)
		setting[:,0,0] = 1
		setting[:,1,1] = 1
		setting[:,1,0] = 0
		setting[:,0,1] = 0
		return setting

	# def dynaMeas(self, discriminator):
	# 	assert discriminator.getN() == self._n
	# 	setting = discriminator.gen()
	# 	return setting, self.__mps.genSample(setting)
		
	def measureUpTo(self, size):
		for _ in range(size-len(self.spinor_outcomes)):
		# if size <= len(self.states): doing nothing
			spin_state = self._singlemeas()
			self.spinor_outcomes.append(spin_state)

	def getMats(self):
		# DON'T revise the returned!
		return (self.__mps.matrices)

	def __getstate__(self):
		""" Return a dictionary of state values to be pickled """
		exclusion=['_ProjMeasureSet__mps']
		mydict={}
		for key in self.__dict__.keys():
			if key not in exclusion:
				mydict[key] = self.__dict__[key]
		return mydict

	def save(self, name):
		# if name[-4:] == '.npz':
		# 	name = name[:-4]
		# np.savez_compressed(name+'.npz', spin=asarray(self.spinor_settings), stat=asarray(self.states), mode=self.__mode)
		
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
			fp = open(mps_name+'.bondim','w')
			print(self.__mps.bondims,file=fp)
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
		# self.spinor_settings = list(info['spin'])
		# self.states = list(info['stat'])
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
			self._ProjMeasureSet__mps = pickle.load(fmps)

# class Discriminator:
# 	def __init__(self, space_size, lr=2e-3):
# 		self.__n = space_size
# 		self.spinor_angles = []
# 		self.lr = lr

# 	def getN(self):
# 		return self.__n
# 	def gen(self):
# 		pass

class MLETomoTrainer(MPS):
	"""
Tomograpy Trainer:
Attribute:
	dat:	the dataset containing the outcome batch
	dat_head, dat_rear:		indices defining where the cumulants concern -- dat[dat_head:dat_rear]
	_cumulantL, _cumulantR:	1) _cumulantL[0] == ones((n_sample, 1))
							2) _cumulantR[0] == ones((1, n_sample))
							3) if j>0: _cumulantL[j] = A(0)...A(j-1)
									   _cumulantR[j] = A(N-j)...A(N-1)

When saving, dat won't be saved along with a TomoTrainer instance
so when loading, dat need extra attaching, using attach_dat
	"""
	def __init__(self, dataset, batch_size=40, initV=80, add_mode=True, **kwarg):
		if kwarg == {}:
			MPS.__init__(self, dataset.getN(), 'random')
		else:
			MPS.__init__(self, dataset.getN(), **kwarg)
		self.leftCano()
		self.loss = []
		self.succ_fid = []
		self.real_fid = []
		self.train_history = []
		self.ibatch = -1
		self.add_mode = add_mode
		self.batch_size = batch_size
		self.dat_head = 0
		self.dat_rear = initV
		self.attach_dat(dataset)

		self.grad_mode = 'plain' # plain/gnorm/RMSProp/Adam
		self.learning_rate = 0.1
		self.descent_steps = 2
		self.penalty_rate = None

		# self.loss.append(self.calcLoss(dataset))
	def attach_dat(self, dataset):
		self.dat = dataset
		dataset.measureUpTo(self.dat_rear)

	def _showPsi(self, istate):
		"""Evaluate with cumulant"""
		# state = self.dat.states[istate]
		spin = self.dat.spinor_outcomes[istate]
		istate -= self.dat_head
		lvec = self._cumulantL[-1][istate]
		rvec = self._cumulantR[-1][istate]
		k = self.merged_bond
		if k is not None:
			tmp = dot(self.merged_matrix, rvec)
			tmp = dot(tmp, spin[k+1])
			tmp = dot(tmp, spin[k])
		else:
			k = len(self._cumulantL)-1
			tmp = dot(dot(self.matrices[k+1], rvec), spin[k+1])
			tmp = dot(dot(self.matrices[k], tmp), spin[k])
		return dot(lvec, tmp)

	def _showLoss(self):
		"""Evaluate with cumulant"""
		res = -2*mean([log(np.abs(self._showPsi(i))) for i in range(self.dat_head, self.dat_rear)])
		print('Loss =',res)
		return res

	def calcLoss(self, dataset=None, head=0, rear=-1):
		"""
		Evaluate NLL on dataset
		if dataset is None:
			dataset = self.dat
			and it will call self._showPsi, using cumulant
		else: 
			given dataset
		"""
		if dataset is None:
			dataset = self.dat
		if rear == -1:
			rear = len(dataset.states)
		if dataset:
			res = -2*mean([log(self.calcProb(dataset.spinor_outcomes[i])) for i in range(head, rear)])
		else:
			res = sum([log(np.abs(self._showPsi(i))) for i in range(max(self.dat_head,head), min(rear,self.dat_rear))])
			for i in range(head, max(self.dat_head,head)):
				res += log(self.calcProb(dataset.spinor_outcomes[i]))
			for i in range(min(rear,self.dat_rear), rear):
				res += log(self.calcProb(dataset.spinor_outcomes[i]))
			res /= (rear-head)
		print('Loss =',res)
		return res

	def _initCumulant(self):
		"""
		Initialize self.cumulants for a batch.
		During the training process, it will be kept unchanged that:
		1) _cumulantL[0] == ones((n_sample, 1))
		2) _cumulantR[0] == ones((1, n_sample))
		3) if j>0: _cumulantL[j] = A(0)...A(j-1)
				   _cumulantR[j] = A(N-j)...A(N-1)
		"""
		if self.id_uncanonical == self._n-1:
			self.leftCano()
		self._cumulantL = [ones((self.dat_rear-self.dat_head, 1))]
		for n in range(0, self._n-2):
			tmp = []
			for k in range(self.dat_head, self.dat_rear):
				mid = tensordot(self._cumulantL[-1][k-self.dat_head], self.matrices[n], ([0],[0]))
				# try:
				mid = dot(self.dat.spinor_outcomes[k][n], mid)
				# except TypeError:
				# 	print(self.dat.spinor_settings[k].shape, n, (self.dat.states[k]>>n)%2)
				# 	sys.exit(-3)
				tmp.append(mid)
			self._cumulantL.append(asarray(tmp))
		self._cumulantR = [ones((self.dat_rear-self.dat_head, 1))]

	def _updateCumulant(self, going_right):
		k = len(self._cumulantL)-1
		if going_right:
			tmp = []
			for i in range(self.dat_head, self.dat_rear):
				mid = tensordot(self._cumulantL[-1][i-self.dat_head], self.matrices[k], ([0],[0]))
				mid = dot(self.dat.spinor_outcomes[i][k], mid)
				tmp.append(mid)
			self._cumulantL.append(asarray(tmp))
			self._cumulantR.pop()
		else:
			k += 1
			tmp = []
			for i in range(self.dat_head, self.dat_rear):
				mid = tensordot(self._cumulantR[-1][i-self.dat_head], self.matrices[k], ([0],[2]))
				mid = dot(mid, self.dat.spinor_outcomes[i][k])
				tmp.append(mid)
			self._cumulantR.append(asarray(tmp))
			self._cumulantL.pop()

	def _neGrad(self):
		"""negative gradient"""
		conjgrad = zeros((self.bondims[(self.merged_bond - 1) % self._n], 2, 2, self.bondims[(self.merged_bond + 1) % self._n]), dtype=complex)
		k = self.merged_bond
		# kp1 = (k+1)%self._n
		# km1 = (k-1)%self._n
		for istate in range(self.dat_head, self.dat_rear):
			spin = self.dat.spinor_outcomes[istate]
			# state= self.dat.states[istate]
			psi = self._showPsi(istate)
			conjgrad += (self._cumulantL[-1][istate-self.dat_head].reshape(-1,1,1,1))\
						*(spin[k].reshape(1,-1,1,1))\
						*(spin[k+1].reshape(1,1,-1,1))\
						*(self._cumulantR[-1][istate-self.dat_head].reshape(1,1,1,-1))\
						/psi
		conjgrad /= (self.dat_rear - self.dat_head)
		# conjgrad -= conj(self.merged_matrix) 
		
		if self.penalty_rate is not None and self.penalty_rate != 0.:
			S2_penalty = tensordot(conj(self.merged_matrix), self.merged_matrix, ([2,3], [2,3]))
			S2_penalty = tensordot(S2_penalty, conj(self.merged_matrix), ([2,3], [0,1]))
			expS2 = tensordot(S2_penalty, self.merged_matrix, ([0,1,2,3], [0,1,2,3]))
			# print("current S2:", -log(expS2))
			# self.merged_matrix += conj(gradient) * self.learning_rate - S2_penalty / expS2 * penalty_rate

			#<<<revision by zlwag 06092017
			conjgrad += S2_penalty / expS2 * self.penalty_rate
			#>>>revision by zlwag 06092017
		
		# grad = conj(conjgrad) * self.learning_rate
		# if self.gnorm:
		# 	gnorm = norm(grad)/(grad.size**0.25)
		# 	if gnorm < 1:
		# 		grad /= gnorm
		# self.merged_matrix += grad
		# normalize(self.merged_matrix)
		return conj(conjgrad)

	def _multiSteps(self):
		if self.grad_mode == 'plain':
			for j in range(self.descent_steps):
				ngrad = self._neGrad()
				self.merged_matrix += ngrad * self.learning_rate
				if j < self.descent_steps-1:
					normalize(self.merged_matrix)
		elif self.grad_mode == 'gnorm':
			for j in range(self.descent_steps):
				ngrad = self._neGrad()
				gnorm = norm(ngrad)/((ngrad.size)**0.25)
				if gnorm < 1:
					ngrad /= gnorm
				self.merged_matrix += ngrad * self.learning_rate
				if j < self.descent_steps-1:
					normalize(self.merged_matrix)
		elif self.grad_mode == 'RMSProp' or self.grad_mode=='RMSprop':
			rho = 0.9
			delta = 1e-15
			r = np.zeros(self.merged_matrix.shape)
			for j in range(self.descent_steps):
				ngrad = self._neGrad()
				r = (1-rho)*ngrad*ngrad + rho*r
				self.merged_matrix += (ngrad/np.sqrt(delta+r))*self.learning_rate
				if j < self.descent_steps-1:
					normalize(self.merged_matrix)
		elif self.grad_mode == 'RMSProp-momentum':
			rho = 0.9
			alpha = 0.8
			delta = 1e-15
			r = np.zeros(self.merged_matrix.shape)
			v = np.zeros(self.merged_matrix.shape)
			for j in range(self.descent_steps):
				if j > 0:
					self.merged_matrix += alpha*v
					normalize(self.merged_matrix)
				ngrad = self._neGrad()
				r = (1-rho)*ngrad*ngrad + rho*r
				v = alpha*v + ngrad/np.sqrt(r)*self.learning_rate
				self.merged_matrix += v
		elif self.grad_mode == 'Adam' or self.grad_mode == 'adam':
			rho1 = 0.9
			rho2 = 0.99
			delta = 1e-15
			s = np.zeros(self.merged_matrix.shape)
			r = np.zeros(self.merged_matrix.shape)
			for j in range(1,self.descent_steps+1):
				ngrad = self._neGrad()
				s = rho1*s - (1-rho1)*ngrad
				r = rho2*r + (1-rho2)*ngrad*ngrad
				self.merged_matrix -= self.learning_rate * s/(1-rho1**j)/(delta+np.sqrt(r/(1-rho2**j)))
				if j < self.descent_steps:
					normalize(self.merged_matrix)

	def train(self, loops):
		tmp_rear = self.batch_size + self.dat_rear
		self.dat.measureUpTo(tmp_rear)
		if not self.add_mode:
			self.dat_head = self.dat_rear
		self.dat_rear = tmp_rear
		late_mats = deepcopy(self.matrices)
		self._initCumulant()
		for lp in range(loops):
			for b in range(self._n-2, 0, -1):
				self.mergeBond(b)
				self._multiSteps()
				self.rebuildBond(False)
				self._updateCumulant(False)
			# self.calcLoss()
			for b in range(0, self._n-2):
				self.mergeBond(b)
				self._multiSteps()
				self.rebuildBond(True)
				self._updateCumulant(True)
			# self.calcLoss()
		self.ibatch += 1
		self.train_history.append((self.cutoff, self.descent_steps, self.learning_rate, self.penalty_rate, self.dat_head, self.dat_rear, loops))	
		
		self.succ_fid.append(self.giveFidelity(late_mats))
		self.real_fid.append(self.giveFidelity(self.dat.getMats()))

	def save(self, stamp):
		try:
			mkdir('./'+stamp+'/')
		except:
			shutil.rmtree(stamp)
			mkdir('./'+stamp+'/')
		chdir('./'+stamp+'/')
		# fp = open('MPS.log', 'w')
		# fp.write("Present State of MPS:\n")
		# fp.write("space_size=%d\t,cutoff=%.10f\t,step_len=%f\n"% (self._n, self.cutoff,self.learning_rate)) 
		# fp.write("bond dimension:"+str(self.bondims))
		# fp.write("\tloss=%1.6e\n"%self.loss[-1])
		save('Bondim.npy',self.bondims)
		save('Mats.npy',self.matrices)
		save('ibatch.npy',self.ibatch)

		# print('Saved')
		# fp.write("cutoff\tn_descent\tstep_length\tpenalty\t(dat_h, dat_r)\tn_loop\n")
		# for history in self.train_history:
		# 	fp.write("%1.2e\t%d\t%1.2e\t%1.2e\t(%d,%d)\t%d\n"%tuple(history))
		# fp.close()
		chdir('..')
		# save('Loss.npy',self.loss)
		savez('Fidelity.npz', succ=self.succ_fid, real=self.real_fid)
		with open('TrainHistory.pickle', 'wb') as thp:
			pickle.dump(self.train_history, thp)

		# self.dat.save('dataset.npz')
	
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

		try:
			self.ibatch = np.load('ibatch.npy')
			# if self.ibatch ==
		except FileNotFoundError:
			if srch_pwd is None:
				raise ValueError("ibatch can't be found")
			else:
				self.ibatch = readLfromdir(srch_pwd)
		try:
			with open('../TrainHistory.pickle', 'rb') as thp:
				self.train_history = pickle.load(thp)[:self.ibatch+1]
		except:
			self.train_history = load('TrainHistory.npy').tolist()
			self.train_history = [(t[0],int(t[1]),t[2],None if abs(t[3])<1e-10 else t[3],\
							   int(t[4]), int(t[5]), int(t[6])) for t in self.train_history]
		try:
			self.cutoff,self.descent_steps,self.learning_rate,self.penalty_rate,\
				self.dat_head, self.dat_rear, _lp = self.train_history[-1]
		except:
			raise IndexError("len(self.train_history)=%d, ibatch=%d"%(len(self.train_history),self.ibatch))
		# self.descent_steps = int(self.descent_steps)
		# if abs(self.penalty_rate) < 1e-10:
		# 	self.penalty_rate = None
		# self.dat_head = int(self.dat_head)
		# self.dat_rear = int(self.dat_rear)

		try:
			fids = np.load('../Fidelity.npz')
		except:
			fids = np.load('Fidelity.npz')
		self.succ_fid = fids['succ'].tolist()[:self.ibatch+1]
		self.real_fid = fids['real'].tolist()[:self.ibatch+1]
		self.matrices = load('Mats.npy').tolist()
		
		self.merged_bond = None
		self.merged_matrix=None
		if srch_pwd is not None:
			os.chdir(oripwd)

def readLfromdir(srch_pwd):
	if srch_pwd[-1]=='/' or srch_pwd[-1]=='_':
		srch_pwd = srch_pwd[:-1]
	k = -1
	while k > -len(srch_pwd):
		k -= 1
		if srch_pwd[k] == 'L':
			k += 1
			break
	k1 = k
	while k1 <= -1:
		if srch_pwd[k1] == 'R':
			break
		k1 += 1
	if k1==0:
		return int(srch_pwd[k:])
	else:
		return int(srch_pwd[k:k1])

def preparation(typ, nn, tot):
	sm = MPS(nn,typ)
	ds = ProjMeasureSet(nn, tot, mps=sm)
	return ds

if __name__ == '__main__':
	pass
	# typ = sys.argv[1]
	# space_size = int(sys.argv[2])
	# sm = MPS(space_size, typ)
	# comm = MPI.COMM_WORLD
	# rk = comm.Get_rank()
	# seed(rk)
	
	# mxBatch = 200
	# batch_size = 40
	# ds = preparation(typ, space_size, mxBatch*batch_size)

	# measout = measout_dir+'/%s/%d/'%(typ, space_size)
	# try:
	# 	mkdir(measout)
	# except FileExistsError:
	# 	pass
	# ds.save(measout+"/R%dSet"%(rk))

	# fload = open("N%dL%dR%dSet"%(space_size,mxBatch-1,rk)+'.pickle','rb')
	# ds1 = pickle.load(fload)
	# fload.close()
	# # sm1 = MPS(20,'W')
	# # ds1 = ProjMeasureSet(20,0,mps=sm1)
	# # ds1.load("N%dL%dR%dSet"%(space_size,mxBatch-1,rk))
	# print('#%d'%rk, ds1.getN())


	# dat = ProjMeasureSet(space_size, init_set_size=80, mode='uniform', mps = sm)

	# m = TomoTrainer(dat)
	# m.add_mode = True
	# m.grad_mode = argv[1]
	# m.batch_size = 40
	# m.cutoff = 0.8
	# m.descent_steps = 10
	# m.learning_rate = float(argv[2])
	# m.penalty_rate = None
	
	# mxBatch = 160
	# for b in range(mxBatch):
	# 	m.train(nloop)
	# 	if b%10==9:
	# 		stmp = 'L%d'%b
	# 		m.save(stmp)

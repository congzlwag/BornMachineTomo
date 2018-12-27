# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:37:55 2017

@author: lenovo
"""

## todo list fidelity basis generate

from numpy import *
import numpy as np
from numpy.random import rand, seed, randn
from numpy.linalg import norm,eig,solve,svd
from copy import deepcopy
from os import mkdir,chdir, listdir
import shutil
import pickle
import os

# from mpi4py import MPI
from sys import argv

def normalize(tens):
	tens /= norm(tens)

class MPS:
	"""
	Base MPS Class
	Parameter:
		initial_state_type:
			"GHZ"/"W"/"dimer"/"cluster" and you will obtain the designated MPS state.
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
		
		self.id_uncanonical = -1 
		"""pointer to matrix
		-1: neither right-canon nor left-canon, need canonicalization
		in range(space_size-1): self.matrix[id_uncanonical] is the only one not being cannonical
		space_size-1: left-canonicalized
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
		return np.abs(self.calcPsi(state, spinor_setting)) ** 2.
	
	def mergeBond(self, bond):
		self.merged_bond = bond
		self.merged_matrix = tensordot(self.matrices[bond],self.matrices[bond + 1], ([2], [0]))
		
	def rebuildBond(self, going_right, keep_bondim=False):
		U, s, V = svd(reshape(self.merged_matrix, (self.bondims[(self.merged_bond - 1)] * 2, 2 * self.bondims[(self.merged_bond + 1)])))
		# print("bond:", bond, s)
		minibond = 2
		bdm = minibond
		if keep_bondim:
			bdm = self.bondims[self.merged_bond]
		else:
			while bdm < s.size and s[bdm] >= s[0] * self.cutoff:
				bdm += 1
		s = np.diag(s[:bdm])
		U = U[:, :bdm]
		V = V[:bdm, :]
		if going_right:
			V = dot(s, V)
			normalize(V)
			self.id_uncanonical = (self.merged_bond + 1)
		else:
			U = dot(U, s)
			normalize(U)
			self.id_uncanonical = self.merged_bond
		
		if not keep_bondim:
			# print("Bondim #%d: %d -> %d", self.bondims[self.merged_bond], bdm)
			self.bondims[self.merged_bond] = bdm
		self.matrices[self.merged_bond] = reshape(U, (self.bondims[(self.merged_bond - 1) % self._n], 2, bdm))
		self.matrices[(self.merged_bond + 1)] = reshape(V, (bdm, 2, self.bondims[(self.merged_bond + 1)]))
		self.merged_bond = None
		self.merged_matrix=None
		return np.diag(s)
		
	def leftCano(self):
		if self.merged_bond is not None:
			self.rebuildBond(True)
		for bond in range(max(self.id_uncanonical,0), self._n - 1):
			self.mergeBond(bond)
			self.rebuildBond(True, keep_bondim=True)

	def genSample(self,spinor_setting):
		v = ones((1,),dtype=complex)
		restate = 0
		pointer = self._n - 1
		while pointer >= 0:
			restate *= 2
			nv = [dot(dot(self.matrices[pointer],v),spinor_setting[pointer,i]) for i in range(2)]
			p = [norm(nv[i]) ** 2 for i in range(2)]
			if rand() < p[1] / (p[0] + p[1]):
				restate += 1
				v = nv[1]
			else:
				v = nv[0]
			pointer -= 1
			normalize(v)
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
	Available measuring modes: uniform/2n+1/onlyZ/dynamic
	"""
	def __init__(self, space_size, init_set_size=0, mode='uniform', **kwarg):
		self.__n = space_size
		self.__mps = None
		self.states = []
		self.spinor_settings = []
		self.setMode(mode)

		if 'mps' in kwarg:
			self.designateMPS(kwarg['mps'])
		elif 'MPS' in kwarg:
			self.designateMPS(kwarg['MPS'])
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

	def setMode(self, mod):
		if self.states == []:
			self.__mode = mod
			if mod=='uniform':
				self.__singlemeas = self.uniforMeas
			elif mod=='onlyZ':
				self.__singlemeas = self.zzMeas
			# else:
			# 	self.__singlemeas = None
		else:
			print('Error: You cannot set another mode because there are measured data.')
	
	def uniforMeas(self):
		setting = empty((self.__n, 2,2),dtype=complex)
		c = rand(self.__n)*2-1
		c1 = (0.5*(1+c))**0.5 #cos(theta/2)
		s1 = (0.5*(1-c))**0.5 #sin(theta/2)
		phi = rand(self.__n) * 2 * pi
		phas= exp(1.0j*phi)
		setting[:,0,0] = c1
		setting[:,0,1] = s1*phas
		setting[:,1,0] = -s1*phas.conj()
		setting[:,1,1] = c1
		state = self.__mps.genSample(setting)
		return setting, state

	def zzMeas(self):
		setting = empty((self.__n, 2,2),dtype=int8)
		setting[:,0,0] = 1
		setting[:,1,1] = 1
		setting[:,1,0] = 0
		setting[:,0,1] = 0
		state = self.__mps.genSample(setting)
		return setting, state
		
	def measureUpTo(self, size):
		for _ in range(size-len(self.states)):
			setting, state = self.__singlemeas()
			self.spinor_settings.append(setting)
			self.states.append(state)

	def getMats(self):
		return deepcopy(self.__mps.matrices)

	def __getstate__(self):
		""" Retuen a dictionary of state values to be pickled """
		exclusion=["spinor_settings",'states']
		mydict={}
		for key in self.__dict__.keys():
			if key not in exclusion:
				mydict[key] = self.__dict__[key]
		return mydict

	def save(self, name):
		if name[-4:] != '.npz':
			name = name + '.npz'
		np.savez_compressed(name, spin=asarray(self.spinor_settings), stat=asarray(self.states), mode=self.__mode)
		with open(name[:-4]+".pickle","wb") as fsav:
			pickle.dump(self,fsav)

def loadProjSet(nam):
	fload = open(nam+'.pickle','rb')
	ds = pickle.load(fload)
	fload.close()
	info = load(nam+'.npz')
	print(ds.__dict__.keys())
	ds.spinor_settings = info['spin'].tolist()
	ds.states = info['stat'].tolist()
	return ds

class TomoTrainer(MPS):
	"""Trainer"""
	def __init__(self, dataset, add_mode=True):
		MPS.__init__(self, dataset.getN(), 'random')
		self.leftCano()
		self.loss = []
		self.succ_fid = []
		self.real_fid = []
		self.train_history = []
		self.add_mode = add_mode
		self.batch_size = 40
		self.dat = dataset
		self.dat_head = 0
		self.dat_rear = len(dataset.states)

		self.grad_mode = 'plain' # plain/gnorm/RMSProp/Adam
		self.learning_rate = 0.01
		self.descent_steps = 5
		self.penalty_rate = None

		self.loss.append(self.calcLoss(dataset))

	def _showPsi(self, istate):
		"""Evaluate with cumulant"""
		state = self.dat.states[istate]
		spin = self.dat.spinor_settings[istate]
		istate -= self.dat_head
		lvec = self._cumulantL[-1][istate]
		rvec = self._cumulantR[-1][istate]
		k = self.merged_bond
		if k:
			tmp = dot(self.merged_matrix, rvec)
			tmp = dot(tmp, spin[k+1,(state>>(k+1))%2])
			tmp = dot(tmp, spin[k, (state>>k)%2])
		else:
			k = len(self._cumulantL)-1
			tmp = dot(dot(self.matrices[k+1], rvec), spin[k+1,(state>>(k+1))%2])
			tmp = dot(dot(self.matrices[k], tmp), spin[k,(state>>(k))%2])
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
		if not dataset:
			dataset = self.dat
		if rear == -1:
			rear = len(dataset.states)
		if dataset:
			res = -2*mean([log(self.calcProb(dataset.states[i], dataset.spinor_settings[i])) for i in range(head, rear)])
		else:
			res = sum([log(np.abs(self._showPsi(i))) for i in range(max(self.dat_head,head), min(rear,self.dat_rear))])
			for i in range(head, max(self.dat_head,head)):
				res += log(self.calcProb(dataset.states[i], dataset.spinor_settings[i]))
			for i in range(min(rear,self.dat_rear), rear):
				res += log(self.calcProb(dataset.states[i], dataset.spinor_settings[i]))
			res /= (rear-head)
		print('Loss =',res)
		return res

	def _initCumulant(self):
		"""
		Initialize self.cumulants for a batch. Call it right before merging bond self._n-2
		During the training process, it will be kept unchanged that:
		1) len(cumulant)== space_size
		2) cumulant[0]  == ones((n_sample, 1))
		3) cumulant[-1] == ones((1, n_sample))
		4)  k = current_bond
			cumulant[j] = 	if 0<j<=k: A(0)...A(j-1)
							elif k<j<space_size-1: A(j+1)...A(space_size-1)
		"""
		if self.id_uncanonical == self._n-1:
			self.leftCano()
		self._cumulantL = [ones((self.dat_rear-self.dat_head, 1))]
		for n in range(0, self._n-2):
			tmp = []
			for k in range(self.dat_head, self.dat_rear):
				mid = tensordot(self._cumulantL[-1][k-self.dat_head], self.matrices[n], ([0],[0]))
				mid = dot(self.dat.spinor_settings[k][n,(self.dat.states[k]>>n)%2], mid)
				tmp.append(mid)
			self._cumulantL.append(asarray(tmp))
		self._cumulantR = [ones((self.dat_rear-self.dat_head, 1))]

	def _updateCumulant(self, going_right):
		k = len(self._cumulantL)-1
		if going_right:
			tmp = []
			for i in range(self.dat_head, self.dat_rear):
				mid = tensordot(self._cumulantL[-1][i-self.dat_head], self.matrices[k], ([0],[0]))
				mid = dot(self.dat.spinor_settings[i][k,(self.dat.states[i]>>k)%2], mid)
				tmp.append(mid)
			self._cumulantL.append(asarray(tmp))
			self._cumulantR.pop()
		else:
			k += 1
			tmp = []
			for i in range(self.dat_head, self.dat_rear):
				mid = tensordot(self._cumulantR[-1][i-self.dat_head], self.matrices[k], ([0],[2]))
				mid = dot(mid, self.dat.spinor_settings[i][k,(self.dat.states[i]>>k)%2])
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
			spin = self.dat.spinor_settings[istate]
			state= self.dat.states[istate]
			psi = self._showPsi(istate)
			conjgrad += (self._cumulantL[-1][istate-self.dat_head].reshape(-1,1,1,1))\
						*(spin[k,(state>>k)%2].reshape(1,-1,1,1))\
						*(spin[k+1,(state>>(k+1))%2].reshape(1,1,-1,1))\
						*(self._cumulantR[-1][istate-self.dat_head].reshape(1,1,1,-1))\
						/psi
		conjgrad /= (self.dat_rear - self.dat_head)
		conjgrad -= conj(self.merged_matrix) 
		
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
				# for _ in range(self.descent_steps):
				# 	self._graDescent()
				self._multiSteps()
				self.rebuildBond(False)
				self._updateCumulant(False)
			self.calcLoss()
			for b in range(0, self._n-2):
				self.mergeBond(b)
				# for _ in range(self.descent_steps):
				# 	self._graDescent()
				self._multiSteps()
				self.rebuildBond(True)
				self._updateCumulant(True)
			self.calcLoss()
		self.train_history.append((self.cutoff, self.descent_steps, self.learning_rate, self.penalty_rate if self.penalty_rate else 0, self.dat_head, self.dat_rear, loops))	
		
		self.succ_fid.append(self.giveFidelity(late_mats))
		self.real_fid.append(self.giveFidelity(self.dat.getMats()))

	def save(self, stamp):
		try:
			mkdir('./'+stamp+'/')
		except:
			shutil.rmtree(stamp)
			mkdir('./'+stamp+'/')
		chdir('./'+stamp+'/')
		fp = open('MPS.log', 'w')
		fp.write("Present State of MPS:\n")
		fp.write("space_size=%d\t,cutoff=%.10f\t,step_len=%f\n"% (self._n, self.cutoff,self.learning_rate)) 
		fp.write("bond dimension:"+str(self.bondims))
		fp.write("\tloss=%1.6e\n"%self.loss[-1])
		save('Loss.npy',self.loss)
		save('Bondim.npy',self.bondims)
		save('TrainHistory.npy',self.train_history)
		save('Mats.npy',self.matrices)
		print('Saved')
		fp.write("cutoff\tn_descent\tstep_length\tpenalty\t(dat_h, dat_r)\tn_loop\n")
		for history in self.train_history:
			fp.write("%1.2e\t%d\t%1.2e\t%1.2e\t(%d,%d)\t%d\n"%tuple(history))
		fp.close()
		savez('Fidelity.npz', succ=self.succ_fid, real=self.real_fid)
		chdir('..')
		self.dat.save('dataset.npz')
	
	def load(self, srch_pwd=None):
		if srch_pwd is not None:
			oripwd = os.getcwd()
			os.chdir(srch_pwd)
		self.bondims = load('Bondim.npy').tolist()
		self.__n = len(self.bondims)
		self.trainhistory = load('TrainHistory.npy').tolist()
		self.Loss = load('Loss.npy').tolist()
		self.cutoff = self.trainhistory[-1][0]
		self.descent_steps = self.trainhistory[-1][1]
		self.learning_rate = self.trainhistory[-1][2]
		self.penalty_rate = self.trainhistory[-1][3]

		self.matrices = load('Mats.npy').tolist()
		
		self.merged_bond = None
		self.merged_matrix=None
		if srch_pwd is not None:
			os.chdir(oripwd)

def preparation(typ, nn, tot):
	sm = MPS(nn,typ)
	ds = ProjMeasureSet(nn, tot, mps=sm)
	return ds

if __name__ == '__main__':
	typ = 'cluster'
	space_size = 40
	# comm = MPI.COMM_WORLD
	# rk = comm.Get_rank()
	# seed(rk)
	rk = 0
	mxBatch = 200
	# ds = preparation(typ, space_size, mxBatch*40+80)
	ds = preparation(typ, space_size, 1)
	ds.save('test')
	# ds.save("N%dL%dR%dSet"%(space_size,mxBatch-1,rk))

	# fload = open("N%dL%dR%dSet"%(space_size,mxBatch-1,rk)+'.pickle','rb')
	# ds1 = pickle.load(fload)
	# fload.close()
	# # sm1 = MPS(20,'W')
	# # ds1 = ProjMeasureSet(20,0,mps=sm1)
	# # ds1.load("N%dL%dR%dSet"%(space_size,mxBatch-1,rk))
	# print('#%d'%rk, ds1.getN())
	# print(dir(ds))
	ds1 = loadProjSet('test')
	print(ds1.getN(),ds1.states)

	# typ = 'GHZ'
	# space_size = 20
	# sm = MPS(space_size, typ)
	# comm = MPI.COMM_WORLD
	# rk = comm.Get_rank()
	# sed = rk
	# seed(sed)
	# nloop = int(argv[3])
	# if rk ==0:
	# 	with open('hyper.txt','w') as fout:
	# 		print('grad mode:',argv[1], '\nlr:',argv[2], '\nnloop:',argv[3], file=fout)
	# os.mkdir('R%d'%sed)
	# os.chdir('R%d'%sed)

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

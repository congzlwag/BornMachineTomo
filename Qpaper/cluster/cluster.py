# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:37:55 2017

@author: lenovo
"""

## todo list fidelity basis generate

from numpy import zeros,ones,exp,sort,conj,array,argsort,log,dot,tensordot,outer,reshape,swapaxes,angle,save,load
import numpy as np
#import matplotlib.pyplot as plt
from numpy.random import rand
from numpy.linalg import norm,eig,solve,svd
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from time import strftime
from os import mkdir,chdir
import shutil

def mylog(x):
	if x <= 0:
		return - 1000000
	else:
		return log(x)
	
spin_eigens = []
spin_eigens.append(array([[1., 0],[0, 1.]]))
spin_eigens.append(array([[1., -1.],[1., 1.]]) / 2 ** 0.5)
spin_eigens.append(array([[1., -1.j],[-1.j, 1]]) / 2 ** 0.5)
#012 : zxy
		

class training_set(object):
	def __init__(self, space_size, distri, **kwarg):
		
		#training_set(10, "MPSreal", mps = m)
		#training_set(10, "Hreal", HamiltonianParameter = [...])
		#training_set(10, "MPSsampled", batch_size = 100, initial_batch_number = 10, mps = m)
		#training_set(10, "Hsampled", batch_size = 100, initial_batch_number = 10, HamiltonianParameter = [...])
		
		self.space_size = space_size
		self.state_number = 2 ** space_size
		self.Self_Likelihood = []
		self.pow = array([2 ** i for i in range(space_size)])
		self.distri = distri

		self.direction_settings = []
		
		self.setting_number = 0
		for i in range(0, 1):
			self.direction_settings.append(int(i) * ones(self.space_size, dtype = 'int'))
			self.setting_number += 1
			for j in range(self.space_size):
				self.direction_settings.append(int(i) * ones(self.space_size, dtype = 'int'))
				self.direction_settings[-1][j] = int((i + 1) % 3)
				self.direction_settings.append(int(i) * ones(self.space_size, dtype = 'int'))
				self.direction_settings[-1][j] = int((i + 2) % 3)
				self.setting_number += 2
				
		
		if distri == "MPSreal":
			self.indstate = [self.ind2state(i) for i in range(self.state_number)]
			self.mps = kwarg['mps']
			self.real_probab = []
			self.Self_Likelihoods = zeros(self.setting_number)
			for ind_setting in range(self.setting_number):
				self.real_probab.append(zeros(self.state_number))
				for istate in range(self.state_number):
					self.real_probab[ind_setting][istate] = self.mps.Give_probab(self.indstate[istate],self.direction_settings[ind_setting])
					self.Self_Likelihoods[ind_setting] -= self.real_probab[ind_setting][istate] * mylog(self.real_probab[ind_setting][istate])
			self.Self_Likelihood = np.sum(self.Self_Likelihoods)
		
		elif distri == "MPSsampled":
			self.indstate = {}
			self.mps = kwarg['mps']
			self.batch_size = kwarg['batch_size']
			self.sampled_probab = []
			self.count = []
			
			self.Self_Likelihoods = []
			self.set_size = 0
			
			for ind_setting in range(self.setting_number):
				self.sampled_probab.append({})
				self.count.append({})
				self.Self_Likelihoods.append(0.)
			
			
			for i in range(kwarg['initial_batch_number']):
				self.add_batch()
			
			

		# print(self.Self_Likelihood)
		# print(self.Self_Likelihoods.tolist())
	
	def state2ind(self, state):
		return int(np.sum(state * self.pow))
	
	def ind2state(self, ind):
		state = zeros(self.space_size, dtype = 'int')
		for i in range(self.space_size):
			state[i] = int(ind % 2)
			ind //= 2		
		return state
	
	def add_batch(self):
		self.set_size += self.batch_size
		
		for ind_setting in range(self.setting_number):
			for s in range(self.batch_size):
				state = self.mps.Give_Sample(self.direction_settings[ind_setting])
				istate = self.state2ind(state)
				if istate not in self.indstate:
					self.indstate[istate] = state
				
				if istate in self.count[ind_setting]:
					self.count[ind_setting][istate] += 1.
				else:
					self.count[ind_setting][istate] = 1.
			
			self.Self_Likelihoods[ind_setting] = 0.
			for istate in self.count[ind_setting]:
				self.sampled_probab[ind_setting][istate] = self.count[ind_setting][istate] / self.set_size
				self.Self_Likelihoods[ind_setting] -= log(self.sampled_probab[ind_setting][istate]) * self.sampled_probab[ind_setting][istate]
			
		self.Self_Likelihood = np.sum(self.Self_Likelihoods)

class MPS(object):
	def __init__(self, space_size, initial_state_type):
		self.space_size = space_size
		self.state_number = 2 ** space_size
		self.cutoff = 0.01
		self.descenting_step_length = 0.01
		
		
		if initial_state_type == "random":
			init_bond_dimension = 2
			self.bond_dimension = [init_bond_dimension] * self.space_size ## bond[i] connect i i+1
			self.bond_dimension[-1] = 1
			self.matrices = []
			for i in range(space_size):
				self.matrices.append(rand(self.bond_dimension[(i - 1) % self.space_size], 2, self.bond_dimension[i]) + rand(self.bond_dimension[(i - 1) % self.space_size], 2, self.bond_dimension[i]) * 1.j)
		elif initial_state_type == "GHZ":
			init_bond_dimension = 2
			self.bond_dimension = [init_bond_dimension] * self.space_size ## bond[i] connect i i+1
			self.bond_dimension[-1] = 1
			self.matrices = []
			eiphi = exp(0.1j)
			l = zeros((1,2,2)) + 0.j
			l[0,0,0] = eiphi
			l[0,1,1] = 1.
			r = zeros((2,2,1)) + 0.j
			r[0,0,0] = eiphi
			r[1,1,0] = 1.
			bulk = zeros((2,2,2)) + 0.j
			bulk[0,0,0] = eiphi
			bulk[1,1,1] = 1.
			self.matrices.append(l.copy())
			for i in range(space_size - 2):
				self.matrices.append(bulk.copy())
			self.matrices.append(r.copy())
		elif initial_state_type == "dimer":
			init_bond_dimension = 3
			self.bond_dimension = [init_bond_dimension] * self.space_size ## bond[i] connect i i+1
			self.bond_dimension[-1] = 1
			self.matrices = []
			
			l = zeros((1,2,3)) + 0.j
			l[0,1,1] = 1
			l[0,0,2] = 1.
			r = zeros((3,2,1)) + 0.j
			r[1,0,0] = -1.
			r[2,1,0] = 1.
			bulk = zeros((3,2,3)) + 0.j
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
			self.bond_dimension = [init_bond_dimension] * self.space_size ## bond[i] connect i i+1
			self.bond_dimension[-1] = 1
			self.matrices = []
		
			l = zeros((1,2,2)) + 0.j
			l[0,0,0] = 1
			l[0,1,1] = 1.
			r = zeros((2,2,1)) + 0.j
			r[1,0,0] = 1.
			r[0,1,0] = exp(0.1j * (self.space_size - 1))
			bulk = zeros((2,2,2)) + 0.j
			bulk[0,0,0] = 1.
			bulk[1,0,1] = 1.
			
			self.matrices.append(l.copy())
			for i in range(1, space_size - 1):
				bulk[0,1,1] = exp(0.1j * i)
				self.matrices.append(bulk.copy())
			self.matrices.append(r.copy())
		elif initial_state_type == "Cluster":
			init_bond_dimension = 4
			self.bond_dimension = [init_bond_dimension] * self.space_size ## bond[i] connect i i+1
			self.bond_dimension[-1] = 1
			self.bond_dimension[0] = 2
			self.bond_dimension[-2] = 2
			
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

			self.bond_dimension = [4] * self.space_size
			self.bond_dimension[0] = 2
			self.bond_dimension[-1] = 1
			self.bond_dimension[-2] = 2

			S_x = np.zeros((2,2))
			S_x[0][1] = 1
			S_x[1][0] = 1

			S_y = np.array([[0, -1.j], [1.j, 0]])

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
			N = self.space_size
			for i in range(N - 4):
				self.matrices.append(bulk)
			self.matrices.append(righttwo)
			self.matrices.append(rightone)
		
		self.merged_bond = -1
		self.merged_matrix = []
		
		self.normalization_pointer = 0
		self.left_cano()
		self.Loss = []
		self.Losses = []
		self.trainhistory = []
		
	def ind2state(self, ind):
		state = zeros(self.space_size, dtype = 'int')
		for i in range(self.space_size):
			state[i] = int(ind % 2)
			ind //= 2		
		return state
	
	def Give_Psi(self, state, direction_setting):
		if self.merged_bond == -1:
			left_bond = self.space_size - 1
			right_bond = 0
			current_matrix = tensordot(self.matrices[0], spin_eigens[direction_setting[0]][state[0]], ([1],[0]))
		else :
			left_bond = (self.merged_bond - 1) % self.space_size
			right_bond = (self.merged_bond + 1) % self.space_size
			direction1 = direction_setting[self.merged_bond]
			direction2 = direction_setting[(self.merged_bond + 1) % self.space_size]
			current_matrix = tensordot(self.merged_matrix, spin_eigens[direction1][state[self.merged_bond]], ([1],[0]))
			current_matrix = tensordot(current_matrix, spin_eigens[direction2][state[(self.merged_bond + 1) % self.space_size]], ([1],[0]))
		while right_bond != left_bond:
			next_bond = (right_bond + 1) % self.space_size
			next_direction = direction_setting[next_bond]
			next_matrix = tensordot(self.matrices[next_bond], spin_eigens[next_direction][state[next_bond]], ([1],[0]))
			current_matrix = dot(current_matrix, next_matrix)
			right_bond = next_bond
		
		p = np.sum(np.diag(current_matrix))
		return p

	def Give_probab(self, state, direction_setting): 
		return np.abs(self.Give_Psi(state, direction_setting)) ** 2.
		
	def renormalize(self):
		if self.merged_bond == -1:
			nor = norm(self.matrices[self.normalization_pointer][:])
			self.matrices[self.normalization_pointer] /= nor
		else :
			nor = norm(self.merged_matrix)
			self.merged_matrix /= nor
		#print("ReNormalization Factor:", nor)
		return nor

	def Show_Loss(self, dataset):
		Losses = zeros(dataset.setting_number)
		if dataset.distri == "MPSreal" or dataset.distri == "Hreal":
			for ind_setting in range(dataset.setting_number):
				for istate in range(dataset.state_number):
					state = dataset.indstate[istate]
					Losses[ind_setting] -= dataset.real_probab[ind_setting][istate] * log(self.Give_probab(state, dataset.direction_settings[ind_setting]))
		else:
			for ind_setting in range(dataset.setting_number):
				for istate in dataset.sampled_probab[ind_setting]:
					state = dataset.indstate[istate]
					Losses[ind_setting] -= dataset.sampled_probab[ind_setting][istate] * mylog(self.Give_probab(state, dataset.direction_settings[ind_setting]))
		Losses -= dataset.Self_Likelihoods
		Loss = np.sum(Losses)
		self.Loss.append(Loss)
		self.Losses.append(Losses)
		# print("Current loss:", Losses.tolist())
		print("Current loss:", Loss)
		return Loss
	
	def merge_bond(self, bond):
		self.merged_bond = bond
		self.merged_matrix = tensordot(self.matrices[bond],self.matrices[(bond + 1) % self.space_size], ([2], [0]))
		
	def rebuild_bond(self, bond, sweeping_direction_is_right):
		self.merged_bond = -1
		#cutoff = 0.09 - 0.40 * min(self.Loss[-1], 0.2)
		U, s, V = svd(reshape(self.merged_matrix, (self.bond_dimension[(bond - 1) % self.space_size] * 2, 2 * self.bond_dimension[(bond + 1) % self.space_size])))
		
		print("bond:", bond, s)
		
		current_bond_dimension = s.size
		for i in range(s.size):
			if s[i] < s[0] * self.cutoff:
				current_bond_dimension = i
				break
		current_bond_dimension = max(current_bond_dimension, 1)	
		s = np.diag(s[:current_bond_dimension])
		
		if sweeping_direction_is_right:
			U = U[:,:current_bond_dimension]
			V = dot(s, V[:current_bond_dimension,:])
			self.normalization_pointer = (bond + 1) % self.space_size
		else:
			U = dot(U[:,:current_bond_dimension], s)
			V = V[:current_bond_dimension,:]
			self.normalization_pointer = bond
		
		self.bond_dimension[bond] = current_bond_dimension
		print("bond dimensions:", self.bond_dimension)
		self.matrices[bond] = reshape(U, (self.bond_dimension[(bond - 1) % self.space_size], 2, self.bond_dimension[bond]))
		self.matrices[(bond + 1) % self.space_size] = reshape(V, (self.bond_dimension[bond], 2, self.bond_dimension[(bond + 1) % self.space_size]))
		
		self.renormalize()
		return np.diag(s)
		
	def Gradient_descent(self, dataset):
		gradient = zeros((self.bond_dimension[(self.merged_bond - 1) % self.space_size], 2, 2, self.bond_dimension[(self.merged_bond + 1) % self.space_size])) + 0.j
		
		if dataset.distri == "MPSreal" or dataset.distri == "Hreal":
			for ind_setting in range(dataset.setting_number):
				direction_setting = dataset.direction_settings[ind_setting]
				for istate in range(dataset.state_number):
					state = dataset.indstate[istate]
				
					psi = self.Give_Psi(state, dataset.direction_settings[ind_setting])
					proba = np.abs(psi) ** 2
					
					left_bond = (self.merged_bond - 1) % self.space_size
					right_bond = (self.merged_bond + 2) % self.space_size
					right_direction = direction_setting[right_bond]
					current_matrix = tensordot(self.matrices[right_bond], spin_eigens[right_direction][state[right_bond]], ([1],[0]))
					
					while right_bond != left_bond:
						next_bond = (right_bond + 1) % self.space_size
						next_direction = direction_setting[next_bond]
						next_matrix = tensordot(self.matrices[next_bond], spin_eigens[next_direction][state[next_bond]], ([1],[0]))
						current_matrix = dot(current_matrix, next_matrix)
						right_bond = next_bond
					
					direction1 = direction_setting[self.merged_bond]
					direction2 = direction_setting[(self.merged_bond + 1) % self.space_size]
					for w1 in range(2):
						for w2 in range(2):
							gradient[:, w1, w2, :] +=  dataset.real_probab[ind_setting][istate] * current_matrix.T *  conj(psi) / proba * spin_eigens[direction1][state[self.merged_bond]][w1] * spin_eigens[direction2][state[(self.merged_bond + 1) % self.space_size]][w2]
		else:
			for ind_setting in range(dataset.setting_number):
				direction_setting = dataset.direction_settings[ind_setting]
				for istate in dataset.sampled_probab[ind_setting]:
					state = dataset.indstate[istate]
					
					psi = self.Give_Psi(state, dataset.direction_settings[ind_setting])
					proba = np.abs(psi) ** 2
					
					left_bond = (self.merged_bond - 1) % self.space_size
					right_bond = (self.merged_bond + 2) % self.space_size
					right_direction = direction_setting[right_bond]
					current_matrix = tensordot(self.matrices[right_bond], spin_eigens[right_direction][state[right_bond]], ([1],[0]))
					
					while right_bond != left_bond:
						next_bond = (right_bond + 1) % self.space_size
						next_direction = direction_setting[next_bond]
						next_matrix = tensordot(self.matrices[next_bond], spin_eigens[next_direction][state[next_bond]], ([1],[0]))
						current_matrix = dot(current_matrix, next_matrix)
						right_bond = next_bond
					
					direction1 = direction_setting[self.merged_bond]
					direction2 = direction_setting[(self.merged_bond + 1) % self.space_size]
					for w1 in range(2):
						for w2 in range(2):
							gradient[:, w1, w2, :] +=  dataset.sampled_probab[ind_setting][istate] * current_matrix.T *	 conj(psi) / proba * spin_eigens[direction1][state[self.merged_bond]][w1] * spin_eigens[direction2][state[(self.merged_bond + 1) % self.space_size]][w2]
		
		gradient /= dataset.setting_number
		gradient -= conj(self.merged_matrix) 
		
		self.merged_matrix += conj(gradient) * self.descenting_step_length
		
		self.renormalize()
		#self.Show_Loss(dataset)
	
	def Give_Sample(self, direction_setting):
		v = array([1])
		pointer = self.space_size - 1
		state = []
		while pointer >= 0:
			direction = direction_setting[pointer]
			nv = [dot(tensordot(self.matrices[pointer],spin_eigens[direction][i,:], ([1],[0])),v) for i in range(2)]
			p = [norm(nv[i]) ** 2 for i in range(2)]
			if rand() < p[0] / (p[0] + p[1]):
				state = [0] + state
				v = nv[0]
			else:
				state = [1] + state
				v = nv[1]
			pointer -= 1
		return array(state)
		
	def Give_fidelity(self, m):
		left_bond = self.space_size - 1
		right_bond = 0
		current_matrix = tensordot(self.matrices[0], conj(m.matrices[0]), ([1],[1]))
		current_matrix = swapaxes(current_matrix, 1, 2)
		
		while right_bond != left_bond:
			next_bond = (right_bond + 1) % self.space_size
			next_matrix = tensordot(self.matrices[next_bond], conj(m.matrices[next_bond]), ([1],[1]))
			current_matrix = tensordot(current_matrix, next_matrix, ([2,3], [0,2]))
			right_bond = next_bond
		
		return np.abs(np.sum(current_matrix, axis = None)) ** (1. / self.space_size)
	
	def Give_Entanglement(self, cut_bond):
		
		for bond in range(self.space_size - 2, cut_bond - 1, -1):
			self.merge_bond(bond)
			Entanglement_Spectrum = self.rebuild_bond(bond, False)
		for bond in range(cut_bond, self.space_size - 1):
			self.merge_bond(bond)
			self.rebuild_bond(bond, True)
		
		Entanglement_Spectrum = sort(Entanglement_Spectrum)
		Entanglement_Entropy = - np.sum(Entanglement_Spectrum * mylog(Entanglement_Spectrum))
		
		return (Entanglement_Entropy, Entanglement_Spectrum.tolist())


		
	def Give_Correlation_length(self, site, operator):
		###assume left canonical 
		if operator == "z":
			op = array([[1,0],[0,-1]])
		E = tensordot(self.matrices[site], conj(self.matrices[site]), ([1], [1]))
		E = swapaxes(E, 1, 2)
		E = reshape(E, (self.bond_dimension[(site - 1) % self.space_size] ** 2, self.bond_dimension[site] ** 2))
		w, v = eig(E)
		ind = argsort(np.abs(w))
		
		v = reshape(v,(self.bond_dimension[site],self.bond_dimension[site],-1))
		
		rho = v[:,:,ind[-1]]
		
		Mz = tensordot(self.matrices[site], op, ([1], [0]))
		Ezl = tensordot(Mz, conj(self.matrices[site]), ([0,2], [0,1]))
		Ezr = tensordot(Mz, conj(self.matrices[site]), ([2], [1]))
		Ezr = tensordot(Ezr, rho, ([1,3], [0,1]))
		
		Aml = tensordot(Ezl, v, ([0,1], [0,1]))
		
		Ezr = reshape(Ezr, (self.bond_dimension[site]**2))
		v = reshape(v,(self.bond_dimension[site]**2,-1))
		Amr = solve(v, Ezr)
		
		Am = Aml * Amr
		
		Am = Am[ind[:]]
		w = w[ind[:]]
		
		print(Am.tolist(),w.tolist())
		
		xi = - 1 / log(sort(np.abs(w)))
		return xi
	
	def left_cano(self):
		for bond in range(2 - self.space_size, self.space_size - 1):
			self.merge_bond(np.abs(bond))
			self.rebuild_bond(np.abs(bond), (bond>=0))
		
		
	
	def train(self, dataset, Loops, descent_steps):
		#if len(self.Loss) == 0:
		self.Show_Loss(dataset)		
		
		for loop in range(Loops):
			for bond in range(2 - self.space_size, self.space_size - 2):
				self.merge_bond(np.abs(bond))
				for steps in range(descent_steps):
					self.Gradient_descent(dataset)
				self.rebuild_bond(np.abs(bond), (bond>=0))
				self.Show_Loss(dataset)
		
		self.trainhistory.append([self.cutoff,Loops,descent_steps,self.descenting_step_length])		
				
	def saveMPS(self, succ_loop):
		stamp = 'N%dL%d' % (self.space_size, succ_loop)
		try:
			mkdir('./'+stamp+'/')
		except:
			shutil.rmtree(stamp)
			mkdir('./'+stamp+'/')
		chdir('./'+stamp+'/')
		fp = open('MPS.log', 'w')
		fp.write("Present State of MPS:\n")
		fp.write("space_size=%d\t,cutoff=%.10f\t,lr=%f\n"% (self.space_size, self.cutoff,self.descenting_step_length)) 
		fp.write("bond dimension:"+str(self.bond_dimension))
		fp.write("\tloss=%1.6e\n"%self.Loss[-1])
		save('Loss.npy',array(self.Loss))
		save('Bondim.npy',array(self.bond_dimension))
		save('TrainHistory.npy',array(self.trainhistory))
		for i in range(len(self.matrices)):
			save('Mat_%d.npy'%i,self.matrices[i])
		fp.write("cutoff\tn_loop\tn_descent\tlearning_rate\n")
		for history in self.trainhistory:
			fp.write("%1.2e\t%d\t%d\t%1.2e\n"%tuple(history))
		fp.close()
		chdir('..')
	
	def loadMPS(self, current_pwd=True):
		if not current_pwd:
			chdir(current_pwd)
		self.bond_dimension = load('Bondim.npy').tolist()
		self.space_size = len(self.bond_dimension)
		self.trainhistory = load('TrainHistory.npy').tolist()
		self.Loss = load('Loss.npy').tolist()
		self.cutoff = self.trainhistory[-1][0]
		self.descenting_step_length = self.trainhistory[-1][-1]
		self.matrices = []
		for i in range(self.space_size):
			self.matrices.append(load('Mat_%d.npy'%i))
		self.merged_bond=-1
		self.merged_matrix=[]
				# print("current angle:", (angle(m.Give_Psi(array([0]*8),[0]*8))-angle(m.Give_Psi(array([1]*8),[0]*8))) % (2*pi))


def successive_train(t, m, lstm, successive_fidelity, real_fidelity, i):
	t.add_batch()
	##successive ???	
	m = MPS(m.space_size, "random")
	m.cutoff = 0.1
	m.descenting_step_length = 0.1
	m.train(t, 5, 10)
	m.saveMPS(i)
	if i!=0:
		stamp = 'N%dL%d' % (space_size, i)
		chdir('./'+stamp+'/')
		fp = open('MPS.log', 'a')
		fp.write("Present set_size:%d\n"%t.set_size)
		fp.write("Present setting_number:%d\n"%t.setting_number)
		successive_fidelity.append(m.Give_fidelity(lstm))
		real_fidelity.append(m.Give_fidelity(t.mps))
		save('successive_fidelity.npy',array(successive_fidelity))
		save('successive_fidelity.npy',array(real_fidelity))
		fp.close()
		chdir('..')
		print("set size:", t.set_size, " successive fidelity:", successive_fidelity)
		print("Current Real Fidelity:", real_fidelity)
	return deepcopy(m)
	

for space_size in range(8, 50, 1):
	sm = MPS(space_size, "cluster")
	t = training_set(space_size, "MPSsampled", batch_size = 3, initial_batch_number = 1, mps = sm)
	m = MPS(space_size, "random")
	m.cutoff = 0.1
	m.descenting_step_length = 0.1
	m.Show_Loss(t)

	
	successive_fidelity = []
	real_fidelity = []
	last_m = []
	for i in range(200):
		last_m = successive_train(t, m, last_m, successive_fidelity, real_fidelity, i)
		if i > 10 and successive_fidelity[-1] > 0.99995:
			break
	
#!!!!!!!!!!!!!!!!!!!!!!!dont forget to change setting!!!!!!!


# plt.plot(log(1-array(successive_fidelity)))
# plt.show()
# plt.plot(log(m.Loss))
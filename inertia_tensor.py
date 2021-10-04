from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.linalg import eigh
from pylab import Line2D
from numpy import linalg


class InertiaTensor():
	"""
	A class for computing the mass/inertia tensor for 2D distributions.

	...

	Attributes
	__________
	locs 	: (N, 2) array of particle/pixel locations	
	weights : (N, ) array of weights
	apert 	: Aperture imposed by the user
	inertia_tensor 	: Inertia tensor matrix
	sub_length 		: Number of particles/pixels within the final ellipsoid
	n_its 	: Number of iterations undertaken by algorithm
	As/Bs 	: Axis lengths of morphing ellipse for each iteration
	angs 	: Orientation of morphing ellipse for each iteration


	Methods
	__________
	DataLoad(locs, weights, apert = None)
		Loads the input data positions and weights

	Simple(apert = None):
		Computes the simple inertia tensor

	SimpleIterative(self, apert = None, max_iterations = 100, convergence_sensitvity = 0.01):
		Computes the iterative form of the simple inertia tensor

	Reduced(apert = None):
		Computes the reduced inertia tensor

	ReducedIterative(apert = None, max_iterations = 100, convergence_sensitvity = 0.01):
		Computes the iterative form of the reduced inertia tensor

	ShapeParameters(scale = True, scale_to = None):
		Computes the shape parameters of an ellipse described by the inertia tensor

	PlotFit():
		Plots the ellipse described by the inertia tensor over the input particles

	"""

	def __init__(self):
		self.locs 	 = []
		self.weights = []
		self.apert   = np.inf
		self.inertia_tensor = []
		self.sub_length = []
		self.n_its 		= [] 
		self.As			= []
		self.Bs			= []
		self.angs		= []

	def DataLoad(self, locs, weights, apert = None):
		"""
		
		This function loads the data from which the inertia tensor is to be computed.


			Parameters:
					locs 	(float)	: A 2D array of (x, y) particle positions
					weights (float)	: An array of particle weights
					apert 	(float) : Optional aperture to be applied in shape computation

			Returns:
					self.apert		(float)	: A 2D array of (x, y) particle positions in the correct orientation
					self.locs		(float)	: An array of particle weights in the correct orientation
					self.weights	(float)	: Optional aperture to be applied in shape computation		

		"""
		if apert is not None:
			self.apert = apert
		loc_shape = np.shape(locs)
		weights_shape = np.shape(weights)
		if np.isin(2, loc_shape) == False:
			print('Bad data input')
		else:
			if loc_shape[0] == 2:
				locs = locs.T
				loc_shape = np.shape(locs)
			assert loc_shape[0] == weights_shape[0], 'Bad data input, arrays do not match in size'
			self.locs = locs
			self.weights = weights
			print('New data loaded, sizes %s and %s' % (loc_shape, weights_shape))

	def Simple(self, apert = None):
		"""
		
		This function computes the 'simple' form of the inertia tensor. No iteration.


			Parameters:
					apert 	(float) : Optional aperture to be applied in shape computation

			Returns:
					self.inertia_tensor	(float)	: The inertia tensor, a 2x2 Matrix
					self.sub_length		(float)	: The number of particels within the initial aperture

		"""

		# Computes the 'simple' form of the inertia tensor. No iteration.
		if apert is not None:
			self.apert = apert
		dists = np.linalg.norm(self.locs, axis = 1)
		aperture = dists < self.apert
		weight, locs = self.weights[aperture], self.locs[aperture]
		if len(weight)>0:
			R_matrix 	= np.array(locs[:,:,np.newaxis] * locs[:,np.newaxis,:]) # Outer product of locations
			mass_weighted_mat = np.squeeze(np.multiply(R_matrix, weight[:,np.newaxis, np.newaxis])) # Weights each object's outer product
			M_temp		=  np.sum(mass_weighted_mat, axis = 0) 
			sum_weight	=  np.sum(weight, axis = 0) 
			M 			=  np.divide(M_temp, sum_weight) # Dividing by the sum of the weights ensures that the units of matrix elements are dist^2
		self.inertia_tensor, self.sub_length = M, sum(aperture)

	def SimpleIterative(self, apert = None, max_iterations = 100, convergence_sensitvity = 0.01):
		"""
		
		This function computes the 'simple' form of the inertia tensor. Iterative.


			Parameters:
					max_iterations 			(int) : Maximum number of iterations before convergence deemed to have failed
					convergence_sensitvity 	(int) : Fractional change threshold before deemed converged

			Returns:
					self.inertia_tensor	(float)	: The inertia tensor, a 2x2 Matrix
					self.sub_length		(float)	: The number of particels within the initial aperture
					self.As				(float)	: Minor axis length at each iteration
					self.Bs				(float)	: Major axis length at each iteration
					self.angs			(float)	: Angle of ellipse orientation at each iteration
					self.n_its			(float)	: Number of iterations before convergence

		"""

		def simple(weight, locs, r_sph):
			dists = np.linalg.norm(self.locs, axis = 1)
			aperture = dists < self.apert
			weight, locs = self.weights[aperture], self.locs[aperture]
			if len(weight)>0:
				R_matrix 	= np.array(locs[:,:,np.newaxis] * locs[:,np.newaxis,:]) # Outer product of locations
				mass_weighted_mat = np.squeeze(np.multiply(R_matrix, weight[:,np.newaxis, np.newaxis])) # Weights each object's outer product
				M_temp		=  np.sum(mass_weighted_mat, axis = 0) 
				sum_weight	=  np.sum(weight, axis = 0) 
				M 			=  np.divide(M_temp, sum_weight) # Dividing by the sum of the weights ensures that the units of matrix elements are dist^2
			return M

		def ellip(MATRIX):
			vals, vecs 	= sp.linalg.eigh(MATRIX)
			ab			= np.sqrt(vals)  ## Some negative values here, why?
			order		= np.argsort(ab)
			a, b	= ab[order][0], ab[order][1]
			e 			= a/b
			return e

		def conv(r, r1):
			return abs(1. - (r1/r))

		def orient(tensor):
			eigs, eigv = sp.linalg.eigh(tensor)
			order			= np.argsort(eigs).astype('int64')
			structural_axes	= eigv[:,order].T
			maj		= structural_axes[1]
			line	= np.array([1.,0.])
			top 	= np.dot(maj, line)
			bottom  = np.linalg.norm(maj)*np.linalg.norm(line)
			tot		= top/bottom
			ang	 	= np.rad2deg( np.arccos(abs(tot))  )
			ang1	= 90 - ang
			return ang1

		if apert is not None:
			self.apert = apert

		self.As = [self.apert]
		self.Bs = [self.apert]
		self.angs = [90]

		R_matrix 		= np.array(self.locs[:,:,np.newaxis] * self.locs[:,np.newaxis,:]) # Outer product of all vectors outside of for loop to avoid dubplication of efforts
		M			 	= simple(self.weights, self.locs, self.apert) # First iteration 
		final_M 		= np.empty((2,2,))
		final_M[:]		= np.nan
		max_it  		= np.nan
		sub_length		= np.nan
		if (np.isnan(M).any() == False) * (np.isinf(M).any() == False) == True:
			e		= ellip(M) # Sets up first e = a/b value, used to check for convergence
			for iteration in range(1, max_iterations):
				eigenvalues, eigenvectors = sp.linalg.eigh(M) # Computes axis lengths and vectors of previous iteration
				order			= np.argsort(eigenvalues).astype('int64')
				a, b			= np.sqrt(eigenvalues[order]) # Here b>a
				scale_term		= self.apert * (a*b)**(-1./2.)  
				A, B			= a*scale_term, b*scale_term # Rescale ellipse so area same as initial circle
				self.As.append(A)
				self.Bs.append(B)
				self.angs.append(orient(M))
				P_ab			= eigenvectors[:,order].T
				new_locs 		= np.squeeze(np.matmul(P_ab, self.locs[:,:, np.newaxis])) # Rotate coordinates into the frame of the ellipse
				nlocs_sqr		= np.square(new_locs)
				a_sqrs			= np.repeat(A**2, len(new_locs))
				b_sqrs			= np.repeat(B**2, len(new_locs))
				divisors		= np.zeros(np.shape(nlocs_sqr))
				divisors[:,0], divisors[:,1] = a_sqrs, b_sqrs
				r_tilde_sqr 	= np.sum(nlocs_sqr/divisors, axis = 1)	
				WEIGHT			= self.weights/r_tilde_sqr
				corr 			= r_tilde_sqr <= 1. # Check if particles are within ellipse radius
				length			= len(WEIGHT[(corr)*(WEIGHT > 0)])
				if length < 2: # If fewer than two particles in ellipse, break
					break
				else:
					mass_weighted_mat = np.squeeze(np.multiply(R_matrix, self.weights[:,np.newaxis, np.newaxis]))
					M1 				=  np.sum(mass_weighted_mat[corr], axis = 0)
					sum_weight		=  np.sum(self.weights[corr], axis = 0) 
					M1 				=  M1/sum_weight
					if (np.isnan(M1).any()) * (np.isinf(M1).any()) == True:
						break
					e1 =  ellip(M1) # i-th ellipticity value
					converged		= (conv(e, e1) < convergence_sensitvity) 
					if converged == True: # If converged, output final results
						final_M 	= M1
						max_it 		= iteration
						sub_length	= length
						break					
					else:
						M, e = M1, e1 # Replace (i-1)th inertia tensor and ellipticity values with i-th, then go around again

		self.inertia_tensor = final_M
		self.sub_length = sub_length
		self.n_its = max_it

	def Reduced(self, apert = None):
		"""
		
		This function computes the 'reduced' form of the inertia tensor. No iteration.


			Parameters:
					apert 	(float) : Optional aperture to be applied in shape computation

			Returns:
					self.inertia_tensor	(float)	: The inertia tensor, a 2x2 Matrix
					self.sub_length		(float)	: The number of particels within the initial aperture

		"""
		if apert is not None:
			self.apert = apert
		R_matrix 	= np.array(self.locs[:,:,np.newaxis] * self.locs[:,np.newaxis,:])
		locs_sqr 	= np.square(self.locs)
		r_tilde_sqr = np.sum(np.divide(locs_sqr, np.square(self.apert)), axis = 1) 
		weight		= np.divide(self.weights, r_tilde_sqr)
		corr  		= r_tilde_sqr <= 1.
		if sum(corr) > 2:
			mass_weighted_mat = np.squeeze(np.multiply(R_matrix, weight[:,np.newaxis, np.newaxis]))
			M 			=  np.sum(mass_weighted_mat[corr], axis = 0)
			sum_weight	=  np.sum(weight[corr], axis = 0)
			M 			=  np.divide(M, sum_weight) 
		self.inertia_tensor, self.sub_length = M, sum(corr)

	def ReducedIterative(self, apert = None, max_iterations = 100, convergence_sensitvity = 0.01):
		"""
		
		This function computes the 'reduced' form of the inertia tensor. Iterative.


			Parameters:
					max_iterations 			(int) : Maximum number of iterations before convergence deemed to have failed
					convergence_sensitvity 	(int) : Fractional change threshold before deemed converged

			Returns:
					self.inertia_tensor	(float)	: The inertia tensor, a 2x2 Matrix
					self.sub_length		(float)	: The number of particels within the initial aperture
					self.As				(float)	: Minor axis length at each iteration
					self.Bs				(float)	: Major axis length at each iteration
					self.angs			(float)	: Angle of ellipse orientation at each iteration
					self.n_its			(float)	: Number of iterations before convergence

		"""

		def reduced(weight, locs, r_sph):
			M = np.ones((2,2))
			M[:] = np.nan
			R_matrix 	= np.array(locs[:,:,np.newaxis] * locs[:,np.newaxis,:])
			locs_sqr 	= np.square(locs)
			r_tilde_sqr = np.sum(np.divide(locs_sqr, np.square(r_sph)), axis = 1) # Working
			WEIGHT		= np.divide(weight, r_tilde_sqr)
			corr  		= r_tilde_sqr <= 1.
			if sum(corr) > 2:
				mass_weighted_mat = np.squeeze(np.multiply(R_matrix, WEIGHT[:,np.newaxis, np.newaxis]))
				M 			=  np.sum(mass_weighted_mat[corr], axis = 0)
				sum_weight	=  np.sum(WEIGHT[corr], axis = 0) ### ERROR HERE?
				M 			=  np.divide(M, sum_weight) 
			return M

		def ellip(MATRIX):
			vals, vecs 	= sp.linalg.eigh(MATRIX)
			ab			= np.sqrt(vals)  ## Some negative values here, why?
			order		= np.argsort(ab)
			a, b	= ab[order][0], ab[order][1]
			e 			= a/b
			return e

		def conv(r, r1):
			return abs(1. - (r1/r))

		def orient(tensor):
			eigs, eigv = sp.linalg.eigh(tensor)
			order			= np.argsort(eigs).astype('int64')
			structural_axes	= eigv[:,order].T
			maj		= structural_axes[1]
			line	= np.array([1.,0.])
			top 	= np.dot(maj, line)
			bottom  = np.linalg.norm(maj)*np.linalg.norm(line)
			tot		= top/bottom
			ang	 	= np.rad2deg( np.arccos(abs(tot))  )
			return ang

		if apert is not None:
			self.apert = apert

		self.As = [self.apert]
		self.Bs = [self.apert]
		self.angs = [90]

		R_matrix 		= np.array(self.locs[:,:,np.newaxis] * self.locs[:,np.newaxis,:]) # Outer product of all vectors outside of for loop to avoid dubplication of efforts
		M			 	= reduced(self.weights, self.locs, self.apert) # First iteration 
		final_M 		= np.empty((2,2,))
		final_M[:]		= np.nan
		max_it  		= np.nan
		sub_length		= np.nan
		if (np.isnan(M).any() == False) * (np.isinf(M).any() == False) == True:
			e		= ellip(M) # Sets up first e = a/b value, used to check for convergence
			for iteration in range(1, max_iterations):
				eigenvalues, eigenvectors = sp.linalg.eigh(M) # Computes axis lengths and vectors of previous iteration
				order			= np.argsort(eigenvalues).astype('int64')
				a, b			= np.sqrt(eigenvalues[order]) # Here b>a
				scale_term		= self.apert * (a*b)**(-1./2.)  
				A, B			= a*scale_term, b*scale_term # Rescale ellipse so area same as initial circle
				self.As.append(A)
				self.Bs.append(B)
				self.angs.append(orient(M))
				P_ab			= eigenvectors[:,order].T
				new_locs 		= np.squeeze(np.matmul(P_ab, self.locs[:,:, np.newaxis])) # Rotate coordinates into the frame of the ellipse
				nlocs_sqr		= np.square(new_locs)
				a_sqrs			= np.repeat(A**2, len(new_locs))
				b_sqrs			= np.repeat(B**2, len(new_locs))
				divisors		= np.zeros(np.shape(nlocs_sqr))
				divisors[:,0], divisors[:,1] = a_sqrs, b_sqrs
				r_tilde_sqr 	= np.sum(nlocs_sqr/divisors, axis = 1)	
				WEIGHT			= self.weights/r_tilde_sqr
				corr 			= r_tilde_sqr <= 1. # Check if particles are within ellipse radius
				length			= len(WEIGHT[(corr)*(WEIGHT > 0)])
				if length < 2: # If fewer than two particles in ellipse, break
					break
				else:
					mass_weighted_mat 		=  np.squeeze(np.multiply(R_matrix, WEIGHT[:,np.newaxis, np.newaxis])) # Weight unrotated matrix by weights
					M1_temp 				=  np.sum(mass_weighted_mat[corr], axis = 0)
					sum_weight				=  np.sum(WEIGHT[corr], axis = 0) 
					M1 						=  M1_temp/sum_weight
					if (np.isnan(M1).any()) * (np.isinf(M1).any()) == True:
						break
					e1 =  ellip(M1) # i-th ellipticity value
					converged		= (conv(e, e1) < convergence_sensitvity) 
					if converged == True: # If converged, output final results
						final_M 	= M1
						max_it 		= iteration
						sub_length	= length
						break					
					else:
						M, e = M1, e1 # Replace (i-1)th inertia tensor and ellipticity values with i-th, then go around again
		self.inertia_tensor = final_M
		self.sub_length = sub_length
		self.n_its = max_it

	def ShapeParameters(self, scale = True, scale_to = None):
		"""
		
		This function computes the shape parameters of the ellipse described by the inertia tensor


			Parameters:
					scale 		(bool) 	: Scalse axis lengths to ensure that same area as initial circular aperture
					scale_to 	(float) : Scale term - if None, scales to initial circular aperture

			Returns:
					self.a				(float)	: Minor axis length 
					self.b				(float)	: Major axis length 
					self.minor_axis		(float)	: Minor axis vector
					self.major_axis		(float)	: Minor axis vector

		"""

		vals, vecs 	= sp.linalg.eigh(self.inertia_tensor)
		ab			= np.sqrt(vals)
		order 		= np.argsort(ab)
		a, b		= ab[order][0], ab[order][1]
		if scale:
			# Scale axis lengths to ensure that the area of the ellipse is the same as the area of the initial circular aperture
			if scale_to is None:
				scale_term		= self.apert * (a*b)**(-1./2.) # Ensure that area of ellipse is the same as initial circle 
			else:
				scale_term = scale_to * (a*b)**(-1./2.) # Ensure that area of ellipse is the same as circle with a radis 'scale_to'
			a = a*scale_term
			b = b*scale_term
		orientation = vecs[order]
		self.a = a
		self.b = b
		self.minor_axis = orientation[0]
		self.major_axis = orientation[1]

	def PlotFit(self):
		"""
		
		This function plots the ellpise described by the inertia tensor in comparison to the input position array

		"""


		def orient(tensor):
			eigs, eigv = sp.linalg.eigh(tensor)
			order			= np.argsort(eigs).astype('int64')
			structural_axes	= eigv[:,order].T
			maj		= structural_axes[1]
			line	= np.array([1.,0.])
			top 	= np.dot(maj, line)
			bottom  = np.linalg.norm(maj)*np.linalg.norm(line)
			tot		= top/bottom
			ang	 	= np.rad2deg( np.arccos(abs(tot))  )
			return ang

		ang = orient(self.inertia_tensor)

		e1 = Ellipse((0, 0), 2.*self.apert, 2.*self.apert,
							angle=0, linewidth=2, fill=False, zorder=2, edgecolor = 'k')

		e2 = Ellipse((0, 0), 2.*self.b, 2.*self.a,
							angle=ang, linewidth=2, fill=False, zorder=2, edgecolor = 'orange')

		custom_lines = [Line2D([0], [0], color= 'k', ls = '-', lw=2),
						Line2D([0], [0], color= 'orange', ls = '-', lw=2)]

		fig, axs = plt.subplots(1,1, figsize = [6,6])
		axs.scatter(self.locs[:,0], self.locs[:,1], s = 0.4, alpha = 0.5)
		axs.add_patch(e1)
		axs.add_patch(e2)
		axs.set_ylim([-0.3, 0.3])
		axs.set_xlim([-0.3, 0.3])
		axs.legend(custom_lines, ['Initial Aperture', 'Final Ellipse'])
		plt.show()

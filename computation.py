from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.linalg import eigh
from pylab import Line2D
from numpy import linalg
from inertia_tensor import InertiaTensor

################################## TICKET ##################################
locs = np.loadtxt('locs.txt') # Load in particle data x, y coordinates in Mpc. Numpy array.
aperture = 0.03 # Aperture in Mpc
################################## END TICKET ##################################
locs[:,0]*= -1

################################## INITIALISE CLASS OBJECT ##################################
I = InertiaTensor() 
I.DataLoad(locs, np.ones(len(locs)), apert = aperture) # Load particle data into object, set aperture
################################## END INITIALISE CLASS OBJECT ##################################


################################## COMPUTE SIMPLE TENSOR ##################################
I.Simple() # Compute simple inertia tensor for the particle distribution
I.ShapeParameters() # Compute axis lengths and orientations
inertia_tensor_simp = I.inertia_tensor # Grab simple tensor from object library
a_simp, b_simp = I.a, I.b # Axis lengths
min_simp, maj_simp = I.minor_axis, I.major_axis # Orientations
I.PlotFit() # Displays ellipse in relation to particle distribution
################################## END COMPUTE SIMPLE TENSOR ##################################



################################## COMPUTE REDUCED TENSOR ##################################
I.Reduced() # Compute reduced inertia tensor for the particle distribution
I.ShapeParameters()
a_red, b_red = I.a, I.b
min_red, maj_red = I.minor_axis, I.major_axis
inertia_tensor_red = I.inertia_tensor
################################## END COMPUTE REDUCED TENSOR ##################################



################################## COMPUTE REDUCED ITERATIVE TENSOR ##################################
I.ReducedIterative() # Compute reduced iterative inertia tensor for the particle distribution
I.ShapeParameters()
a_redit, b_redit = I.a, I.b
min_redit, maj_redit = I.minor_axis, I.major_axis
inertia_tensor_redit = I.inertia_tensor
################################## END COMPUTE REDUCED ITERATIVE TENSOR ##################################

################################## COMPUTE SIMPLE ITERATIVE TENSOR ##################################
I.SimpleIterative() # Compute simple iterative inertia tensor for the particle distribution
I.ShapeParameters()
a_simpit, b_simpit = I.a, I.b
min_simpit, maj_simpit = I.minor_axis, I.major_axis
inertia_tensor_simpit = I.inertia_tensor
################################## END COMPUTE SIMPLE ITERATIVE TENSOR ##################################


################################## COMPARE INERTIA TENSORS ##################################

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
	return -1.*ang1

ang_simp = orient(inertia_tensor_simp)
ang_red = orient(inertia_tensor_red)
ang_simpit = orient(inertia_tensor_simpit)
ang_redit = orient(inertia_tensor_redit)

e1 = Ellipse((0, 0), 2.*aperture, 2.*aperture,
					angle=0, linewidth=2, fill=False, zorder=2, edgecolor = 'k')

e1 = Ellipse((0, 0), 2.*aperture, 1.*aperture,
					angle=90-20, linewidth=2, fill=False, zorder=2, edgecolor = 'k')



e2 = Ellipse((0, 0), 2.*a_simp, 2.*b_simp,
					angle=ang_simp, linewidth=2, fill=False, zorder=2, edgecolor = 'green')

e3 = Ellipse((0, 0), 2.*a_red, 2.*b_red,
					angle=ang_red, linewidth=2, fill=False, zorder=2, edgecolor = 'orange')

e4 = Ellipse((0, 0), 2.*a_simpit, 2.*b_simpit,
					angle=ang_simpit, linewidth=2, fill=False, zorder=2, edgecolor = 'red')

e5 = Ellipse((0, 0), 2.*a_redit, 2.*b_redit,
					angle=ang_redit, linewidth=2, fill=False, zorder=2, edgecolor = 'yellow')

custom_lines = [Line2D([0], [0], color= 'k', ls = '-', lw=2),
				Line2D([0], [0], color= 'green', ls = '-', lw=2),
				Line2D([0], [0], color= 'orange', ls = '-', lw=2),
				Line2D([0], [0], color= 'red', ls = '-', lw=2),
				Line2D([0], [0], color= 'yellow', ls = '-', lw=2)]

fig, axs = plt.subplots(1,1, figsize = [6,6])
axs.scatter(locs[:,0], locs[:,1], s = 0.4, alpha = 0.5)
axs.add_patch(e1)
#axs.add_patch(e2)
#axs.add_patch(e3)
#axs.add_patch(e4)
#axs.add_patch(e5)
axs.set_ylim([-0.3, 0.3])
axs.set_xlim([-0.3, 0.3])
axs.legend(custom_lines, ['Initial Aperture', 'Final Ellipse : Simple', 'Final Ellipse : Reduced', 'Final Ellipse : Simple Iterative', 'Final Ellipse : Reduced Iterative'])
#plt.savefig('Compare_Fits.png')
plt.show()
################################## END COMPARE INERTIA TENSORS ##################################

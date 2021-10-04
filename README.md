# MassTensor

MassTensor is a python package used to compute the mass distribution tensor for a 2-dimensional distribution of points, particles or pixels.



Script to compute the 2-dimensional mass distribution tensor.

Casper is a python package aimed at predicting the concentration and shape parameter of dark matter haloes as a function of mass and redshift for a specified cosmology.

## Requirements

MassTensor requires:

* Python version 3.6 or later
* Numpy version 1.20.3 or later
* Scipy version 1.7.1 or later
* Matplotlib version 3.4.2 or later (optional)

## Install



## Use

The computation of the mass distribution tensor is undertaken by the InertiaTensor class within **inertia_tensor.py**. An example of use can be found in **computation.py**, as well as some plotting code to illustrate the performance of the algorithm.

## Acknowledgements 

If this code is used in any published work, please acknowledge and cite the original paper for which it was produced:

>> @ARTICLE{2021MNRAS.505...65H,
       author = {{Hill}, Alexander D. and {Crain}, Robert A. and {Kwan}, Juliana and {McCarthy}, Ian G.},
        title = "{The morphology of star-forming gas and its alignment with galaxies and dark matter haloes in the EAGLE simulations}",
      journal = {\mnras},
     keywords = {gravitational lensing: weak, methods: numerical, large-scale structure of Universe, radio continuum: ISM, Astrophysics - Astrophysics of Galaxies, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2021,
        month = jul,
       volume = {505},
       number = {1},
        pages = {65-87},
          doi = {10.1093/mnras/stab1272},
archivePrefix = {arXiv},
       eprint = {2102.13603},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.505...65H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


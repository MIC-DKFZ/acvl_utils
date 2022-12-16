# ACVL Utilities

This repository contains functions / algorithms that the ACVL group uses frequently. Thus, we created this repository to have a central place with efficient implementations of these functions / algorithms.

## Packages

### Array manipulation

#### Slicer:
A dynamic N-dimensional array slicer that returns a tuple that can be used for slicing an N-dimensional array. Works exactly as Python and Numpy slicing, only with different syntax.
The conventional slicing method has the drawback that one must know the dimensionality of the array beforehand. By contrast, this slicer can be adapted dynamically at runtime.


### Miscellaneous

#### Imap tqdm:
Run a function in parallel with a tqdm progress bar and an arbitrary number of arguments.
Results are always ordered and the performance should be the same as of Pool.map.

# Acknowledgements

<p align="left">
  <img src="logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="logos/DKFZ_Logo.png" width="500"> 
</p>

This Repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/).
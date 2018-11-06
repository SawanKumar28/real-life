# ReAl-LiFE: Accelerating the discovery of individualized brain connectomes with GPUs

# About
This software implements a Regularized and GPU-accelerated version of LiFE. The original implementation of LiFE is available at https://github.com/brain-life/encode

## License.
#### Copyright (2018-), anonymized
 
## [Stable code release]
TBA

## Funding.
anonymized

## Installation.
1. Download the base version of LiFE(https://github.com/brain-life/encode).
2. Refer to https://github.com/brain-life/encode to download and install all the dependencies mentioned there.
2. Download (real-life) into the same folder as #1. The sequence is important, as some files in #1 are updated in this step.
2. [Start MatLab](http://www.mathworks.com/help/matlab/startup-and-shutdown.html).
3. Add repository to the [matlab search path](http://www.mathworks.com/help/matlab/ref/addpath.html).

## Dependencies.
* [MATLAB](http://www.mathworks.com/products/matlab/).

## Getting Started

Refer to https://github.com/brain-life/encode to get started.


## [Run the ReAl LiFE code].
(scripts/real/real\_life\_encode.m)
Run 'help real\_life\_encode' for details in the arguments to be used.
Based on which dataset is being used and its location, the paths to diffusion data for training (dwiFile), diffusion data for cross validation (dwiFileRepeat), the anatomical MRI (t1File) and the tractography connectome to be evaluated (tck file) need to be updated.

real\_life\_encode(use\_gpu, tck\_file,subnum, Niter, lambda, gpudev)



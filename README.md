# ReAl-LiFE: Accelerating the discovery of individualized brain connectomes with GPUs

# About
This software implements a Regularized and GPU-accelerated version of LiFE. The original implementation of LiFE is available at https://github.com/brain-life/encode

## License
#### Copyright (2018-), anonymized
 
## Stable code release
TBA

## Funding
anonymized

## Installation
1. Download the base version of LiFE(https://github.com/brain-life/encode).
2. Refer to https://github.com/brain-life/encode to download and install all the dependencies mentioned there.
2. Download (real-life) into the same folder as #1. The sequence is important, as some files in #1 are updated in this step.
2. [Start MatLab](http://www.mathworks.com/help/matlab/startup-and-shutdown.html).
3. Add repository to the [matlab search path](http://www.mathworks.com/help/matlab/ref/addpath.html).

## Dependencies
* [MATLAB](http://www.mathworks.com/products/matlab/).

## Getting Started

Refer to https://github.com/brain-life/encode to get started.


## Run the ReAl LiFE code
(scripts/real/real\_life\_encode.m)
Run 'help real\_life\_encode' for details in the arguments to be used.
Based on which dataset is being used and its location, the paths to diffusion data for training (dwiFile), diffusion data for cross validation (dwiFileRepeat), the anatomical MRI (t1File) and the tractography connectome to be evaluated (tck file) need to be updated.

```
  >>  real_life_encode(use_gpu, tck_file,subnum, Niter, lambda, gpudev)
```
Details on all arguments can be obtained by running "help real\_life\_encode" in MATLAB.

## Reproducing the results of the paper
Based on which dataset is being used and its location, the paths to diffusion data for training (dwiFile), diffusion data for cross validation (dwiFileRepeat), the anatomical T1 image (t1File) and the connectome to be evaluated (tck\_file) need to be updated: <br>
<img src="images/workflow_files.png" alt="workflow" width="300"/> <br>

The baseline results (CPU) for N iterations can be evaluated by running: 
```
  >>  real_life_encode(false, 'x_size1M.tck', x_id, N, 0, 0)
```
where x\_size1M.tck and x\_id are used to identify the location of the input files.

The regularized and GPU-accelerated version with regularization parameter lambda can be evaluated by running:
```
  >>  real_life_encode(true, 'x_size2M.tck', x_id, N, lambda, gpu_device)
```
where gpu\_device is a valid GPU device on the machine.


### Evaluation and expected result.
The performance numbers for CPU can be obtained under "CPU: Time taken during optimization:" in the output of real\_life\_encode: <br><br>
<img src="images/perf_evaluation_cpu.png" alt="perf_cpu" width="300"/> <br>

The performance number for GPU can be obtained under "ReAl: Time taken during optimization:". The one time GPU overhead can be obtained under "ReAl: Time taken during GPU pre-processing:" in the output: <br><br>
<img src="images/perf_evaluation_gpu.png" alt="perf_gpu" width="350"/> <br>

The CPU and GPU times thus obtained can be used to reproduce the results in Figure 2. 

For both CPU and GPU, the training error, cross validation error, summed weights and number of non zero weights can be obtained under RMSE1, RMSE2, wnorm and nnz respectively in the output of real\_life\_encode as seen above.

The script  real\_life\_encode  also saves the optimized weights, lengths of fibers in the pruned connectome, and voxel wise RMSE ratios, at weights\_\<lambda\>l\_\<N\>\_X.mat, lengths\_opt\_\<lambda\>l\_\<N\>\_X.mat, and rrmse\_\<lambda\>l\_\<N\>\_X.mat respectively, where lambda is the regularization parameter, N is the number of iterations, and X.tck is the connectome being evaluated. The effect of regularization, as shown in Figure 3, was obtained by varying the regularization parameter, lambda, uniformly in the log-space between 10^(-4) and 1. These data can be used to reproduce Figures 3. 

function fe = real_life_encode(use_gpu, tck_file,subnum, Niter, lambda, gpudev)
%% Example of initialization and fitting LiFE model - created from the script demo_LiFE
%
%  fe = real_life_encode(use_gpu, tck_file,subnum, Niter, lambda, gpudev)
%
%  use_gpu determines if gpu (true) or cpu (false) will be used
%
%  tck_file and subnum are dataset specific indentifiers, which determine the locations of
%  dwiFile, dwiFileRepeat and fgFileName. Update these paths in the script according to the
%  dataset.
%
%  Niter determines the number of iterations for the otimization algorithm
%
%  lambda determines the strength of regularization parameter
%
%  gpudev determines the gpu device to use, usually a positive integer
%
%  This file was created using demo_LiFE.m 
%  anonymized
%  email: anonymized

% Intialize a local matlab cluster if the parallel toolbox is available.
% This helps speeding up computations espacially for large conenctomes.

% Build the file names for the diffusion data, the anatomical MRI.
dwiFile       = sprintf('run01_fliprot_aligned_trilin.nii.gz',subnum);
dwiFileRepeat = sprintf('run02_fliprot_aligned_trilin.nii.gz',subnum);
t1File        = sprintf('t1.nii.gz');
%Tractography connectome
fgFileName = tck_file;

% The final connectome and data astructure will be saved with this name:
feFileName    = sprintf('LiFE_Subject_%d',subnum);

%% (1.1) Initialize the LiFE model structure, 'fe' in the code below. 
% This structure contains the forward model of diffusion based on the
% tractography solution. It also contains all the information necessary to
% compute model accuracry, and perform statistical tests. You can type
% help('feBuildModel') in the MatLab prompt for more information.

L = 360; % Discretization parameter
fe = feConnectomeInit(dwiFile,fgFileName,feFileName,[],dwiFileRepeat,t1File,L,[1,0]);


%% (1.2) Fit the model. 
% Hereafter we fit the forward model of tracrography using a least-squared
% method. The information generated by fitting the model (fiber weights
% etc) is then installed in the LiFE structure.

M = feGet(fe,'model');
if use_gpu
    %Set regularization parameter
    M.lambda = lambda;
    %Set gpu device id
    M.gpudev = gpudev;
    method = 'bbnnls_gpu';
else
    method = 'bbnnls';
end
fe = feSet(fe,'fit',feFitModel(M,feGet(fe,'dsigdemeaned'),method,Niter,'preconditioner'));

%Training error
rmse1 = feGet(fe, 'total rmse');
%Cross validation error
rmse2 = feGetRep(fe, 'total rmse');

weights = feGet(fe, 'fiber weights');
%Summed weights
wnorm = sum(weights);
%Number of non zero weights
nnz = sum(weights~=0);

%Voxel wise rrmse
rrmse = feGetRep(fe, 'voxel rmse ratio');

%fiber lengths
fg = feGet(fe,'fibers acpc');
w = weights;
w(w>0) = 1;
fgopt = fgExtract(fg, logical(w),'keep');
fg_opt_nodisc = dtiFiberExcludeDiscontinuous(fgopt, 1);
for kk = 1:length(fg_opt_nodisc.fibers)
    fblength_opt(kk) = length(fg_opt_nodisc.fibers{kk});
end
lengths_opt = fblength_opt;

fprintf('********************************\n')
fprintf('Lambda: %f\n', lambda)
fprintf('RMSE1 (train error): %f\n', rmse1);
fprintf('RMSE2 (cross validation error): %f\n', rmse2);
fprintf('wnorm (summed weights): %f\n', wnorm);
fprintf('nnz (number of non zero weihgts): %d\n', nnz);
fprintf('********************************\n')

%Save the output
[~,name,ext] = fileparts(tck_file);
fname = sprintf('fe_%fl_%dNiter_%s.mat', lambda, Niter, name);
save(fname,'fe','-v7.3');
fname = sprintf('rrmse_%fl_%dNiter_%s.mat', lambda, Niter, name);
save(fname, 'rrmse');
fname = sprintf('weights_%fl_%dNiter_%s.mat', lambda, Niter, name);
save(fname, 'weights');
fname = sprintf('lengths_opt_%fl_%dNiter_%s.mat', lambda, Niter, name);
save(fname, 'lengths_opt');

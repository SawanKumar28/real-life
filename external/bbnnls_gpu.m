function out = bbnnls(M, b, x0, opt)
% function out = bbnnls_New(A, b, x0, opt)
% 
% This is a modified version of BBNNLS code originally written by Suvrit Sra, Dongmin Kim
% This version accept as parameter a factorization of matrix A which is a
% structure M containing:
%       1) The Dictionary M.DictSig;
%       2) A sparse 3D array M.Phi with size [nFibers,nAtoms,Nvoxels]

% BBNNLS   -- Solve NNLS problems via SBB
% 
% WARNING Use at own risk!
% NOTE --- guaranteed convergence phase: *REMOVED* for speedup!!
% NOTE --- To speed up code further, *REMOVE* debugging part
%
%
% function out = bbnnls(A, b, x0, )
% Solve a bound-constrained least squares problem, 
%    min    0.5*||Ax-b||^2, s.t. x >= 0
%
%
% x0 -- Starting vector (useful for warm-starts).
%
% OPT -- This structure contains important opt that control how the
% optimization procedure runs. To obtain a default structure the user can
% use 'opt = solopt'. Use 'help solopt' to get a description of
% what the individual opt mean.
%
% Most important options to tune as: opt.tolg, opt.maxit
%
%
% OUT contains the solution and other information about the optimization or
% info indicating whether the method succeeded or failed.
%
% See also: solopt, bcls
%
% Version 1.1 (c) 2010 Suvrit Sra, Dongmin Kim
% 
% Released under the GNU General Public License
% For details on license and terms please see http://www.gnu.org/copyleft/gpl.html
%
%  This version was updated from the orignal
%  Copyright (2015), Franco Pestilli (Indiana Univ.) - Cesar F. Caiafa (CONICET)
%  email: pestillifranco@gmail.com and ccaiafa@gmail.com
%
%  This file was created from a version of bbnnls.m to incorporate ReAL-LiFE
%  anonymized
%  email: anonymized


    

    % do some initialization for maintaining statistics

    out.startTime = tic;
    global ptxFile
    global cudaFile

    st = tic;
    sortVoxels = true;
    if sortVoxels
        [~, order] = sort(M.Phi.subs(:,2));
        nSubs = M.Phi.subs;
        nSubs(:,1) = nSubs(order,1);
        nSubs(:,2) = nSubs(order,2);
        nSubs(:,3) = nSubs(order,3);
        nSubs = nSubs(:,[2,1,3]);
        nVals = M.Phi.vals(order);
        M.Phi = sptensor(nSubs, nVals);
        M.Phi = permute(M.Phi, [2,1,3]);
    end

    M.atoms = uint64(M.Phi.subs(:,1) - 1);
    M.voxels = uint64(M.Phi.subs(:,2) - 1);
    M.fibers = uint64(M.Phi.subs(:,3) - 1);
    M.vals = M.Phi.vals;
    M.nVoxels = size(M.Phi, 2);
    M.nFibers = size(M.Phi, 3);
    M.nCoeffs = size(M.Phi.vals, 1);

    %Pad theta
    M.nTheta = size(M.DictSig, 1);
    nThetaDesired = ceil(size(M.DictSig,1)/32)*32;
    padSize = nThetaDesired - M.nTheta;
    M.DictSig = padarray(M.DictSig, [padSize 0], 'post');
    b = reshape(b, [M.nTheta, M.nVoxels]);
    b = padarray(b, [gather(padSize) 0], 'post');
    b = b(:);
    M.nTheta = size(M.DictSig, 1);

    %Complile ptx
    fver = 'cuda_methods';
    cudaFile = strcat(fver, '.cu');
    ptxFile = strcat(fver, '.ptx');
    [pathstr{1},name{1},ext{1}] = fileparts(which(cudaFile));
    filename = [pathstr{1} filesep name{1} ext{1}];
    ptxfilename = [pathstr{1} filesep name{1} '.ptx'];
    cmdline = sprintf('nvcc -ptx -DnTh=%d --output-file %s %s\n',...
                int32(M.nTheta/32), ptxfilename, filename);
    [status,cmdout] = system(cmdline);

    gpudev = gpuDevice(M.gpudev);
    M = rmfield(M, 'Phi');
    M = structfun(@gpuArray,M,'UniformOutput',false);
    b = gpuArray(b);
    x0 = gpuArray(x0);

    [~, M.v_ia, ~] = unique(M.voxels);
    M.v_ia = M.v_ia - 1;
    M.v_ia(1 + size(M.v_ia, 1)) = M.nCoeffs;

    M.atoms = uint64(double(M.atoms) * M.nTheta);
    M.voxels = uint64(double(M.voxels) * M.nTheta);

    out.ppTime = toc(st);

    out.iter = 0;
    out.status = 'Failure';
    out.iterTime = 0;
    out.iterTimes = nan*ones(opt.maxit,1,'gpuArray');
    out.objTimes  = nan*ones(opt.maxit,1,'gpuArray');
    out.pgTimes   = nan*ones(opt.maxit,1,'gpuArray');
    out.trueError = nan*ones(opt.maxit,1,'gpuArray');

    %fgx = @(x) funcGrad(A,b, x); % function to compute obj and grad
    
    fgx = @(x) funcGrad(M, b, x); % function to compute obj and grad
    
    % HINT: Very important for overall speed is to have a good x0
    out.x      = x0;
    out.refx   = x0;
    [out.refobj, out.grad]   = fgx(out.x);
    out.oldg   = out.grad;
    out.refg   = out.oldg;

    %% Begin the main algorithm
    fprintf('Lambda %d\n', M.lambda);
    if (opt.verbose)
       fprintf('Running: **** SBB-NNLS ****\n\n');
       fprintf('Iter   \t     Obj\t\t  ||pg||_inf\t\t ||x-x*||\n');
       fprintf('-------------------------------------------------------\n');
    end

    objectives = zeros(opt.maxit,1,'gpuArray');
    %f = figure;
    while 1
        out.iter = out.iter + 1;
        st = tic;

        % HINT: edit checkTermination to determine good criterion for yourself!
        [termReason, out.pgTimes(out.iter)] = checkTermination(opt, out);
        if (termReason > 0), break; end

        % HINT: computeBBStep is the one to implement most carefully
        %[step out] = computeBBStep(A, b, out);
        [step out] = computeBBStep(M, b, out);
        out.x = out.x - step * out.grad;
        out.oldg = out.grad;
        
        % HINT: projection step: can replace the 0 by an epsilon to truncate
        % values close to 0
        out.x(out.x < 0) = 0;

        [out.obj out.grad] =  fgx(out.x);

        objectives(out.iter) = out.obj;
%         clf;
%         plot(objectives);
%         title('Objective ||Ax-b||^2');
%         xlabel('Iteration');
%         ylabel('Objective');
%         drawnow;
        
        % HINT: can remove, as this is just for statistics
        out.objTimes (out.iter) = out.obj;
        out.iterTimes(out.iter) = toc(out.startTime);
        
        % HINT: for debugging, to see how result develops if true x* is known
        if (opt.truex), out.trueError(out.iter) = norm(opt.xt-out.x); end
        if (opt.verbose)
            fprintf('%04d\t %E\t%E\t%E\n', out.iter, out.obj, out.pgTimes(out.iter), out.trueError(out.iter)); 
        end

        out.iterTime = out.iterTime + toc(st);
    end % of while

    %%  Final statistics and wrap up
    out.time = toc(out.startTime);
    fprintf('%d iters : %f %f PP %f\n', opt.maxit, out.time, out.iterTime, out.ppTime);
    out.status = 'Success';
    out.termReason = setTermReason(termReason);
    out.x    = gather(out.x);
    out.refx = gather(out.refx);
    out.grad = gather(out.grad);
    out.oldg = gather(out.oldg);
    out.refg = gather(out.refg);

    fprintf('----------------------------\n')
    fprintf('ReAl: Time taken during optimization: %f s\n', out.iterTime)
    fprintf('ReAl: Time taken during GPU pre-processing: %f s\n', out.ppTime)
    fprintf('----------------------------\n')
    
end

% Compute BB step; for SBB also modifies out.oldg, and this change must be
% passed back to the calling routine, else it will fail!
%function [step out] = computeBBStep(A, b, out)
function [step out] = computeBBStep(A, b, out)
    % HINT: Can tune the x==0 to replace it by an epsilon to do TUNING
    gp = find(out.x == 0 & out.grad > 0);
    out.oldg(gp) = 0;

    Ag = M_times_w_gpu(A, out.oldg);
    
    % HINT: In my experience, the falling alternating steps perform better
    if (mod(out.iter, 2) == 0)
        step = (out.oldg' * out.oldg) / (Ag' * Ag);
    else
        numer = Ag' * Ag;
        Ag = Mtransp_times_b_gpu(A, Ag);
        Ag(gp) = 0;
        step = numer / (Ag' * Ag);
    end
end

% compute obj function and gradient --- requires good implementation of A*x
% and A'*y for appropriate x and y
function [f g] = funcGrad(A, b, x)   
    Ax = M_times_w_gpu(A, x) - b;

    f = 0.5*norm(Ax)^2;
    if (nargout > 1)
        g = Mtransp_times_b_gpu(A, Ax) + A.lambda;
    end
end

function [Ax] = M_times_w_gpu(A,x)
    global ptxFile
    global cudaFile
    [nTheta]  = size(A.DictSig,1);
    [nAtoms] = size(A.DictSig, 2);
    [nFibers] = A.nFibers;
    [nCoeffs] = A.nCoeffs;
    [nVoxels] = A.nVoxels;

    MTW_CU_OBJ= parallel.gpu.CUDAKernel(ptxFile,cudaFile,'M_times');
    MTW_CU_OBJ.GridSize        = [size(A.v_ia, 1)-1 1];;
    MTW_CU_OBJ.ThreadBlockSize = [32 1];

    D = A.DictSig;
    D_vec   =  D(:);
    Y       =  zeros(nTheta,nVoxels,'gpuArray');
    Y_vec   =  Y(:);

    Ax = feval(MTW_CU_OBJ,Y_vec,A.atoms,A.voxels,A.fibers,...
                A.vals,D_vec,x,nTheta,nVoxels,nCoeffs, A.v_ia);
end

function [Atx] = Mtransp_times_b_gpu(A, x)
    global ptxFile
    global cudaFile
    [nTheta]  = size(A.DictSig,1);
    [nAtoms] = size(A.DictSig, 2);
    [nFibers] = A.nFibers;
    [nCoeffs] = A.nCoeffs;
    [nVoxels] = A.nVoxels;

    MTB_CU_OBJ= parallel.gpu.CUDAKernel(ptxFile,cudaFile,'Mtransp_times');
    MTB_CU_OBJ.GridSize        = [size(A.v_ia, 1)-1 1];;
    MTB_CU_OBJ.ThreadBlockSize = [32 1];

    D = A.DictSig;
    D_vec   =  D(:);
    w       =  zeros(nFibers,1,'gpuArray');
    Atx     =  feval(MTB_CU_OBJ,w,A.atoms,A.voxels,A.fibers,A.vals,D_vec,...
                    x,nFibers,nTheta,nCoeffs,A.v_ia);
end

% check various termination criteria; return norm of pg
% the strictest is norm of pg
% HINT: for speedup, use maybe just opt.tolo or some other criterion that
% you like.
function [v pg] = checkTermination(options, out)
    % pgnorm limit -- need to check this first of all
    gp = find( (out.x ~= 0 | out.grad < 0));

    pg = norm(out.grad(gp), 'inf');
    if (pg < options.tolg), v=8; return; end

    % First check if we are doing termination based on running time
    if (options.time_limit)
        out.time = etime(clock, out.start_time);
        if (out.time >= options.maxtime)
            v = 1;
            return;
        end
    end

    % Now check if we are doing break by tolx
    if (options.use_tolx)
        if (norm(out.x-out.oldx)/norm(out.oldx) < options.tolx)
            v = 2;
            return;
        end
    end

    % Are we doing break by tolo (tol obj val)
    if (options.use_tolo && out.iter > 2)
        delta = abs(out.objTimes(out.iter-1)-out.objTimes(out.iter-2));
        if (delta < options.tolo)
            v = 3;
            return;
        end
    end

    % Finally the plain old check if max iter has been achieved
    if (out.iter >= options.maxit)
        v = 4;
        return;
    end

    % KKT violation
    if (options.use_kkt)
        if abs(out.x' * out.grad) <= options.tolk
            v = 7;
            return;
        end
    end


    % All is ok...
    v = 0;
end

%% Prints status
function showStatus(out, options)
    if (options.verbose)
        fprintf('.');
        if (mod(out.iter, 30) == 0)
            fprintf('\n');
        end
    end
end

% String representation of termination
function r = setTermReason(t)
    switch t
      case 1
        r = 'Exceeded time limit';
      case 2
        r = 'Relative change in x small enough';
      case 3
        r = 'Relative change in objvalue small enough';
      case 4
        r = 'Maximum number of iterations reached';
      case 5
        r = '|x_t+1 - x_t|=0 or |grad_t+1 - grad_t| < 1e-9';
      case 6
        r = 'Line search failed';
      case 7
        r = '|x^T * grad| < opt.pbb_gradient_norm';
      case 8
        r = '|| grad ||_inf < opt.tolg';
      case 100
        r = 'The active set converged';
      otherwise
        r = 'Undefined';
    end
end


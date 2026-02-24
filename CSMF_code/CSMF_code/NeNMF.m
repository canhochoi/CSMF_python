function [W,H,iter,elapse,HIS]=NeNMF(V,r,varargin)

% Non-negative Matrix Factorization via Nesterov's Optimal Gradient Method.
% NeNMF: Matlab Code for TSP Submission

% Reference
%  N. Guan, D. Tao, Z. Luo, and B. Yuan, "NeNMF: An Optimal Gradient Method
%  for Non-negative Matrix Factorization", IEEE Transactions on Signal
%  Processing, Vol. 60, No. 6, PP. 2882-2898, 2012.

% The model is V \approx WH, where V, W, and H are defined as follows:
% V (m x n-dimension): data matrix including n samples in m-dimension
% space;
% W (m x r-dimension): basis matrix including r bases in m-dimension space;
% H (r x n-dimension): coefficients matrix includeing n encodings in
% r-dimension space.

% Written by Naiyang Guan (ny.guan@gmail.com)
% Copyright 2010-2011 by Naiyang Guan and Dacheng Tao
% Last Modified Sept. 25 2011

% <Inputs>
%        A : Input data matrix (m x n)
%        r : Target low-rank
%
%        (Below are optional arguments: can be set by providing name-value pairs)
%        MAX_ITER : Maximum number of iterations. Default is 1,000.
%        MIN_ITER : Minimum number of iterations. Default is 10.
%        MAX_TIME : Maximum amount of time in seconds. Default is 100,000.
%        W_INIT : (m x r) initial value for W.
%        H_INIT : (r x n) initial value for H.
%        TOL : Stopping tolerance. Default is 1e-7. If you want to obtain a more accurate solution, decrease TOL and increase MAX_ITER at the same time.
%        VERBOSE : 0 (default) - No debugging information is collected.
%                  1 (debugging purpose) - History of computation is returned by 'HIS' variable.
%                  2 (debugging purpose) - History of computation is additionally printed on screen.
% <Outputs>
%        W : Obtained basis matrix (m x r)
%        H : Obtained coefficients matrix (r x n)
%        iter : Number of iterations
%        elapse : CPU time in seconds
%        HIS : (debugging purpose) History of computation
%
% <Usage Examples>
%        NeNMF(A,10)
%        NeNMF(A,20,'verbose',1)
%        NeNMF(A,30,'verbose',2,'w_init',rand(m,r))
%        NeNMF(A,5,'verbose',2,'tol',1e-5)

% Note: another file 'GetStopCriterion.m' should be put in the same
% directory as this file.

if ~exist('V','var'),    error('please input the sample matrix.\n');    end
if ~exist('r','var'),    error('please input the low rank.\n'); end

[m,n]=size(V);

% Default setting
MaxIter=1000;
MinIter=10;
MaxTime=100000;
W0=rand(m,r);
H0=rand(r,n);
tol=1e-5;
verbose=0;

% Read optional parameters
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'MAX_ITER',    MaxIter=varargin{i+1};
            case 'MIN_ITER',    MinIter=varargin{i+1};
            case 'MAX_TIME',    MaxTime=varargin{i+1};
            case 'W_INIT',      W0=varargin{i+1};
            case 'H_INIT',      H0=varargin{i+1};
            case 'TOL',         tol=varargin{i+1};
            case 'VERBOSE',     verbose=varargin{i+1};
            otherwise
                error(['Unrecognized option: ',varargin{i}]);
        end
    end
end

ITER_MAX=1000;      % maximum inner iteration number (Default)
ITER_MIN=10;        % minimum inner iteration number (Default)
global STOP_RULE;
STOP_RULE = 1;      % '1' for Projected gradient norm (Default)
                    % '2' for Normalized projected gradient norm
                    % '3' for Normalized KKT residual

% Initialization
W=W0; H=H0;
WtW=W'*W; WtV=W'*V;
HHt=H*H'; HVt=H*V';
GradH=WtW*H-WtV;
GradW=W*HHt-HVt';
init_delta=GetStopCriterion(STOP_RULE,[W',H],[GradW',GradH]);
tolH=max(tol,1e-3)*init_delta;
tolW=tolH;               % Stopping tolerance
constV=sum(sum(V.^2));

% Historical information
HIS.niter=0;
HIS.t=0;
HIS.f=sum(sum(WtW.*HHt))-2*sum(sum(WtV.*H));
HIS.p=init_delta;

% Iterative updating
elapse=cputime;
W=W';
for iter=1:MaxIter,
    % Optimize H with W fixed
    [H,iterH]=NNLS(H,WtW,WtV,ITER_MIN,ITER_MAX,tolH);
    if iterH<=ITER_MIN,
        tolH=tolH/10;
    end
    HHt=H*H';   HVt=H*V';
    
    % Optimize W with H fixed
    [W,iterW,GradW]=NNLS(W,HHt,HVt,ITER_MIN,ITER_MAX,tolW);
    if iterW<=ITER_MIN,
        tolW=tolW/10;
    end
    WtW=W*W'; WtV=W*V;
    GradH=WtW*H-WtV;
    HIS.niter=HIS.niter+iterH+iterW;
    delta=GetStopCriterion(STOP_RULE,[W,H],[GradW,GradH]);
    
    % Output running detials
    if verbose,
        HIS.f=[HIS.f,sum(sum(WtW.*HHt))-2*sum(sum(WtV.*H))];
        HIS.t=[HIS.t,cputime-elapse];
        HIS.p=[HIS.p,delta];
        if (verbose==2) && (rem(iter,10)==0),
            fprintf('%d:\tstopping criteria = %e,\tobjective value = %f.\n', iter,delta/init_delta,0.5*(HIS.f(end)+constV));
        end
    end
    
    % Stopping condition
    if (delta<=tol*init_delta && iter>=MinIter) || HIS.t(end)>=MaxTime,
        break;
    end
end
W=W';
elapse=cputime-elapse;

if verbose,
    HIS.f=0.5*(HIS.f+constV);
    if verbose==2,
        fprintf('\nFinal Iter = %d,\tFinal Elapse = %f.\n', iter,elapse);
    end
end

% Non-negative Least Squares with Nesterov's Optimal Gradient Method
function [H,iter,Grad]=NNLS(Z,WtW,WtV,iterMin,iterMax,tol)

global STOP_RULE;

L=norm(WtW);    % Lipschitz constant
H=Z;    % Initialization
Grad=WtW*Z-WtV;     % Gradient
alpha1=1;

for iter=1:iterMax,
    H0=H;
    H=max(Z-Grad/L,0);    % Calculate sequence 'Y'
    alpha2=0.5*(1+sqrt(1+4*alpha1^2));
    Z=H+((alpha1-1)/alpha2)*(H-H0);
    alpha1=alpha2;
    Grad=WtW*Z-WtV;
    
    % Stopping criteria
    if iter>=iterMin,
        % Lin's stopping condition
        pgn=GetStopCriterion(STOP_RULE,Z,Grad);
        if pgn<=tol,
            break;
        end
    end
end

Grad=WtW*H-WtV;
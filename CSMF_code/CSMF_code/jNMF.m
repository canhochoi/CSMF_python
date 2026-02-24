function [W,H,err,elapse] = jNMF(X,vecPara,vecN,cut,varargin)
%<Input>
%        X (M x sumN-dimension): data matrices
%        vecPara: Target low-rank 
%        vecN: sample number
%        cut:the thrshold to select significantly  element

%        (Below are optional arguments: can be set by providing name-value pairs)
%        MAX_ITER : Maximum number of inner NeNMF iterations. Default is 1,000.
%        MIN_ITER : Minimum number of inner NeNMF iterations. Default is 10.
%        MAX_TIME : Maximum amount of inner NeNMF time in seconds. Default is 100,000.      
%        TOL : Stopping tolerance. Default is 1e-6. If you want to obtain a more accurate solution, decrease TOL.
%<Output>
%        W : Obtained basis matrix 
%        H : Obtained coefficients matrix
%        err : record object function
%        elapse : CPU time in seconds


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Default configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     % maximum outer iteration number (Default)
tol = 1e-6;
MaxIter = 100;
MinIter = 2;
MaxTime = 100000;
% Read optional parameters
if (rem(length(varargin),2) == 1)
    error('Optional parameters should always go by pairs');
else
    for i = 1:2:(length(varargin)-1)
        switch upper(varargin{i})
           
            case 'MAX_ITER',    MaxIter = varargin{i+1};
            case 'MIN_ITER',    MinIter = varargin{i+1};
            case 'MAX_TIME',    MaxTime = varargin{i+1};
            case 'TOL',         tol = varargin{i+1};
            otherwise
                error(['Unrecognized option: ',varargin{i}]);
        end
    end
end

global STOP_RULE;
STOP_RULE = 1;      % '1' for Projected gradient norm (Default)
% '2' for Normalized projected gradient norm
% '3' for Normalized KKT residual
numN = length(vecN);
sumN = zeros(1,numN);
sumN(1) = vecN(1);
for i1 = 2:numN
    sumn = 0;
    for j1 = 1:i1
        sumn = sumn+vecN(j1);
    end
    sumN(i1) = sumn;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X0 = X;
[Wc,Hc,~,~,~] = NeNMF(X0,vecPara(1),'max_iter',MaxIter,'min_iter',MinIter,'max_time',MaxTime,'tol',tol);
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Remove the significantly common part from original data matrices ,then
% compute the specific basis and coefficient matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
wc = zscore(Wc);Wc(logical(wc <= cut)) = 0;
hc = zscore(Hc');hc = hc';Hc(logical(hc <= cut)) = 0;
Hc_record = cell(1,numN);Hc_record{1,1} = Hc(:,1:sumN(1));
for i = 2:numN
    Hc_record{1,i} = Hc(:,sumN(i-1)+1:sumN(i));
end
X_record = cell(1,numN);
X_record{1,1} = max(X0(:,1:sumN(1))-Wc*Hc_record{1,1},0);
for i = 2:numN
    X_record{1,i} = max(X0(:,sumN(i-1)+1:sumN(i))-Wc*Hc_record{1,i},0);
end
Ws_record = cell(1,numN);Hs_record = cell(numN);
for i = 1:numN
    if vecPara(i+1) > 0
         [Ws_record{1,i},Hs_record{1,i},~,~,~] = NeNMF(X_record{1,i},vecPara(i+1),'max_iter',MaxIter,'min_iter',MinIter,'max_time',MaxTime,'tol',tol);
    else
        Ws_record{1,i} = [];Hs_record{i,i} = [];
    end
end
for i = 1:numN
    for j = 1:numN
        if i ~= j
            Hs_record{i,j} = zeros(vecPara(i+1),vecN(j));
        end
    end
end

Hs = cell(numN,1);
for i = 1:numN
    hs = [];
    for j = 1:numN
        hs = [hs,Hs_record{i,j}];
    end
    Hs{i,1} = hs;
end
W = Wc;Hc = [];
for i = 1:numN
    W = [W,Ws_record{1,i}];Hc = [Hc,Hc_record{1,i}];
end
H = Hc;
for i = 1:numN
    H = [H;Hs{i,1}];
end
lambda = diag(sum(W)); W = W*pinv(lambda);H = lambda*H;
err = (norm(X0-W*H,'fro'))^2;
elapse = cputime;

function [W,H,err,elapse] = iNMF(X,vecN,vecPara,varargin)
%<Input>
%        X (M x sumN-dimension): data matrices
%        vecN: sample number
%        vecPara: Target low-rank

%        (Below are optional arguments: can be set by providing name-value pairs)
%        MAX_ITER : Maximum number of inner NeNMF iterations. Default is 1,000.
%        MIN_ITER : Minimum number of inner NeNMF iterations. Default is 10.
%        MAX_TIME : Maximum amount of inner NeNMF time in seconds. Default is 100,000.      
%        TOL : Stopping tolerance. Default is 1e-6. If you want to obtain a more accurate solution, decrease TOL.
%<Output>
%        W : Obtained basis matrix (M x (nc+ns1+ns2+...)
%        H : Obtained coefficients matrix ((nc+ns1+ns2+...) x (N1+N2+...)
%        err : record object function
%        elapse : CPU time in seconds
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numN = length(vecN);
global STOP_RULE;
STOP_RULE = 1;
% '1' for Projected gradient norm (Default)
% '2' for Normalized projected gradient norm
% '3' for Normalized KKT residual

X0 = X;
% record the sum of vecN
sumN = zeros(1,numN);
sumN(1) = vecN(1);
for i1 = 2:numN
    sumn = 0;
    for j1 = 1:i1
        sumn = sumn+vecN(j1);
    end
    sumN(i1) = sumn;
end
vecParas=vecPara(2:end) + vecPara(1);%%%%%%%%%parameter for individual factorization
% record the sum of vecParas
sumParas = zeros(1,numN);
sumParas(1) = vecParas(1);
for i2=2:numN
    sumn = 0;
    for j2 = 1:i2
        sumn = sumn+vecParas(j2);
    end
    sumParas(i2) = sumn;
end

% record each matrix
X_record = cell(1,numN);X_record{1,1} = X(:,1:vecN(1));
for i3 = 2:numN
    X_record{1,i3} = X(:,sumN(i3-1)+1:sumN(i3));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tol = 1e-3;
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% factorization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W_record = cell(1,numN);H_record = cell(1,numN);
for i4 = 1:numN
    [W_record{1,i4},H_record{1,i4},~,~,~] = NeNMF(X_record{1,i4},vecParas(i4),'max_iter',MaxIter,'min_iter',MinIter,'max_time',MaxTime,'tol',tol);
    % normalization
    lambda = diag(sum(W_record{1,i4}));W_record{1,i4} = W_record{1,i4}*pinv(lambda);H_record{1,i4} = lambda*H_record{1,i4};
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute Wc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Wc_record = cell(1,numN-1);Hc_record = cell(1,numN);Ws_record = cell(1,numN);Hs_record = cell(1,numN);
C = corr(W_record{1,1},W_record{1,2});C1 = ones(size(C));D = C1-C;[M,~] = Hungarian(D);[c1,c2] = find(M == 1);
w = zeros(1,length(c1));
for r = 1:length(c1)
    w(r) = D(c1(r),c2(r));
end
[~,id] = sort(w);Wc_record{1,1} = (W_record{1,1}(:,c1(id(1:vecPara(1))))+W_record{1,2}(:,c2(id(1:vecPara(1)))))/2;
Hc_record{1,1} = H_record{1,1}(c1(id(1:vecPara(1))),:);Hc_record{1,2} = H_record{1,2}(c2(id(1:vecPara(1))),:);
M(c1(id(1:vecPara(1))),c2(id(1:vecPara(1)))) = 0;
[ir,jc] = find(M~=0);
if vecParas(1) > vecParas(2)
    ir1 = setdiff(1:vecParas(1),c1);ir = union(ir,ir1);
else
    jc1 = setdiff(1:vecParas(2),c2);jc = union(jc,jc1);
end
Ws_record{1,1} = W_record{1,1}(:,ir);Ws_record{1,2} = W_record{1,2}(:,jc);
Hs_record{1,1} = H_record{1,1}(ir,:);Hs_record{1,2} = H_record{1,2}(jc,:);
for i = 2:numN-1
    C = corr(Wc_record{1,i-1},W_record{1,i+1});C1 = ones(size(C));D = C1-C;[M,~] = Hungarian(D);[c1,c2] = find(M == 1);
    w = zeros(1,length(c1));
    for r = 1:length(c1)
        w(r) = D(c1(r),c2(r));
    end
    [~,id] = sort(w);Wc_record{1,i} = (Wc_record{1,i-1}+W_record{1,i+1}(:,c2(id(1:vecPara(1)))))/2;
    Hc_record{1,i+1} = H_record{1,i+1}(c2(id(1:vecPara(1))),:);
    M(:,c2(id(1:vecPara(1)))) = 0;
    [~,jc] = find(M ~= 0);
    jc1 = setdiff(1:vecParas(i+1),c2);jc = union(jc,jc1);
    Ws_record{1,i+1} = W_record{1,i+1}(:,jc);Hs_record{1,i+1} = H_record{1,i+1}(jc,:);
end
% intergration
W = Wc_record{1,numN-1};
for i = 1:numN
    W = [W,Ws_record{1,i}];
end
Hc = [];
for j = 1:numN
    Hc = [Hc,Hc_record{1,j}];
end

H = Hc;
H = [H;Hs_record{1,1},zeros(vecPara(2),sumN(numN)-sumN(1))];
for i3 = 2:numN-1
    H = [H;zeros(vecPara(i3+1),sumN(i3-1)),Hs_record{1,i3},zeros(vecPara(i3+1),sumN(numN)-sumN(i3))];
end
H = [H;zeros(vecPara(numN+1),sumN(numN-1)),Hs_record{1,numN}];
lambda = diag(sum(W)); W = W*pinv(lambda);H = lambda*H;
err = (norm(X0-W*H,'fro'))^2;
elapse = cputime;

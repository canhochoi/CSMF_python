function [W,H,iter,elapse,HIS] = CSMF(X,vecN,vecPara,varargin)
% Common and Specific non-negative matrix sparse factorization via Nesterov's Optimal Gradient Method.
% <Inputs>
%        X (M x multi-dimension): contain more than two data matrices,the number of input matrices can be any number.
%        vecN : The vector of mutiple sample sets
%        vecPara: The vector of target low-rank
%        (Below are optional arguments: can be set by providing name-value pairs)
%        ITER : Maximum number of out iterations. Default is 500.
%        MAX_ITER : Maximum number of inner NeNMF iterations. Default is 1,000.
%        MIN_ITER : Minimum number of inner NeNMF iterations. Default is 2.
%        MAX_TIME : Maximum amount of inner NeNMF time in seconds. Default is 100,000.
%        W_INIT : (m x r) initial value for W.
%        H_INIT : (r x n) initial value for H.
%        VERBOSE : 0 (default) - No debugging information is collected.
%                  1 (debugging purpose) - History of computation is returned by 'HIS' variable.
%                  2 (debugging purpose) - History of computation is additionally printed on screen. 
%        TOL : Stopping tolerance. Default is 1e-6. If you want to obtain a more accurate solution, decrease TOL.
%        
% <Outputs>
%        W : Obtained basis matrix (M x (nc+ns1+ns2)
%        H : Obtained coefficients matrix ((nc+ns1+ns2) x (N1+N2))
%        iter : record step
%        elapse : CPU time in seconds
%        HIS: record running information
%
% <Usage Examples>
%        CSMF([X1,X2],[100,100],[4,1,2]);
%        CSMF([X1,X2,X3],[100,100,200],[4,1,2,5],'verbose',1)       
%        

% Note: another file 'GetStopCriterion.m' and 'NeNMF.m'should be put in the same
% directory as this file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = size(X,1); numN = length(vecN);
global STOP_RULE;
STOP_RULE = 1;
% '1' for Projected gradient norm (Default)
% '2' for Normalized projected gradient norm
% '3' for Normalized KKT residual
if numN<= 1
    disp('Please input more than two matrices!')
    W = []; H = []; iter = 0; elapse = cputime; HIS = [];
else
    % record the sum of vecN
    sumN = zeros(1,numN); sumN(1) = vecN(1);
    for i1 = 2:numN
        sumn = 0;
        for j1=1:i1
            sumn = sumn+vecN(j1);
        end
        sumN(i1) = sumn;
    end
    % record the sum of vecPara
    sumPara = zeros(1,numN+1);
    sumPara(1) = vecPara(1);
    for i2=2:numN+1
        sumn = 0;
        for j2 = 1:i2
            sumn = sumn+vecPara(j2);
        end
        sumPara(i2) = sumn;
    end
    
    % record each matrix
    X_record = cell(1,numN); X_record{1,1} = X(:,1:vecN(1));
    for i3 = 2:numN
        X_record{1,i3} = X(:,sumN(i3-1)+1:sumN(i3));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Default configuration
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Iter = 500;      % maximum outer iteration number (Default)
    tol = 1e-6;
    MaxIter = 100;
    MinIter = 2;
    MaxTime = 100000;
    verbose = 0;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Hp_record = cell(numN+1,numN);
    for i4 = 1:numN
        Hp_record{1,i4} = max(rand(vecPara(1),vecN(i4)),0);
    end
    for j3 = 1:numN
        for k1 = 1:numN
            Hp_record{j3+1,k1} = zeros(vecPara(j3+1),vecN(k1));
        end
    end
    for s1 = 1:numN
        Hp_record{s1+1,s1} = max(rand(vecPara(s1+1),vecN(s1)),0);
    end
    
    H_record = cell(numN+1,1);H0 = [];
    for t1 = 1:numN+1
        H_record{t1,1} = [];
        for t2 = 1:numN
            H_record{t1,1} = [H_record{t1,1},Hp_record{t1,t2}];
        end
    end
    
    for t3 = 1:numN+1
        H0 = [H0;H_record{t3,1}];
    end
    W0 = max(rand(M,sumPara(numN+1)),0);
    
    % Read optional parameters
    if (rem(length(varargin),2) == 1)
        error('Optional parameters should always go by pairs');
    else
        for i = 1:2:(length(varargin)-1)
            switch upper(varargin{i})
                case 'ITER',        Iter = varargin{i+1};
                case 'MAX_ITER',    MaxIter = varargin{i+1};
                case 'MIN_ITER',    MinIter = varargin{i+1};
                case 'MAX_TIME',    MaxTime = varargin{i+1};
                case 'W_INIT',      W0 = varargin{i+1};
                case 'H_INIT',      H0 = varargin{i+1};
                case 'TOL',         tol = varargin{i+1};
                case 'VERBOSE',     verbose = varargin{i+1};
                otherwise
                    error(['Unrecognized option: ',varargin{i}]);
            end
        end
    end
    
  
   
    W = W0;H = H0;
    lambda = diag(sum(W)); W = W*pinv(lambda); H = lambda*H;% normalization
    % initial stopping tolerence value
    V = X;
    WtW = W'*W; WtV = W'*V;
    HHt = H*H'; HVt = H*V';
    GradH = WtW*H-WtV; GradW = W*HHt-HVt';
    init_delta = GetStopCriterion(STOP_RULE,[W',H],[GradW',GradH]);
    
    Wp_record = cell(1,numN+1);
    Wp_record{1,1} = W(:,1:sumPara(1));
    for i5 = 2:numN+1
        Wp_record{1,i5} = W(:,sumPara(i5-1)+1:sumPara(i5));
    end
    
    Hc_record = cell(1,numN); Hc_record{1,1} = H(1:vecPara(1),1:sumN(1));
    for i1 = 2:numN
        Hc_record{1,i1} = H(1:vecPara(1),sumN(i1-1)+1:sumN(i1));
    end
    Hs_record = cell(1,numN); Hs_record{1,1} = H(sumPara(1)+1:sumPara(2),1:sumN(1));
    for i2 = 2:numN
        Hs_record{1,i2} = H(sumPara(i2)+1:sumPara(i2+1),sumN(i2-1)+1:sumN(i2));
    end
    Hp_record(1,:) = Hc_record; Hp_record{2,1} = H(sumPara(1)+1:sumPara(2),1:sumN(1));
    for i3 = 2:numN
        Hp_record{i3+1,i3} = H(sumPara(i3)+1:sumPara(i3+1),sumN(i3-1)+1:sumN(i3));
    end
          
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Historical information
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    HIS.niter = 0;
    HIS.t = 0;
    HIS.f = norm(V-W*H,'fro')^2;
    HIS.p = init_delta;   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    elapse = cputime;
    for iter = 1:Iter
        Wc = Wp_record{1,1};
        % updata Wc,Hc1,Hc2
        CX_record = cell(1,numN); CX_record{1,1} = max(X_record{1,1}-Wp_record{1,2}*Hp_record{2,1},0);
        for i6 = 2:numN
            CX_record{1,i6} = max(X_record{1,i6}-Wp_record{1,i6+1}*Hp_record{i6+1,i6},0);
        end
        
        CX = [];Hc = [];
        for i7 = 1:numN
            CX = [CX,CX_record{1,i7}];
            Hc = [Hc,Hp_record{1,i7}];
        end
        
        [Wp_record{1,1},Hc,iterWHc,~,~] = NeNMF(CX,vecPara(1),'max_iter',MaxIter,'min_iter',MinIter,'max_time',MaxTime,'tol',tol,'w_init',Wc,'h_init',Hc);
        Hp_record{1,1} = Hc(:,1:sumN(1));
        for i8 = 2:numN
            Hp_record{1,i8} = Hc(:,sumN(i8-1)+1:sumN(i8));
        end
        % update Ws1,Hs1,Ws2,Hs2......
        iterWHs_record = zeros(1,numN);
        for i9 = 1:numN
            CCX = max(X_record{1,i9}-Wc*Hp_record{1,i9},0);
            [Wp_record{1,i9+1},Hp_record{i9+1,i9},iterWHs_record(i9),~,~] = NeNMF(CCX,vecPara(i9+1),'max_iter',MaxIter,'min_iter',MinIter,'max_time',MaxTime,'tol',tol,'w_init',Wp_record{1,i9+1},'h_init',Hp_record{i9+1,i9});
        end
        % put all parts together
        W = [];
        for i10 = 1:numN+1
            W = [W,Wp_record{1,i10}];
        end
        
        H_record = cell(numN+1,1);H=[];
        for t4 = 1:numN+1
            H_record{t4,1} = [];
            for t5 = 1:numN
                H_record{t4,1} = [H_record{t4,1},Hp_record{t4,t5}];
            end
        end
        
        for t6 = 1:numN+1
            H = [H;H_record{t6,1}];
        end
        
        lambda = diag(sum(W)); W = W*pinv(lambda); H = lambda*H;
        % record the update result
        Wp_record = cell(1,numN+1); Wp_record{1,1} = W(:,1:sumPara(1));
        for j1 = 2:numN+1
            Wp_record{1,j1} = W(:,sumPara(j1-1)+1:sumPara(j1));
        end
        Hc_record = cell(1,numN); Hc_record{1,1} = H(1:sumPara(1),1:sumN(1));
        for j2 = 2:numN
            Hc_record{1,j2} = H(1:sumPara(1),sumN(j2-1)+1:sumN(j2));
        end
        Hp_record(1,:) = Hc_record;Hp_record{2,1} = H(sumPara(1)+1:sumPara(2),1:sumN(1));
        for j3 = 2:numN
            Hp_record{j3+1,j3} = H(sumPara(j3)+1:sumPara(j3+1),sumN(j3-1)+1:sumN(j3));
        end
        % stopping tolerance
        WtW = W'*W; WtV = W'*V;
        HHt = H*H'; HVt = H*V';
        GradH = WtW*H-WtV; GradW = W*HHt-HVt';
        delta = GetStopCriterion(STOP_RULE,[W',H],[GradW',GradH]);
        HIS.niter = HIS.niter+iterWHc+sum(iterWHs_record);
        
        
        % Output running detials
        
        HIS.f = [HIS.f,norm(V-W*H,'fro')^2];
        HIS.t = [HIS.t,cputime-elapse];
        HIS.p = [HIS.p,delta];
        if (verbose == 2) && (rem(iter,10) == 0),
            fprintf('%d:\tstopping criteria = %e,\tobjective value = %f,\tElapse = %f.\n', iter,delta/init_delta,HIS.f(end),HIS.t(end));
        end
        
        % Stopping condition
        if delta <= tol*init_delta
            break;
        end
        
        if rem(iter,20) == 0
            delta_ob = abs(HIS.f(iter)-HIS.f(iter-10))/HIS.f(iter);
        else
            delta_ob = 1;
        end
        
        if delta_ob <= tol
            break;
        end
    end
    elapse = cputime-elapse;
    if verbose == 2,
        fprintf('\nFinal Iter = %d,\tFinal Elapse = %f.\n', iter,elapse);
    end
end

    
    
    
    
 

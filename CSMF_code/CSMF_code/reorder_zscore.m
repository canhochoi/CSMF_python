function [X,W,H,IndexW_record,IndexH_record] = reorder_zscore(W,H,cutoff1,cutoff2,vecN,vecPara)
% order W and H of te result of CSMF
%<Input>
%        W,H: the result of CSMF
%        cutoff1, cutoff2: the z-score cutoff of W and H, respectively
%        vecN : The vector of mutiple sample sets
%        vecPara: The vector of target low-rank
%<Output>
%        X,W,H:the ordered of X,W,H
%        IndexW_record, IndexH_record: record the order index of each
%        column of W and each row of H
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = size(W,1);
% record sum of vecN
numN = length(vecN); sumN = zeros(1,numN); sumN(1) = vecN(1);
for i1 = 2:numN
    sumn = 0;
    for j1 = 1:i1
        sumn = sumn+vecN(j1);
    end
    sumN(i1) = sumn;
end
% record sum of vecPara
sumPara = zeros(1,numN+1); sumPara(1) = vecPara(1);
for i2 = 2:numN+1
    sumn = 0;
    for j2 = 1:i2
        sumn = sumn+vecPara(j2);
    end
    sumPara(i2) = sumn;
end
% record H
H0_record = cell(1,numN); Hc = H(1:sumPara(1),1:sumN(1)); Hs = H(sumPara(1)+1:sumPara(2),1:sumN(1)); H0_record{1,1} = [Hc;Hs];
for i4 = 2:numN
    Hc = H(1:sumPara(1),sumN(i4-1)+1:sumN(i4)); Hs = H(sumPara(i4)+1:sumPara(i4+1),sumN(i4-1)+1:sumN(i4)); H0_record{1,i4} = [Hc;Hs];
end

% commpute each mean and standard error
MW  =  mean(W,1); VW =  std(W,0,1); 
MH_record = cell(1,numN); VH_record = cell(1,numN);
for p1 = 1:numN
    MH_record{1,p1} = mean(H0_record{1,p1},2); VH_record{1,p1} = std(H0_record{1,p1},0,2);
end
% order W
sumP = sum(vecPara);
IndexW_record = cell(1,sumP); IndexW_record{1,1}  =  find(W(:,1) > MW(1)+cutoff1*VW(1));
n = length(IndexW_record{1,1}); A = 1:M; [~,~,c] = intersect(IndexW_record{1,1},A); A(c) = [];
W0 = W; W(1:n,:) = W0(IndexW_record{1,1},:); W(n+1:M,:) = W0(A,:);  
for j = 2:sumP
    IndexW_record{1,j}  =  find(W(:,j) > MW(j)+cutoff1*VW(j));
    s = 0;
    for k = 1:j-1
        s = s+length(IndexW_record{1,k});
    end
    [~,~,ir2] = intersect(1:s,IndexW_record{1,j});
    IndexW_record{1,j}(ir2) = []; n = length(IndexW_record{1,j});
    B = union(1:s,IndexW_record{1,j}); A = 1:M; [~,~,c] = intersect(B,A); A(c) = [];
    W0 = W; W(s+1:s+n,:) = W0(IndexW_record{1,j},:); W(s+n+1:M,:) = W0(A,:);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% order H
rank=vecPara(2:end)+vecPara(1);
IndexH_record = cell(1,numN); H_record = cell(1,numN); 
for i5 = 1:numN
    h = H0_record{1,i5}; index = rank(i5); 
    mh = MH_record{1,i5}; vh = VH_record{1,i5};
    r = cell(1,index); r{1,1}  =  find(h(1,:) > mh(1)+cutoff2*vh(1));
    n = length(r{1,1}); A = 1:vecN(i5); [~,~,c] = intersect(r{1,1},A); A(c) = [];
    h0 = H0_record{1,i5};
    h(:,1:n) = h0(:,r{1,1}); h(:,n+1:vecN(i5)) = h0(:,A);
    for j1 = 2:index
        r{1,j1}  =  find(h(j1,:) > mh(j1)+cutoff2*vh(j1));
        s = 0;
        for k1 = 1:j1-1
            s = s+length(r{1,k1});
        end
        [~,~,ir2] = intersect(1:s,r{1,j1}); r{1,j1}(ir2) = [];
        n = length(r{1,j1}); B = union(1:s,r{1,j1}); A = 1:vecN(i5); [~,~,c] = intersect(B,A); A(c) = [];
        h0 = h; h(:,s+1:s+n) = h0(:,r{1,j1}); h(:,s+n+1:vecN(i5)) = h0(:,A);
    end
    IndexH_record{1,i5} = r; H_record{1,i5} = h;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% combine H
Hp_record = cell(numN+1,numN);
for q2 = 1:numN
    Hp_record{1,q2} = H_record{1,q2}(1:vecPara(1),:);
end
for q3 = 1:numN
    for k1 = 1:numN
        Hp_record{q3+1,k1} = zeros(vecPara(q3+1),vecN(k1));
    end
end
for s1 = 1:numN
    Hp_record{s1+1,s1} = H_record{1,s1}(sumPara(1)+1:end,:);
end

Hall_record = cell(numN+1,1); H = [];
for t1 = 1:numN+1
    Hall_record{t1,1} = [];
    for t2 = 1:numN
        Hall_record{t1,1} = [Hall_record{t1,1},Hp_record{t1,t2}];
    end
end

for t3 = 1:numN+1
    H = [H;Hall_record{t3,1}];
end
X = W*H;


    
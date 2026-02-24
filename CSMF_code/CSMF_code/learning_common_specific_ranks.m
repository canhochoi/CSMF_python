function vecPara  =  learning_common_specific_ranks(NeNMF_result,minK,vecParas,cutoff)
% learning common and specific ranks
%<Input>
%        NeNMF_result: the output of NeNMF_data.m
%        minK: the low rank
%        vecParas: Target low-rank selcted by stability distance
%        cutoff: cutoff for controling common rank

%<Output>
%        vecPara : Obtained common and specific ranks

% Note: another file 'performancen.m' should be put in the same directory as this file.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Detect the best solution of NeNMF_result and normalization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numN = length(vecParas);
NeNMF_result_vecParas = cell(1,numN);
for i = 1:numN
    NeNMF_result_vecParas{1,i} = NeNMF_result{1,i}{1,vecParas(i)-minK+1};
end
W_record = cell(1,numN);
for i = 1:numN
    best_performance = performancen(NeNMF_result_vecParas{1,i},5,[vecParas(i),zeros(1,numN-1)]);
    W_record{1,i} = best_performance.W;
    % normalization
    lambda = diag(sum(W_record{1,i})); W_record{1,i} = W_record{1,i}*pinv(lambda);
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the common  part Wc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W = [W_record{1,1},W_record{1,2}];
a = corr(W);
for i = 1:size(a,1);
    for j = 1:i
        a(i,j) = 0;
    end
end
[idx,idy] = find(a>cutoff);
corr_record = cell(1,size(a,1));
for i = 1:size(a,1)
    corr_record{1,i} = [idy(idx == i);idx(idy == i)];
end
% Check whether Wc derives from specific part according to the value of Hc
flag = zeros(size(a,1),2);
for i = 1:size(a,1)
    s = union(corr_record{1,i},i);
    if ~isempty(find(s <= vecParas(1)))
        flag(i,1) = 1;
    end
    if ~isempty(find(s <= vecParas(1)+vecParas(2)))
        idd = s(find(s <= vecParas(1)+vecParas(2)));
        if ~isempty(find(idd > vecParas(1)))
            flag(i,2) = 1;
        end
    end
end
index = find(sum(flag,2) == 2);
if ~isempty(index)
    corr_record_new = corr_record(index);
    s = index;
    for i = 1:length(index)
        s = union(index,corr_record_new{1,i});
    end
    s1 = setdiff(s,vecParas(1)+1:vecParas(1)+vecParas(2));
    s2 = setdiff(s,1:vecParas(1));
    W1new = W_record{1,1}(:,s1); W2new = W_record{1,2}(:,s2-vecParas(1));
    C = corr(W1new,W2new); C1 = ones(size(C)); D = C1-C; [M,~]  =  Hungarian(D); [c1,c2] = find(M == 1);
    Wc = (W1new(:,c1)+W2new(:,c2))/2;
    nc = length(c1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 3:numN
        C = corr(Wc,W_record{1,i}); C1 = ones(size(C)); D = C1-C; [M,~]  =  Hungarian(D); [c1,c2] = find(M == 1); w = zeros(1,length(c1));
        for r = 1:length(c1)
            w(r) = C(c1(r),c2(r));
        end
        id = find(w > cutoff);
        if ~isempty(id)
            Wc = (Wc(:,c1(id))+W_record{1,i}(:,c2(id)))/2;
        else
            break;
        end
        nc = length(id);
    end
else
    nc = 0;
end
vecPara = [nc,vecParas-nc];
    

function stability_top_record = staNMF_rank(NeNMF_result,minK,maxK,repeat)
% give out ranks according to the top 10,20,30 stability
%<Input>:
%NeNMF_result: the result of NeNMF_data
%minK,maxK:low rank range
%repeat: times of applying NeNMF on each data matrix with each rank
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
size2 = size(NeNMF_result,2); K = minK: maxK; m = length(K);
NeNMF_top_record = cell(1,3);
for t = 1:3
    NeNMF_result_top = cell(1,size2);
    for j = 1:size2
        record = NeNMF_result{1,j};
        result_new = cell(1,m);
        for k = 1:m
            re = record{1,k};
            X_corr = zeros(1,repeat);
            for s = 1:repeat
                X_corr(s) = X_difference_corr(re{1,s}.W);
            end
            %sort X_corr
            [~,index] = sort(X_corr);
            re_new = re(index(1:10*t));
            result_new{1,k} = re_new;
        end
        NeNMF_result_top{1,j} = result_new;
    end
    NeNMF_top_record{1,t} = NeNMF_result_top;
end
stability_top_record = cell(1,size2);
for j = 1:size2
    record = zeros(4,m);% the fourth row of record is the average value.
    for k = 1:3
        result = NeNMF_top_record{1,k}{1,j};
        for t = 1:m
            B = result{1,t};
            record(k,t) = stability_result(B);
        end
        record(4,:) = sum(record(1:3,:))/3;
        stability_top_record{1,j} = record;
    end
end

% sum of correlation of itself
function X_difference_value = X_difference_corr(X)
%index = X_difference_matrix<0;X_difference_matrix(index) = X_difference_matrix(index)+1;
n = size(X,2);
if sum(X(:)) == 0
    X_difference_matrix = zeros(n,n);
else
    X_difference_matrix = corr(X);
end
X_difference_value = 0;
for i = 1:n
    for j = i+1:n
        X_difference_value = X_difference_value+X_difference_matrix(i,j);
    end
end

% compute stability
function Sta = stability_result(result)
n = size(result,2);D = zeros(n,n);
for i = 1:n
    W1 = result{1,i}.W;
    for j = i+1:n
        W2 = result{1,j}.W;
        A = corr(W1,W2);
        [D(i,j),~,~] = amariMaxError(A);   
    end
end
Sta = 2*sum(D(:))/(n*(n-1));

% A version of Amari error
% Computed as
%   1/(2K)*(sum_j (1 - max_{ij} A_{ij}) + sum_i (1 - max_{ij} A_{ij}))
function [distM,rowTemp,colTemp] = amariMaxError(A)
[n,m] = size(A);
if n == 1 && m == 1
    distM = double(A == 0);
    rowTemp = 0;
    colTemp = 0;
    return;
end

maxCol = max(abs(A),[],1);
colTemp0 = 1 - maxCol;
colTemp = mean(colTemp0);

maxRow = max(abs(A),[],2);
rowTemp0 = 1 - maxRow;
rowTemp = mean(rowTemp0);
distM = (rowTemp + colTemp)/2;
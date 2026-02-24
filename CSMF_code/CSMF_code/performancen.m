function best_performance = performancen(result,err_n,vecPara)
% Choose the best solution from the result
%<Input>
%       result  [cell]  set of result after run many times
%       err_n  the number used for selecting  top err_n result with little error
%       vecPara  low rank
%<Output>
%       best_performance the best one in result
% select result with little error and litter similiraty
n = size(result,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% screen err_n results by considering err 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err = zeros(1,n);
for q = 1:n
    err(q) = result{1,q}.err;
end
[~,err_index] = sort(err);
top_result = cell(1,err_n);
for p = 1:err_n
    top_result{1,p} = result{1,err_index(p)};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% screen best performance by difference measured by pearson correlation between W
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numP = length(vecPara);
sumPara = zeros(1,numP);
sumPara(1) = vecPara(1);
for i1 = 2:numP
    sumn = 0;
    for j1 = 1:i1
        sumn = sumn+vecPara(j1);
    end
    sumPara(i1) = sumn;
end
%%%%% begin from the first nonzero
flag = 1;
if sumPara(1) == 0
    for i = 2:numP
        if sumPara(i) ~= 0
            flag = i;
            break;
        end
    end
end
%%%% begin from flag
sumPara = sumPara(flag:end);numP = length(sumPara);
W_difference_corr = zeros(1,err_n);
for m = 1:err_n
    W = top_result{1,m}.W;
    W_record = cell(1,numP);
    W_record{1,1} = W(:,sumPara(1));
    for i2 = 2:numP
        W_record{1,i2} = W(:,sumPara(i2-1)+1:sumPara(i2));
    end
    % correlation matrix
    W_diff_matr = zeros(numP,numP);
    for i3 = 1:numP
        W1 = W_record{1,i3};
        for i4 = i3:numP
            W2 = W_record{1,i4};
            if ~isempty(W1) && ~isempty(W2)
                W_diff_matr(i3,i4) = W1_W2_difference_corr(W1,W2)+X_difference_corr(W1)+X_difference_corr(W2);
            elseif ~isempty(W1) && isempty(W2)
                W_diff_matr(i3,i4) = X_difference_corr(W1);
            elseif isempty(W1) &&  ~isempty(W2)
                W_diff_matr(i3,i4) = X_difference_corr(W2);
            else
                W_diff_matr(i3,i4) = 0;
            end
            
        end
    end
    W_difference_corr(1,m) = sum(W_diff_matr(:));
end
[~,Ws_index] = sort(W_difference_corr);
best_performance = top_result{1,Ws_index(1)};

% sum of correlation value of W1 and W2
function X_difference = W1_W2_difference_corr(W1,W2)
n1 = size(W1,2);n2 = size(W2,2);
if sum(W1(:)) == 0 || sum(W2(:)) == 0
    X_difference = zeros(n1,n2);
else
    X_difference = corr(W1,W2);
end
%index = X_difference<-0.5;X_difference(index) = X_difference(index)+1;index = -0.5< = X_difference<0;X_difference(index) = -X_difference(index);
X_difference = sum(sum(X_difference));

% sum of correlation 
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

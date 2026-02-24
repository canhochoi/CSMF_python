function NeNMF_result = NeNMF_data(X,vecN,minK,maxK,repeat)
% run NMF by NeNMF method on each view data matrix with the rank from minK
% to maxK
%<Input>:
%X: each view data matrix
%vecN: A vector recording the number of samples of each view data
%minK, maxK: low rank range
%repeat: times of applying NeNMF on each data matrix with each rank

%<Output>:
% NeNMF_result: record result of NeNMF on each view data matrix with the ranks from minK to maxK

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% open multiple kernels to run
poolobj = gcp('nocreate'); 
if isempty(poolobj)
    parpool('local',poolobj.NumWorkers);
end
% compute sum of the sample numbers
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
% run NeNMF on each view data matrix
NeNMF_result = cell(1,length(vecN));
K = minK:maxK;
for k0 = 1:numN
    if k0 == 1
        x = X(:,1:sumN(k0));
    else
        x = X(:,sumN(k0-1)+1:sumN(k0));
    end
    R = cell(1,length(K));
    for k1 = 1:length(K)
        result = cell(1,repeat);
        parfor k2 = 1:repeat
            [W,H,iter,~,HIS] = NeNMF(x,K(k1),'MAX_ITER',500);
            result{1,k2}.W = W; result{1,k2}.H = H; result{1,k2}.err = abs(HIS.f(end)); result{1,k2}.iter = iter;
        end
        R{1,k1} = result;
    end
    NeNMF_result{1,k0} = R;
end
delete(gcp('nocreate'))
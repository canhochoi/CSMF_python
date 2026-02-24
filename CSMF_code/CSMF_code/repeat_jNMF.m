function [result,best_performance] = repeat_jNMF(X,vecN,vecPara,cut,repeat,err_n)
% run jcNMF repeat times on X
%<Input>
%        X (M x multi-dimension): contain more than two data matrices
%        vecN : The vector of mutiple sample sets
%        vecPara: The vector of target low-rank
%        cut: the thrshold to select significantly element
%        repeat: times of applying jNMF+ on each data matrix with each ran
%        err_n: the number used in select the top err_n minmum error
%        solution
%<Output>
%       result: the repeat times result of jNMF+ 
%       best_performance: the best solution of the result
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result = cell(1,repeat);
% open many kernels 
poolobj = gcp('nocreate'); 
if isempty(poolobj)
    parpool('local',poolobj.NumWorkers);
end
parfor i = 1:repeat
    [W,H,err,elapse] = jNMF(X,vecN,vecPara,cut);
    result{1,i}.W = W;
    result{1,i}.H = H;
    result{1,i}.err = err;
    result{1,i}.elapse = elapse;
end
delete(gcp('nocreate'))
% we select the best solution
best_performance = performancen(result,err_n,vecPara);
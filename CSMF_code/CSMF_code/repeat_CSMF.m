function [result,best_performance] = repeat_CSMF(X,vecN,vecPara,repeat,Winit,Hinit,tho,err_n)
% run CSMF algrorithm on X with repeat times
%<Input>
%        X (M x multi-dimension): contain more than two data matrices
%        vecN : The vector of mutiple sample sets
%        vecPara: The vector of target low-rank
%        repeat: times of applying CSMF on each data matrix with each rank
%        Winit, Hinit: initial value of W and H
%        tho: the standard deviation of Gassian disturbance to the initial
%        value Winit
%        err_n: the number used in select the top err_n minmum error
%        solution
%<Output>
%       result: the repeat times result of CSMF
%       best_performance: the best solution of the result
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(Winit) && isempty(Hinit)
    init = 0;
else
    init = 1;
end
result = cell(1,repeat);
% run CSMF
for i = 1:repeat
    if init == 0
    [W,H,iter,elapse,HIS] = CSMF(X,vecN,vecPara);
    else
        % add disturbance to the initial value, in oder to prevent the
        %result being lost in some bad local solution
        Winit = max(Winit+tho*rand(size(Winit)),0);
        lambda = diag(sum(Winit)); Winit = Winit*pinv(lambda); Hinit = lambda*Hinit;
        [W,H,iter,elapse,HIS] = CSMF(X,vecN,vecPara,'w_init',Winit,'h_init',Hinit);
    end
    result{1,i}.W = W;
    result{1,i}.H = H;
    result{1,i}.err = abs(HIS.f(end));
    result{1,i}.iter = iter;
    result{1,i}.elapse = elapse;
end
% we select the best solution
best_performance = performancen(result,err_n,vecPara);
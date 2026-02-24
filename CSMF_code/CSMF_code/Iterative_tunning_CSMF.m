function ItCSMF_record = Iterative_tunning_CSMF(X,W,H,vecN,vecPara,inner_iter,n1,H_cutoff,cor_cutoff,outer_iter)
%tuning the result of CSMF iteratively
%<Input>
%            X:(M x multi-dimension): contain more than two data matrices
%            W,H: the result matrix you want to adjust
%            vecPara:  the rank
%            inner_iter: the times of running CSMF with tunned ranks
%            n1:  the paremeter in performance
%            vecN: sample number
%            H_cutoff: cutoff for deciding  row of H  in common or specific
%            cor_cutoff: cutoff of corr relationship matrix
%            outer_iter: the iter of running tunning CSMF algorithm
%<Output>
%         ItCSMF_record: The result of tunning CSMF algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ItCSMF_record = cell(1,outer_iter);
for i = 1:outer_iter
    [W_adj,H_adj,extend_class,vecPara_adj,maxcor,flag_Result,mrdh] = tunning_CSMF(X,W,H,vecN,vecPara,inner_iter,n1,H_cutoff,cor_cutoff);
     ItCSMF_record{1,i}.W = W_adj;ItCSMF_record{1,i}.H = H_adj;ItCSMF_record{1,i}.extend_class = extend_class;ItCSMF_record{1,i}.vecPara = vecPara_adj;ItCSMF_record{1,i}.maxcor = maxcor;ItCSMF_record{1,i}.flag_Result = flag_Result;ItCSMF_record{1,i}.mrdh = mrdh;   
    if isempty(extend_class) && (isempty(mrdh) && sum(vecPara == vecPara_adj) == length(vecPara))
        break;
    end
    W = W_adj;H = H_adj;vecPara = vecPara_adj;
end
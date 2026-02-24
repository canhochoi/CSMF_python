function [AUC,AUPR] = compute_accuracy(W,H,W0,H0)
% W0,H0: golden standard basis and coefficient matrices
% W,H: basis and coefficient matrices obtianed by CSMF
X0 = W0*H0; X0(X0 > 0) = 1;% binary
X = W*H;
[AUC,AUPR,~,~,~,~]=ROCcompute(X(:),X0(:),0);

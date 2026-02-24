%This code is used to obtain AUC and AUPR;
%[AUC,AUPR,Acc,Sen,Spe,Pre]=ROCcompute(predictList,trueList,plotOption);
%InPut:
% predictList - prediction score matrix or vector by each models;
% trueList - sign matrix or vector to record whehter or not existing association
% corresponding to each drug-ADR pair in predictList;
% plotOption - to choose whether ploting ROC or not, if plotOption=1, ploting ROC;
%OutPut:
% (omit)
function [AUC,AUPR,Acc,Sen,Spe,Pre]=ROCcompute(predictList,trueList,plotOption)
 if size(predictList,2)>1
 predictList=reshape(predictList,numel(predictList),1);
 trueList=reshape(trueList,numel(trueList),1);
 end

 low=min(predictList);
 high=max(predictList);
 threshold=linspace(high,low,100);

 Sen=zeros(1,100); Spe=zeros(1,100); Pre=zeros(1,100);
 Acc=zeros(1,100);
 for I=1:100
 Vector=zeros(numel(predictList),1);
 v=predictList>=threshold(I);
 Vector(v)=1;

 tp=sum(Vector==1&trueList==1);
 tn=sum(Vector==0&trueList==0);
 np=sum(Vector==1&trueList==0);
 nn=sum(Vector==0&trueList==1);

 Sen(I)=tp/(tp+nn);

 Spe(I)=tn/(tn+np);

 if tp+np==0
 Pre(I)=1;
 else
 Pre(I)=tp/(tp+np);
 end

 Acc(I)=(tn+tp)/(tn+tp+np+nn);
 end

 Sen=[0,Sen];Spe=[1,Spe];Pre=[1,Pre];
 Acc=[sum(trueList==0)/length(trueList) Acc];

 AUC=abs(trapz(1-Spe,Sen));
 AUPR=abs(trapz(Sen,Pre));

 if plotOption==1
 plot(1-Spe,Sen);
 axis([-0.01 1.00 0 1.01]);
 xlabel('1-Spe');
 ylabel('Sen');

 figure;
 plot(Sen,Pre);
 axis([0 1.01 0 1.01]);
 xlabel('Sen');
 ylabel('Pre');
 end
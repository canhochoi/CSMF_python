function [X,W,H] = simulation_CSMF(Mint,vecP,coph,vecPara,vecN,tho)
% simulate multi-view data
%<Input>
%Mint: Initial row dimension of the data
%vecP: A vector recording the paramter of Binomial distribution
%coph: A number to control the overlap of adjacent columns of basis matrix
%vecPara: A vector recording the rank information
%vecN: A vector recording the number of samples of each view data
%tho: The standard deviation of Gaussian noise

%<Output>
%X: The data matrix composed of multiview data matrices
%W, H: The basis matrix and coefficient matrix composed of both common
%part and specific part

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simulate W
index = sum(vecPara);
M = max(index*10,Mint);
W = zeros(M,index);
for j = 1:M
    for k = 1:index
        if j >=  1+(k-1)*(10-coph) && j <= 10+(k-1)*(10-coph)
            W(j,k) = 1;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simulate H by y  =  random('Binomial',n,p,m,n)
% simulate common part firstly, then simulate specific part
Hc_record = cell(1,length(vecN));
for i = 1:length(vecN)
    Hc_record{1,i} = random('Binomial',1,vecP(i),vecPara(1),vecN(i));
end
Hs_record = cell(length(vecN));
for i = 1:length(vecN)
    for j = 1:length(vecN)
        if i == j
            Hs_record{i,j} = random('Binomial',1,vecP(i),vecPara(i+1),vecN(j));
        else
            Hs_record{i,j} = zeros(vecPara(i+1),vecN(j));
        end
    end
end
Hc = [];
for i = 1:length(vecN)
    Hc = [Hc,Hc_record{1,i}];
end
Hss = cell(length(vecN),1);
for i = 1:length(vecN)
    hs = [];
    for j = 1:length(vecN)
        hs = [hs,Hs_record{i,j}];
    end
    Hss{i,1} = hs;
end
Hs = [];
for i = 1:length(vecN)
    Hs = [Hs;Hss{i,1}];
end
H = [Hc;Hs];
X = max(W*H+tho*randn(M,sum(vecN)),0);


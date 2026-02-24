function [W_adj,H_adj,extend_class,vecPara_adj,maxcor,flag_Result,mrdh] = tunning_CSMF(X,W,H,vecN,vecPara,inner_iter,n1,H_cutoff,cor_cutoff)
% fine tune the result of CSMF in one step
%<Input>
%            X:(M x multi-dimension): contain more than two data matrices
%            W,H: the result matrix you want to adjust
%            vecPara:  the rank
%            inner_iter: the times of running CSMF with tunned ranks
%            n1:  the paremeter in performance
%            vecN: sample number
%            H_cutoff: cutoff for deciding  row of H  in common or specific
%            cor_cutoff: cutoff of corr relationship matrix
%<Output>
%           W_adj,H_adj:   the last adjusted result
%           extend_class:    record which column of W to adjust
%           vecPara_adj:  the last rank
%           max_cor:    the max correlation in relationship matrix
%           flag_Result: record which variation
%           mrdh: record the row of H should be changed 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numN = length(vecN);
% record sum of vecN
sumN = zeros(1,numN);
sumN(1) = vecN(1);
for i1 = 2:numN
    sumn = 0;
    for j1 = 1:i1
        sumn = sumn+vecN(j1);
    end
    sumN(i1) = sumn;
end
% record sum of vecPara
sumPara = zeros(1,numN+1);
sumPara(1) = vecPara(1);
for i2 = 2:numN+1
    sumn = 0;
    for j2 = 1:i2
        sumn = sumn+vecPara(j2);
    end
    sumPara(i2) = sumn;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% remove the empty row of Hc and the corresponding column of Wc
Hc = H(1:vecPara(1),:);
Hc_record = cell(1,numN); Hc_record{1,1} = Hc(:,1:sumN(1));
for i = 2:numN
    Hc_record{1,i} = Hc(:,sumN(i-1)+1:sumN(i));
end
Wc = W(:,1:vecPara(1));
%%%%%% remove the nearly null part,if there is null part in not all view
%%%%%% data, which column of bais matrix should be in specific part
colnorh = zeros(vecPara(1),numN);
for i = 1:numN
    colnorh(:,i) = sum(Hc_record{1,i},2)/vecN(i);
end
avg_colnorh = sum(colnorh(:))/(vecPara(1)*numN);
record = zeros(vecPara(1),numN);
for j = 1:numN
    record(:,j) = colnorh(:,j)/avg_colnorh<H_cutoff;%H_cutoff should be small value such as 0.1£¬0.01
end
rdh = find(sum(record,2) == 0);%%%%%%%retain
urdh = find(sum(record,2) == numN);%%%%%%%remove
Wc(:,urdh) = []; Hc(urdh,:) = []; vecPara(1) = vecPara(1)-length(urdh);
%%%%% other circumstance:the part should be in specific part
Hs_record = cell(1,numN); Hs_record{1,1} = H(sumPara(1)+1:sumPara(2),1:sumN(1));
for i = 2:numN
    Hs_record{1,i} = H(sumPara(i)+1:sumPara(i+1),sumN(i-1)+1:sumN(i));
end
Ws_record = cell(1,numN);
for i = 1:numN
    Ws_record{1,i} = W(:,sumPara(i)+1:sumPara(i+1));
end
mrdh = setdiff(1:vecPara(1),union(rdh,urdh));
if ~isempty(mrdh)
    record_m = record(mrdh,:); Wc_s = Wc(:,mrdh);Hc_s = Hc(mrdh,:); Wc(:,mrdh) = [];Hc(mrdh,:) = []; vecPara(1) = vecPara(1)-length(mrdh);
    for j1 = 1:length(mrdh)
        rc2s = find(record_m(j1,:) == 0);
        if rc2s(1) == 1
            Hs_record{1,1} = [Hs_record{1,1};Hc_s(j1,1:sumN(1))];
            Ws_record{1,1} = [Ws_record{1,1},Wc_s(:,j1)]; vecPara(2) = vecPara(2)+1;
            for j2 = 2:length(rc2s)
                Hs_record{1,rc2s(j2)} = [Hs_record{1,rc2s(j2)};Hc_s(j1,sumN(rc2s(j2)-1)+1:sumN(rc2s(j2)))];
                Ws_record{1,rc2s(j2)} = [Ws_record{1,rc2s(j2)},Wc_s(:,j1)];
                vecPara(rc2s(j2)+1) = vecPara(rc2s(j2)+1)+1;
            end
        else
            for j2 = 1:length(rc2s)
                Hs_record{1,rc2s(j2)} = [Hs_record{1,rc2s(j2)}; Hc_s(j1,sumN(rc2s(j2)-1)+1:sumN(rc2s(j2)))];
                Ws_record{1,rc2s(j2)} = [Ws_record{1,rc2s(j2)},Wc_s(:,j1)];
                vecPara(rc2s(j2)+1) = vecPara(rc2s(j2)+1)+1;
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% remove the nearly empty row of Hs and the corresponding column of Ws
for i = 1:numN
    if vecPara(i+1) > 0
        colnorhs = sum(Hs_record{1,i},2)/vecN(i); records = colnorhs/avg_colnorh < H_cutoff;
        rs = find(records == 1);
        Ws_record{1,i}(:,rs) = []; Hs_record{1,i}(rs,:) = []; vecPara(i+1) = vecPara(i+1)-length(rs);
    end
end
% vecPara may be vary
sumPara = zeros(1,numN+1);
sumPara(1) = vecPara(1);
for i2 = 2:numN+1
    sumn = 0;
    for j2 = 1:i2
        sumn = sumn+vecPara(j2);
    end
    sumPara(i2) = sumn;
end
vecParas = vecPara(2:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%compute correlation matrix of W
%%%% represent W
W = Wc;
for i = 1:numN
    W = [W,Ws_record{1,i}];
end
%%%%H
Hsrow_record = cell(1,numN); Hsrow_record{1,1} = [Hs_record{1,1},zeros(vecParas(1),sumN(numN)-sumN(1))];
for i = 2:numN
    Hsrow_record{1,i} = [zeros(vecParas(i),sumN(i-1)),Hs_record{1,i},zeros(vecParas(i),sumN(numN)-sumN(i))];
end
H = Hc;
for i = 1:numN
    H = [H;Hsrow_record{1,i}];
end
Mcorr = corr(W);
%%%%% upper triangular matrix
for i = 1:sumPara(numN+1)
    for j = 1:i
        Mcorr(i,j) = 0;
    end
end
% choose the value bigger than cor_cutoff other than the diagonal of the matrix
index = find(Mcorr > cor_cutoff);
if ~isempty(index)
    Mo = unique(roundn(Mcorr(index),-4)); maxcor = max(Mo);
    Idx = cell(length(Mo),1); Idy = cell(length(Mo),1);
    for i = 1:length(Mo)
        [Idx{i,1},Idy{i,1}] = find(roundn(Mcorr,-4) == Mo(i));
    end
    varied_Id = cell(1,length(Mo)); nid = zeros(1,length(Mo));
    for j = 1:length(Mo)
        ID = cell(1,length(Idx{j,1})); nid(j) = length(Idx{j,1});
        for k = 1:length(Idx{j,1})
            ID_ele = zeros(1,2); ID_ele(1) = Idx{j,1}(k); ID_ele(2) = Idy{j,1}(k); ID{1,k} = ID_ele;
        end
        varied_Id{1,j} = ID;
    end
    %%%%%%put the change together
    sumnid = zeros(1,length(Mo)); sumnid(1) = nid(1);
    for ii = 2:length(Mo)
        l = 0;
        for iii = 1:ii
            l = l+nid(iii);
        end
        sumnid(ii) = l;
    end
    varied_id = zeros(sumnid(length(Mo)),2);
    for j1 = 1:sumnid(1)
       varied_id(j1,:) = varied_Id{1,1}{1,j1};
    end
    for iv = 2:length(Mo)
        for j2 = 1:nid(iv)
            varied_id(sumnid(iv-1)+j2,:) = varied_Id{1,iv}{1,j2};
        end
    end
    extend_class = cell(1,size(varied_id,1));
    for j = 1:size(varied_id,1)
        extend_class{1,j} = varied_id(j,:);
    end
    %%%%%compute the correlation value of extend_class
    corr_class = zeros(1,length(extend_class));
    for i = 1:length(extend_class)
        corr_class(i) = roundn(Mcorr(extend_class{1,i}(1),extend_class{1,i}(2)),-4);
    end
else
    extend_class = []; corr_class = []; maxcor = max(Mcorr(:));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
flag_Result = 0;
while ~isempty(extend_class) && flag_Result == 0
    max_id = find(corr_class == max(corr_class)); change_label = extend_class{1,max_id(1)};
    labelx = change_label(1); labely = change_label(2); idx = min(labelx,labely); idy = max(labelx,labely);
    %%%%%%discuss the location of labelx,labely
    %%%%labelx and labely all in common part or a same view specific part
    single_class_record = zeros(1,numN+1);
    if idy < sumPara(1)
        single_class_record(1) = 1; flag_Result = 1;
    end
    for j1 = 2:numN+1
        if idx > sumPara(j1-1) && idy <= sumPara(j1);
            single_class_record(j1) = 1;
        end
    end
    single_class_flag = find(single_class_record(2:end) == 1);
    if ~isempty(single_class_flag)
        w = [W(:,idx),W(:,idy)]; h = [H(idx,:);H(idy,:)]; vecPara(single_class_flag+1) = vecPara(single_class_flag+1)-2;
        flag_Result = 2; change_index = [idx,idy];
    end
    %%%%%labelx in common part and labely in specific part
    common_specific_record = zeros(1,numN);
    if idx <= sumPara(1)
        for j2 = 1:numN
            if idy > sumPara(j2) && idy <= sumPara(j2+1)
                common_specific_record(j2) = 1;
            end
        end
    end
    common_specific_flag = find(common_specific_record == 1);
    if ~isempty(common_specific_flag)
        w = [W(:,idx),W(:,idy)]; h = [H(idx,:);H(idy,:)]; vecPara(1) = vecPara(1)-1; vecPara(common_specific_flag+1) = vecPara(common_specific_flag+1)-1;
        flag_Result = 3; change_index = [idx,idy];
    end
    if ~isempty(find(vecParas == 0))
        flag_Result = 0; break;
    elseif isempty(single_class_flag) && isempty(common_specific_flag)
        %%%%%labelx and labey in different view specific parts
        %%%%%if there is a specific rank becomes zero, break from the
        %%%%%circulation
        if numN == 2
            w = [W(:,idx),W(:,idy)]; h = [H(idx,:);H(idy,:)];
            vecPara(2) = vecPara(2)-1; vecPara(3) = vecPara(3)-1;
            flag_Result = 4; change_index = [idx,idy];
        else
            labelx_record = zeros(1,numN); labely_record = zeros(1,numN);
            for j3 = 1:numN
                if idx > sumPara(j3) && idx <= sumPara(j3+1)
                    labelx_record(j3) = 1;
                end
                if idy > sumPara(j3) && idy <= sumPara(j3+1)
                    labely_record(j3) = 1;
                end
            end
            labelx_flag = find(labelx_record == 1); labely_flag = find(labely_record == 1); x_varied_id = varied_id(:,1); y_varied_id = varied_id(:,2);
            %%%%%%%find label high correlated with labelx and labely
            labelx_pair_index = union(x_varied_id(find(varied_id(:,1) == idx)),y_varied_id(find(varied_id(:,2) == labelx)));
            labely_pair_index = union(x_varied_id(find(varied_id(:,1) == idy)),y_varied_id(find(varied_id(:,2) == labely)));
            %%%%%%%%%%%%%%union the label highly correlated with both
            %%%%%%%%%%%%%%labelx and labely
            labelxy_pair_index = intersect(labelx_pair_index,labely_pair_index);
            if ~isempty(labelxy_pair_index)
                %%%%%any pair should be highly correlated
                corrxy = corr(W(:,labelxy_pair_index)); recordxy = corrxy > cor_cutoff;
                %%%%%find rows with value all equal to 1
                labelxy_pair_index = labelxy_pair_index(find(sum(recordxy,2) == length(labelxy_pair_index)));
            end
            %%%only focus on one of the most higly correlated label in each
            %%%specific part
            %%%record index and corr
            if ~isempty(labelxy_pair_index)
                labelxy_index_corr = zeros(length(labelxy_pair_index),3);
                for j4 = 1:length(labelxy_pair_index)
                    labelxy_index_corr(j4,1) = labelxy_pair_index(j4); labelxy_index_corr(j4,3) = max(roundn(Mcorr(idx,labelxy_pair_index(j4)),-4),roundn(Mcorr(idy,labelxy_pair_index(j4)),-4));
                    for j5 = 1:numN
                        if labelxy_pair_index(j4) > sumPara(j5) && labelxy_pair_index(j4) <= sumPara(j5+1)
                            labelxy_index_corr(j4,2) = j5;
                        end
                    end
                end
                labelxy_specific = unique(labelxy_index_corr(:,2));
                all_index = union(union(labelx_flag,labely_flag),labelxy_specific); all_index = sort(all_index); labelxy_index = labelxy_index_corr(:,1);
            else
                all_index = [];
            end
            if length(all_index) == numN
                change_index = zeros(1,numN); change_index(1:2) = [idx,idy]; corr_value = labelxy_index_corr(:,3);
                for j6 = 1:numN-2
                    stage = labelxy_specific(j6);
                    stage_index = find(labelxy_index_corr(:,2) == stage); stage_corr = corr_value(stage_index); stage_flag = find(stage_corr == max(stage_corr));
                    change_index(j6+2) = labelxy_index(stage_index(stage_flag(1)));
                end
                w = W(:,change_index); h = H(change_index,:);
                for j7 = 2:numN+1
                    vecPara(j7) = vecPara(j7)-1;
                end
                flag_Result = 4;
            else
                flag_Result = 0;
                %%%%%%which means not all numN specific satisfy high
                %%%%%%correlated, remove idx and idy and repeat again
                extend_class{1,max_id(1)} = []; corr_class(max_id(1)) = [];
                extend_class(cellfun(@isempty,extend_class)) = [];
            end
        end
    end
    if flag_Result ~= 0;
        break;
    end
end

%vecPara may change
sumPara = zeros(1,numN+1);
sumPara(1) = vecPara(1);
for i2 = 2:numN+1
    sumn = 0;
    for j2 = 1:i2
        sumn = sumn+vecPara(j2);
    end
    sumPara(i2) = sumn;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if flag_Result ~= 0
    wc0 = sum(w,2)/size(w,2); hc0 = sum(h,1)/size(h,1);
    wss = W; wss(:,change_index) = [];
    hss = H; hss(change_index,:) = [];
    x = w*h;
    vecPara0 = [1,zeros(1,numN)];
    result0 = cell(1,inner_iter);
    for i = 1:inner_iter
        [Wco,Hco,~,~,HIS] = CSMF(x,vecN,vecPara0,'W_INIT',wc0,'H_INIT',hc0);
        result0{1,i}.W = Wco; result0{1,i}.H = Hco; result0{1,i}.err = abs(HIS.f(end));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %choose the best one
    best_performanceo = performancen(result0,n1,vecPara0);
    Wo = best_performanceo.W; Ho = best_performanceo.H;
    x = Wo*Ho;
    X_new = X-x; X_new = max(X_new,0);
    %%%%%%compute the left part
    result = cell(1,inner_iter);
    for ii = 1:inner_iter
        [W,H,~,~,HIS] = CSMF(X_new,vecN,vecPara,'W_INIT',wss,'H_INIT',hss);
        result{1,ii}.W = W; result{1,ii}.H = H; result{1,ii}.err = abs(HIS.f(end));
    end
    %choose the best one
    best_performance = performancen(result,n1,vecPara);
    W_new = best_performance.W; H_new = best_performance.H;
    %disscuss the value of flag_Result
    if flag_Result == 2
        change_pa = zeros(1,numN+1); change_pa(single_class_flag+1) = 1;
        vecPara_adj = vecPara+change_pa;
        W_adj = [W_new(:,1:sumPara(single_class_flag)),Wo,W_new(:,sumPara(single_class_flag)+1:end)];
        H_adj = [H_new(1:sumPara(single_class_flag),:);Ho;H_new(sumPara(single_class_flag)+1:sumPara(numN+1),:)];
    else
        vecPara_adj = [vecPara(1)+1,vecPara(2:end)];
        W_adj = [Wo,W_new]; H_adj = [Ho; H_new];
    end
else
    W_adj = W; H_adj = H; extend_class = [];
    vecPara_adj = vecPara;
end
%%%%%normalization
lambda = diag(sum(W_adj)); W_adj = W_adj*pinv(lambda); H_adj = lambda*H_adj;
            
    
            
            
            
   
    
    
    
    
    
            
        

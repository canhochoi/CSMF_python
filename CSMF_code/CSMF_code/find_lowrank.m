function positions = find_lowrank(stability_top_record)
% select the maximum rank among the ranks corresponding to local minimum of the scores 
N = length(stability_top_record);
candidates = cell(4,N); values = cell(4,N);
for i = 1:N
    datas = stability_top_record{1,i};
    for j = 1:4
        [pks,pos] = findpeaks(-datas(j,:));
        if ~isempty(pos)
        candidates{j,i} = pos; values{j,i} = -pks;
        else
            [~,pos] = findpeaks(datas(j,:));
            if ~isempty(pos)
                % the dot before the peak
                candidates{j,i} = pos-1;
                values{j,i} = datas(j,candidates{j,i});
            end            
        end       
    end
    if sum(cellfun(@length,candidates(:,i))) == 0
        error(['Please extend the range of [minK,maxK] for the ',num2str(i),'-th data!'])
    end
end
% if there are at least two low-ebb, the position is effective.
candidates_new = cell(1,N); values_new = cell(1,N);
for i = 1:N
    location = []; 
    for j = 1:4
        location = [location,candidates{j,i}];
    end
    union_location = unique(location); 
    M = length(union_location);
    flag_matrix = zeros(4,M); union_values = zeros(4,M);
    for s = 1:M
        for t = 1:4
            Index = find(candidates{t,i} == union_location(s), 1);
            if ~isempty(Index)
                flag_matrix(t,s) = 1;
                union_values(t,s) = values{t,i}(Index);
            end
        end
    end
    sum_flags = sum(flag_matrix);
    ID = find(sum_flags >= 2);
    candidates_new{1,i} = union_location(ID);
    values_new{1,i} = sum(union_values(ID))./sum_flags(ID);
end
% find the maximun k    
positions = zeros(1,N);
for i = 1:N
    value_i = values_new{1,i}; candidate_i = candidates_new{1,i};
    mvalue = mean(stability_top_record{1,i}(:));
    positions(i) = max(candidate_i(value_i < mvalue));
end
        
    


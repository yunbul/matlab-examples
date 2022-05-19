function [data, mins, maxs, diff] = normalization(data)

% (x-min(x))/(max(x)-min(x))
cols = size(data,2)-1; % used to pass label column
mins = min(data(:,1:cols));
maxs = max(data(:,1:cols));
diff = maxs-mins; 

% eliminate division by zeros
for i=1:size(diff,2)
    if diff(i) == 0
        diff(i) = 1e-5;
    end
end

for i=1:size(data,1)
    data(i,1:cols) = (data(i,1:cols)-mins)./diff;
end

end
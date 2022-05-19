function [ Sh, Cs ] = GetSpread( X, C )

%reference to Ethem Alpaydin :
%find the most distant instance covered by that cluster and set
%sh to half its distance from the center.

% C : centroids, X : data points
H = size(C,1);
Cs = cell(H,1); %clusters jagged array
Sh = []; %spreads

%cluster data once with final centroids
for i=1:size(X,1)
        temp = 0;
        for j=1:H
            temp(j) = sqrt( sum(( X(i,:)-C(j,:) ).^2,2) );
        end
        % get index of closest center for this data point
        [dummy,ind]=min(temp);
        % add index of point to cluster
        Cs(ind,1) = {horzcat(cell2mat(Cs(ind,1)) ,i)};
end

%find furthest points belonging for centroids
for j=1:H
    C_arr = cell2mat(Cs(j,1)); %its easier to work on matrices
    max_dist = -1;
    %for each data point index in that cluster
    for i=1:size(C_arr,2) 
        dist = sqrt( sum(( X(C_arr(i),:)-C(j,:) ).^2,2) );
        if max_dist < dist
            max_dist = dist;
        end
    end
    %if there is only 1 point in cluster and centroid overlaps with that point
    %division by Sh produces NaN in rbf function (see GetGaussian.m)
    if max_dist == 0
        max_dist = 1e-10;
    end
    %set half of distance
    Sh = [Sh; max_dist/2];
end

end


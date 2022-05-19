function [C] = OnlineKMeans(X,H,alpha,animate,plotJ)

%initialize random centroids
temp=randperm(size(X,1)); %indices to select coordinates from columns of X
C=X(temp(1:H),:); %set random coordinates
J_history=[]; %cost : overall sum of datapoint <-> centroids distances
epoch=1; 

while(1)
    J_history(epoch) = 0; 
    C_temp = C; 
    %for every data point
    for i=1:size(X,1)
        temp = 0; %distances between this point to all centroids
        for j=1:H
            temp(j) = sqrt( sum(( X(i,:)-C(j,:) ).^2,2) ); %euclidian
        end
        % get index of closest center for this data point
        [dummy,ind]=min(temp);
        % update closest centroid : mi = mi + n(x ? mi)
        C_temp(ind,:)=C_temp(ind,:) + ( alpha * (X(i,:)-C_temp(ind,:)) );
        J_history(epoch)=J_history(epoch) + sum(temp);
    end
    
    %check average (euclidian) centroid displacement for convergence
    %with threshold 10^(-5)
    if sum( sqrt(sum( (C_temp-C).^2, 2 )) ) / H > 1e-5 
        C = C_temp; %update centroids
        % animate centroid updates
        if (nargin>3)
            figure(animate)
            scatter(X(:,1),X(:,2))
            hold on
            scatter(C(:,1),C(:,2),'filled')
            title(strcat('K=',num2str(H)))
            hold off
            pause(0.001) %~1 millis
        end
    else
        break; 
    end
    
% find number of changed centroid coordinates (absolute displacement)
% issue : converges slowly, thus used code above
%     if(sum(sum(~(C==C_temp)))~=0) 
%         C = C_temp; %update
%     else
%         break; %converged
%     end 
    
    
epoch=epoch+1;
end

% plot cost J_history
if (nargin>4)
    figure(plotJ),plot(J_history,'LineWidth', 2)
    xlabel('Iterations');
    ylabel('Cost');
    title(strcat('KMeans Cost K=',num2str(H)));
end


end
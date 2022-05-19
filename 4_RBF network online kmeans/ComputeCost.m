function J = ComputeCost(X, y, theta)

m = length(y); 
J = 0;

J = sqrt( sum((y - X*theta).^2) / m ); %RMSE
%J = sum((y - X*theta).^2) / (2*m);

end

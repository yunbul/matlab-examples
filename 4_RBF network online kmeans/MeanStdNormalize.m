function [X, mu, sigma, Val] = MeanStdNormalize(X, Val, m, s)

%multifunctional normalization function

if nargin < 2
    % return normalized X , mu and sigma
    mu = zeros(1, size(X, 2));
    sigma = zeros(1, size(X, 2));

    for i = 1:1:size(X,2)
        mu(i) = mean(X(:,i));
        sigma(i) = std(X(:,i));
        X(:,i) = (X(:,i) - mu(i)) / sigma(i); 
    end
    Val = 0;
else
    %normalize given data Val with m and s
    for i = 1:1:size(Val,2)
        Val(:,i) = (Val(:,i) - m(i)) / s(i); 
    end
    X = 0;
    mu = 0;
    sigma = 0;
end


end

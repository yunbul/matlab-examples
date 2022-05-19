function [theta, J_history_tra, J_history_val] = GradientDescent(X, Y, theta, alpha, num_iters, Val_X, Val_Y)

% X : G_tra, Val_X: G_val
m = length(Y); % number of training examples
J_history_tra = zeros(num_iters, 1);
J_history_val = zeros(num_iters, 1);

for iter = 1:num_iters

    theta = theta - (alpha/m) * (X' * (X * theta - Y));
       
    J_history_tra(iter) = ComputeCost(X, Y, theta);
    J_history_val(iter) = ComputeCost(Val_X, Val_Y, theta);
end

end

function [J, grad] = lrCostFunction(theta, X, y, lambda)
m = length(y);

J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);
non_reg_j = - (1 / m) * sum(y .* log(h) + (1 - y) .* log(1 - h), 1);
reg = lambda / (2*m) * sum(theta(2:end, 1) .^ 2);
J = non_reg_j + reg;


% [ 1 3 4..] this should be a column vector
non_reg_grad =  ((1 / m) * sum((h - y) .* X, 1))'; % invert to make column vector
reg_factor = (lambda / m) * theta;
reg_factor(1) = 0; % you're not supposed to regularized theta(0)

% grad is expected to be a column vector it's just like theta is.
% otherwise the advance optimization function will not work
grad = non_reg_grad + reg_factor;

end

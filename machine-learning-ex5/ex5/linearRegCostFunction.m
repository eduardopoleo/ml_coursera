function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

h = X * theta;
e = h - y;

non_reg_j = 1 / (2 * m) * sum(e .^ 2, 1);
reg_factor = lambda / (2 * m) * sum(theta(2:end) .^ 2);

J = non_reg_j + reg_factor;


non_reg_grad = 1 / m * sum(e .* X, 1);

reg_factor = (lambda / m) * theta;
reg_factor(1) = 0;
reg_factor = reg_factor';

grad =  non_reg_grad + reg_factor;







% =========================================================================

grad = grad(:);

end

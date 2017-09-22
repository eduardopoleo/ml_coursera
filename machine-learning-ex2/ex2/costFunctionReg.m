function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
inner_sum = y .* log(h) + (1 - y) .* log(1 - h);
total_sum = sum(inner_sum, 1);
j1 = - (1 / m) * total_sum;
reg = lambda / (2*m) * sum(theta(2:end, 1) .^ 2);
J = j1 + reg;


sumation =  (1 / m) * sum((h - y) .* X, 1)
theta = theta';
reg_factor = (lambda / m) * theta(1,2:end);
reg_factor = [0 reg_factor]

grad = sumation + reg_factor;

end

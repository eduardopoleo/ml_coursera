function J = computeCost(X, y, theta)
m = length(y); % number of training examples

% X * thetha does the multiplication/sum and leaves the result in a single
% column which is then easy to substract by y.
S = sum((X * theta - y) .^ 2)
J = 1 / (2 * m) * S


end

function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression
%   NORMALEQN(X,y) computes the closed-form solution to linear
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);
   
  part1 = (X'* X)^(-1)
  part2 = X'*y
  theta = part1 * part2
end

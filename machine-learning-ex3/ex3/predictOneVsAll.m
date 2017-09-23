function p = predictOneVsAll(all_theta, X)
m = size(X, 1);
num_labels = size(all_theta, 1);
% rows in all_theta labels
% columns in all_theta are features for corresponding theta values

% all_thetha looks like this
%          p 1 | p 2 | p 3 | p 4 | p 5 | ...
% label 1 | t1 | t2  | t3  | t4  | t5  | ...
% label 2 | t1 | t2  | t3  | t4  | t5  | ...
% label 3 | t1 | t2  | t3  | t4  | t5  | ...

% X looks like this
%       | p 1   |  p 2  |  p 3 |  p 4 |
% exp 1 |  0.2  |  0.2  |  0.2 |  0.3 |
% exp 2 |  0.5  |  0.2  |  0.7 |  0.2 |
% exp 3 |  0.3  |  0.2  |  0.4 |  0.8 |

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

theta_temp = all_theta';

for i = 1:m;
  experiment = X(i, :);
  z = experiment * theta_temp;
  h = sigmoid(z);
  [p_exp, ix] = max(h);
  p(i) = ix;
end

end

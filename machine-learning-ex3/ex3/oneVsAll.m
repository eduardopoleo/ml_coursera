function [all_theta] = oneVsAll(X, y, num_labels, lambda)
m = size(X, 1); % number of experiments (rows)
n = size(X, 2); % number of features (colums) pixels


all_theta = zeros(num_labels, n + 1); % row per label, 1 + columns per feature


% all_thetha looks like this
%          p 1 | p 2 | p 3 | p 4 | p 5 | ...
% label 1 | t1 | t2  | t3  | t4  | t5  | ...
% label 2 | t1 | t2  | t3  | t4  | t5  | ...
% label 3 | t1 | t2  | t3  | t4  | t5  | ...

% this format is specifically what the exercise wants but the fmincg
% requires having theta's organized by columns

% each label has a certain value of theta per pixel (feature)

X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);
%      label 1 |
% p 1 |   0    |
% p 2 |   0    |
% p 3 |   0    |

options = optimset('GradObj', 'on', 'MaxIter', 50);
theta_init = [];

for c = 1:num_labels;
  % initial theta always requires to be the blank slate on every iteration
  % we are solving the same problem again and again just using different y (y == c)
  [theta] = ...
          fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                  initial_theta, options);

  theta_init = [theta_init theta];
  % This builds a matrix like this
  %      label 1 | label 2 | label 3
  % p 1 |   t1   |  t2     |  t3
  % p 2 |   t1   |  t2     |  t3
  % p 3 |   t1   |  t2     |  t3
  % ...
  % ... where p is a bunch of features 
end

all_theta = theta_init';











% =========================================================================


end

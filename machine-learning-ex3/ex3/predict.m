function p = predict(Theta1, Theta2, X)
% size(Theta1) => 25 x 401
% size(Theta2) => 10 x 26
% size(X) =>   5000 x 400

m = size(X, 1);
X = [ones(m, 1) X];
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

theta1 = Theta1';
theta2 = Theta2';

for i = 1:m
  x = X(i,:); % => [0.4, 0.2, 0.1..] 401
  z = x * theta1;
  h1 = sigmoid(z); % => [0.2, 0.5, 0.6...] 25 row vector

  h1 = [1 h1];

  z = h1 * theta2;
  h2 = sigmoid(z); % => [0.2, 0.5, 0.6...] 10 row vector

  [m, xi] = max(h2);
  p(i) = xi;
end











% =========================================================================


end

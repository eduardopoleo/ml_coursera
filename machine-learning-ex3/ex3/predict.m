function p = predict(Theta1, Theta2, X)
% size(Theta1) => 25 x 401
% size(Theta2) => 10 x 26
% size(X) =>   5000 x 400

m = size(X, 1);
X = [ones(m, 1) X];
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

theta1 = Theta1'; % 401 x 25 which makes a lot more sense
                  % but apparently is more commonly given as Tetha1 is.

% a(2)1  |  a(2)2  | a(2)3
% t10    |  t20    |  t30
% t11    |  t21    |  t31
% t12    |  t22    |  t32
% t13    |  t23    |  t33

% In order to calculate a(2) you need to multiply the single input row X(i)
% by all the colums X(i) * theta1.
% Only one row at a time is needed and the theta is what determines what a gets
% calculated

theta2 = Theta2';

for i = 1:m
  x = X(i,:); % => [0.4, 0.2, 0.1..] 1 x 401
  z = x * theta1; # => 1 x 401 * 401 x 25 = 1 x 25
  h1 = sigmoid(z); % => [0.2, 0.5, 0.6...] 25 row vector

  h1 = [1 h1];

  z = h1 * theta2;
  h2 = sigmoid(z); % => [0.2, 0.5, 0.6...] 10 row vector

%                  ***> h1(x)
%                 ****> h2(x)
%          X(i) *****>  h3(x)    Last layer contains as many nodes as possible
%                 ****> h4(x)    labels. Each input row (Xi) has to go through
%                  ***> h5(x)    all possible hypothesis.
                  % ...
  [m, xi] = max(h2);
  p(i) = xi;
end

end

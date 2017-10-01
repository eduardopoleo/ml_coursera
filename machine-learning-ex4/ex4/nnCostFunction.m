function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% weights are already initialized as nn_params (randomly or not)

%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.


%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.

% size(X) => 5000 x 400
% size(Theta1) => 25 x 401
% size(Theta2) => 10 x 26
% size(y) => 5000 x 1


X = [ones(m,1) X]; % 5000 x 401
y_matrix = eye(num_labels)(y,:); % 5000 x 10

theta1 = Theta1'; % 401 X 25

%       feature1 feature2 feature3 feature4
% exp1


%          node1 | node2 | node3 | node4
% feature1
% feature2
% feature3
% feature4



theta2 = Theta2'; % 26 X 10
acc = [];
for i = 1:m
  z1 = X(i, :) * theta1; % 1 X 25
  g1 = sigmoid(z1);

  g1 = [1 g1]; % 1 X 26

  z2 = g1 * theta2; % 1 X 10
  h = sigmoid(z2); % 1 X 10

  yi = y_matrix(i, :); % 1 x 10

  err = log(h) * yi' + log(1 - h) * (1 - yi');

  acc = [acc err];
end

non_reg = - (1/m) * sum(acc);

theta1 = Theta1(:,2:end);
theta2 = Theta2(:,2:end);

reg_factor1 = sum((theta1 .^ 2)(:));
reg_factor2 = sum((theta2 .^ 2)(:));

reg_factor = lambda / (2 * m) * (reg_factor1 + reg_factor2);

J = non_reg + reg_factor;

%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

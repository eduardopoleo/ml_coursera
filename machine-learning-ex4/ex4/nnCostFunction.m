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


y_matrix = eye(num_labels)(y,:); % 5000 x 10

% COST CALCULATION by forward propagation

a1 = [ones(m,1) X]; % 5000 x 401
z2 = a1 * Theta1'; % 5000 x 25
a2 = sigmoid(z2); % 5000 x 25
a2 = [ones(m, 1) a2]; % 5000 x 26

z3 = a2 * Theta2'; % 5000 x 10
a3 = sigmoid(z3); % 5000 x 10
J = (sum(sum(-y_matrix .* log(a3) - (1 - y_matrix) .* log(1 - a3), 2))) / m;

% Regularization terms
reg_term1 = sum((Theta1(:, 2:end).^ 2)(:));
reg_term2 = sum((Theta2(:, 2:end).^ 2)(:));
reg_term = lambda / (2 * m) * (reg_term1 + reg_term2);
J = J + reg_term;

% backpropagation
d3 = a3 - y_matrix; % => 5000 x 10
d2 = (d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2); % 5000 x 25

D1 = d2' * a1;  % 25 x 401
D2 = d3' * a2;  % 10 x 26

Theta1_grad = Theta1_grad + (1/m) * D1;
Theta2_grad = Theta2_grad + (1/m) * D2;

% you never regularized the bias term
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);





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

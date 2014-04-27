function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feedforward propagation

% W1: hidden x visible = 25 x 64
% W2: visible x hidden = 64 x 25
% b1: hidden = 25 x 1
% b2: visible = 64 x 1
% data: n x m (visible x m, 64 x 10000)

[n, m] = size(data);

% a1 = data;
z2 = W1 * data + repmat(b1, 1, m); % = hidden x m = 25 x 10000
a2 = sigmoid(z2);    % same as z2
z3 = W2 * a2 + repmat(b2, 1, m);   % visible x m = 64 x 10000 = size of data
hx = sigmoid(z3);    % a3, same as z3

% KL-divergence for average activation in hidden layer
rho = mean(a2, 2); % 25 x 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COST function, backpropagation for gradient

% 1. sum square error
cost = 0.5 * sum(sum((hx - data) .^ 2)) / m;

delta3 = -(data - hx) .* sigmoidGradient(z3); % 64 x 10000
W2grad = delta3 * a2' / m;  % 64 x 25
b2grad = mean(delta3,2); % 64

% sparse penalty need to be added here
KLgrad = beta * (-sparsityParam ./ rho + (1-sparsityParam) ./ (1-rho)); % 25 x 1
KLgrad = repmat(KLgrad, 1, m); % match the size of all examples

delta2 = (W2' * delta3 + KLgrad) .* sigmoidGradient(z2); % 25 x 10000
W1grad = delta2 * data' / m; % 25 x 64
b1grad = mean(delta2,2);

% 2. regularization term
cost = cost + 0.5 * lambda * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)));
W2grad = W2grad + lambda * W2;
W1grad = W1grad + lambda * W1;

% 3. sparse penalty
cost = cost + beta * sum(kldiv(sparsityParam, rho));

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

% function sigm = sigmoid(x)
  
%     sigm = 1 ./ (1 + exp(-x));
% end


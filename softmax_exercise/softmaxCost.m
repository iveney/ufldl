function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% k: numClasses, (= 10)
% n: inputSize, (= 28x28 = 784)
% m: numCases, (= 60000)

% theta: k x n
% data: n x m
% groundTruth: k x m

% 1. basic term

% prevent overflow
M = theta * data;
M = bsxfun(@minus, M, max(M, [], 1));
numerator = exp(M); % k x m
denominator = sum(numerator);  % 1 x m
prob = bsxfun(@rdivide, numerator, denominator); % k x m, this is h(x)
cost = -sum(sum(groundTruth .* log(prob))) / numCases;

thetagrad = -(groundTruth - prob) * data' / numCases; % k x n = (k x m) x (m x n)

% 2. add normalization
cost = cost + 0.5 * lambda * sum(sum(theta .* theta));
thetagrad = thetagrad + lambda * theta;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end


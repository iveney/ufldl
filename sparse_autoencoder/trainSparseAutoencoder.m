function [encoderModel] = trainSparseAutoencoder(hiddenSize, visibleSize, ...
								lambda, sparsityParam, beta, data)
%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 200;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, data), ...
                              theta, options);

% Fold into a nicer format
encoderModel.opttheta = reshape(opttheta, hiddenSize, visibleSize);
encoderModel.hiddenSize = hiddenSize;
encoderModel.visibleSize = visibleSize;
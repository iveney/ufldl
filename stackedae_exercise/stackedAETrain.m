function [stackedAEModel] = stackedAETrain(stackedAETheta, ...
							  inputSize, hiddenSize, numClasses, netconfig, ...
							  lambda, data, labels, options)
% Fine tune a stacked autoencoder model with the given parameters and data

% Default options
if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'maxIter')
    options.maxIter = 400;
end

% Use minFunc to minimize the function
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % softmaxCost.m satisfies this.
minFuncOptions.display = 'on';

[optTheta, cost] = minFunc( @(p) stackedAECost(p, ...
                                     inputSize, hiddenSize, ...
                                     numClasses, netconfig, ...
                                     lambda, data, labels), ...                                   
                              	   stackedAETheta, options);

stackedAEModel.optTheta = optTheta;
stackedAEModel.inputSize = inputSize;
stackedAEModel.hiddenSize = hiddenSize;
stackedAEModel.numClasses = numClasses;
stackedAEModel.netconfig = netconfig;
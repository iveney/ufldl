%% CS294A/CS294W Sparse Coding Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  sparse coding exercise. In this exercise, you will need to modify
%  sparseCodingFeatureCost.m and sparseCodingWeightCost.m. You will also
%  need to modify this file, sparseCodingExercise.m slightly.

% Add the paths to your earlier exercises if necessary
% addpath /path/to/solution

%%======================================================================
%% STEP 0: Initialization
%  Here we initialize some parameters used for the exercise.

addpath ../sparse_autoencoder
addpath ../minFunc

numPatches = 20000;   % number of patches
numFeatures = 121;    % number of features to learn
patchDim = 8;         % patch dimension
visibleSize = patchDim * patchDim; 

% dimension of the grouping region (poolDim x poolDim) for topographic sparse coding
poolDim = 3;

% number of patches per batch
batchNumPatches = 2000; 

lambda = 5e-5;  % L1-regularisation parameter (on features)
epsilon = 1e-5; % L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
gamma = 1e-2;   % L2-regularisation parameter (on basis)

%%======================================================================
%% STEP 1: Sample patches

images = load('../sparse_autoencoder/IMAGES.mat');
images = images.IMAGES;

% do not normalize patch!
patches = sampleIMAGES(images, patchDim, numPatches, false);
display_network(patches(:, 1:64));

%%======================================================================
%% STEP 2: Implement and check sparse coding cost functions
%  Implement the two sparse coding cost functions and check your gradients.
%  The two cost functions are
%  1) sparseCodingFeatureCost (in sparseCodingFeatureCost.m) for the features 
%     (used when optimizing for s, which is called featureMatrix in this exercise) 
%  2) sparseCodingWeightCost (in sparseCodingWeightCost.m) for the weights
%     (used when optimizing for A, which is called weightMatrix in this exericse)

% We reduce the number of features and number of patches for debugging

DEBUG = false;
if DEBUG
numFeatures = 5;
numPatches = 5;
patches = patches(:, 1:numPatches);

weightMatrix = randn(visibleSize, numFeatures) * 0.005;
featureMatrix = randn(numFeatures, numPatches) * 0.005;

%% STEP 2a: Implement and test weight cost
%  Implement sparseCodingWeightCost in sparseCodingWeightCost.m and check
%  the gradient

[cost, grad] = sparseCodingWeightCost(weightMatrix, featureMatrix, ...
    visibleSize, numFeatures, patches, gamma, lambda, epsilon);

numgrad = computeNumericalGradient( @(x) sparseCodingWeightCost(...
    x, featureMatrix, visibleSize, numFeatures, patches, gamma, lambda, ...
    epsilon), weightMatrix(:) );
% Uncomment the blow line to display the numerical and analytic gradients side by side
% disp([numgrad grad]);     
diff = norm(numgrad-grad)/norm(numgrad+grad);
fprintf('Weight difference: %g\n', diff);
assert(diff < 1e-8, 'Weight difference too large. Check your weight cost function. ');

%% STEP 2b: Implement and test feature cost (non-topographic)
%  Implement sparseCodingFeatureCost in sparseCodingFeatureCost.m and check
%  the gradient. You may wish to implement the non-topographic version of
%  the cost function first, and extend it to the topographic version later.

% Set epsilon to a larger value so checking the gradient numerically makes sense
epsilon = 1e-2;

% We pass in the identity matrix as the grouping matrix, putting each
% feature in a group on its own.
groupMatrix = eye(numFeatures);

[cost, grad] = sparseCodingFeatureCost(weightMatrix, featureMatrix, ...
    visibleSize, numFeatures, patches, gamma, lambda, epsilon, groupMatrix);

numgrad = computeNumericalGradient( @(x) sparseCodingFeatureCost(...
    weightMatrix, x, visibleSize, numFeatures, patches, gamma, lambda, ...
    epsilon, groupMatrix), featureMatrix(:) );
% Uncomment the blow line to display the numerical and analytic gradients side by side
% disp([numgrad grad]); 
diff = norm(numgrad-grad)/norm(numgrad+grad);
fprintf('Feature difference (non-topographic): %g\n', diff);
assert(diff < 1e-8, 'Feature difference too large. Check your feature cost function. ');

%% STEP 2c: Implement and test feature cost (topographic)
%  Implement sparseCodingFeatureCost in sparseCodingFeatureCost.m and check
%  the gradient. This time, we will pass a random grouping matrix in to
%  check if your costs and gradients are correct for the topographic
%  version.

% Set epsilon to a larger value so checking the gradient numerically makes sense
epsilon = 1e-2;

% This time we pass in a random grouping matrix to check if the grouping is
% correct.
groupMatrix = rand(100, numFeatures);

[cost, grad] = sparseCodingFeatureCost(weightMatrix, featureMatrix, ...
    visibleSize, numFeatures, patches, gamma, lambda, epsilon, groupMatrix);

numgrad = computeNumericalGradient( @(x) sparseCodingFeatureCost(...
    weightMatrix, x, visibleSize, numFeatures, patches, gamma, lambda, ...
    epsilon, groupMatrix), featureMatrix(:) );
% Uncomment the blow line to display the numerical and analytic gradients side by side
% disp([numgrad grad]); 
diff = norm(numgrad-grad)/norm(numgrad+grad);
fprintf('Feature difference (topographic): %g\n', diff);
assert(diff < 1e-8, 'Feature difference too large. Check your feature cost function. ');

end % DEBUG
%%======================================================================
%% STEP 3: Iterative optimization
%  Once you have implemented the cost functions, you can now optimize for
%  the objective iteratively. The code to do the iterative optimization 
%  using mini-batching and good initialization of the features has already
%  been included for you. 
% 
%  However, you will still need to derive and fill in the analytic solution 
%  for optimizing the weight matrix given the features. 
%  Derive the solution and implement it in the code below, verify the
%  gradient as described in the instructions below, and then run the
%  iterative optimization.

% Initialize options for minFunc
options.Method = 'cg';
options.display = 'off';
options.verbose = 0;

% Initialize matrices

% For debug
if DEBUG
    numFeatures = 9;
    batchNumPatches = 2;
end

weightMatrix = rand(visibleSize, numFeatures);
featureMatrix = rand(numFeatures, batchNumPatches);

% Initialize grouping matrix
assert(floor(sqrt(numFeatures)) ^2 == numFeatures, ...
    'numFeatures should be a perfect square');
donutDim = floor(sqrt(numFeatures));
assert(donutDim * donutDim == numFeatures, ...
    'donutDim^2 must be equal to numFeatures');

groupMatrix = zeros(numFeatures, donutDim, donutDim);

groupNum = 1;
for row = 1:donutDim
    for col = 1:donutDim
        groupMatrix(groupNum, 1:poolDim, 1:poolDim) = 1;
        groupNum = groupNum + 1;
        groupMatrix = circshift(groupMatrix, [0 0 -1]);
    end
    groupMatrix = circshift(groupMatrix, [0 -1, 0]);
end

groupMatrix = reshape(groupMatrix, numFeatures, numFeatures);

% ask if want to use topographic
answer = questdlg('Initialize grouping matrix for topographic or non-topographic sparse coding?', ...
        'Topographic/non-topographic?', 'Non-topographic', 'Topographic', 'Non-topographic');
if isequal(answer, 'Non-topographic')
    groupMatrix = eye(numFeatures);
end

% Initial batch
indices = randperm(numPatches);
indices = indices(1:batchNumPatches);
batchPatches = patches(:, indices);                           

fprintf('%6s%12s%12s%12s%12s\n','Iter', 'fObj','fResidue','fSparsity','fWeight');

% save as movie
nIter = 200;
outputVideo = VideoWriter('sparse_coding.avi', 'Grayscale AVI');
outputVideo.FrameRate = 10;
open(outputVideo);

for iteration = 1:nIter
    error = weightMatrix * featureMatrix - batchPatches;
    error = sum(error(:) .^ 2) / batchNumPatches;
    
    fResidue = error;
    
    R = groupMatrix * (featureMatrix .^ 2);
    R = sqrt(R + epsilon);    
    fSparsity = lambda * sum(R(:));    
    
    fWeight = gamma * sum(weightMatrix(:) .^ 2);
    
    fprintf('  %4d  %10.4f  %10.4f  %10.4f  %10.4f\n', iteration, ...
           fResidue+fSparsity+fWeight, fResidue, fSparsity, fWeight)
               
    % Select a new batch
    indices = randperm(numPatches);
    indices = indices(1:batchNumPatches);
    batchPatches = patches(:, indices);                    
    
    % Reinitialize featureMatrix with respect to the new batch
    featureMatrix = weightMatrix' * batchPatches;
    normWM = sum(weightMatrix .^ 2)';
    featureMatrix = bsxfun(@rdivide, featureMatrix, normWM); 
    
    % Optimize for feature matrix    
    options.maxIter = 20;
    [featureMatrix, cost] = minFunc( @(x) sparseCodingFeatureCost(...
        weightMatrix, x, visibleSize, numFeatures, batchPatches, gamma, ...
        lambda, epsilon, groupMatrix), featureMatrix(:), options);
    featureMatrix = reshape(featureMatrix, numFeatures, batchNumPatches);                                      
       
    % Optimize for weight matrix  

    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Fill in the analytic solution for weightMatrix that minimizes 
    %   the weight cost here.     
    %   Once that is done, use the code provided below to check that your
    %   closed form solution is correct.
    %   Once you have verified that your closed form solution is correct,
    %   you should comment out the checking code before running the
    %   optimization.

    weightMatrix = (batchPatches * featureMatrix') / ...
         (featureMatrix * featureMatrix' + ...
          gamma * batchNumPatches * eye(numFeatures));
    
    % [cost, grad] = sparseCodingWeightCost(weightMatrix, featureMatrix, ...
        % visibleSize, numFeatures, batchPatches, gamma, lambda, ...
        % epsilon, groupMatrix);
    % disp(norm(grad))
    % assert(norm(grad) < 1e-12, ...
        % 'Weight gradient should be close to 0. Check your closed form solution for weightMatrix again.')
    % error('Weight gradient is okay. Comment out checking code before running optimization.');
    % -------------------- YOUR CODE HERE --------------------   
    
    % Visualize learned basis
    figure(1);
    [~, I] = display_network(weightMatrix);

    % rescale to [0, 1] and upsample for better display
    Irange = max(I(:)) - min(I(:));
    I =  (I - min(I(:))) ./ Irange;
    writeVideo(outputVideo, imresize(I, 3, 'nearest'));
    % saveas(gcf, ['weightMatrix-' answer '.png']);           
end

close(outputVideo);

% display weight and feature matrix then save them
display_network(weightMatrix);
saveas(gcf, ['weightMatrix-' answer '.png']);
display_network(featureMatrix(:, 1:numFeatures));
saveas(gcf, ['featureMatrix-' answer '.png']);
save([answer '.mat'], 'featureMatrix', 'weightMatrix');
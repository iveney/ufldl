function patches = sampleIMAGES(IMAGES, patchsize, numpatches)
% sampleIMAGES
% Returns 10000 patches for training

if ~exist('IMAGES', 'var')
	load('IMAGES.mat');    % load images from disk 
end

if ~exist('patchsize', 'var')
	patchsize = 8;  % we'll use 8x8 patches 
end

if ~exist('numpatches', 'var')
	numpatches = 10000;
end

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1

[m, n, N] = size(IMAGES);
for i=1:numpatches
	j = randi([1 N]);
	r = randi([1 m-patchsize+1]);
	c = randi([1 n-patchsize+1]);
	patch = IMAGES(r:r+patchsize-1,c:c+patchsize-1,j);
	patches(:, i) = patch(:);
end

% DEBUG: display the first 10 patches
% for i=1:25
% 	subplot(5,5,i);
% 	imagesc(reshape(patches(:,i), 8, 8)), colormap gray, axis equal, axis off;
% end

%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end

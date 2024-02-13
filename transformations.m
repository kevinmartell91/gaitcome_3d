% Read an image
I = imread('./visionData/removeBackground/780.png');

% Define a projective 2D transformation matrix
theta = 10.001;
tm = [4 0 0.0001; ...
      0 4 0.00001; ...
      0 0 4];
tform = projective2d(tm);

% Apply the transformation to the image
outputImage = imwarp(I, tform);

% Display the transformed image
figure
imshow(outputImage);
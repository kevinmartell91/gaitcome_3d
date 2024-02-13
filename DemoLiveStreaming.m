


vid1 = VideoReader('FHD0305.MOV');
vid2 = VideoReader('FHD0318.MOV');
% get(vid1);
% vidHeight = vidObj.Height;
% vidWidth = vidObj.Width;
% 
% s = struct('cdata',zeros(vidHeight,vidWidth,3,'uint8'),...
%     'colormap',[]);
whos vid1
whos vid2
% currAxes = axes;

% while hasFrame(vid1)
while (false)
    I1 = readFrame(vid1);
    I2 = readFrame(vid2);
    %% Load Camera Parameters 
    % This example uses the camera parameters calculated by the 
    % <matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'visionCameraCalibrator'); cameraCalibrator> 
    % app. The parameters are stored in the |cameraParams| object, and include 
    % the camera intrinsics and lens distortion coefficients.

    % Load precomputed camera parameters
    load upToScaleReconstructionCameraParameters.mat
%     load calibrationSession_day.mat

    %% Remove Lens Distortion
    % Lens distortion can affect the accuracy of the final reconstruction. You
    % can remove the distortion from each of the images using the |undistortImage|
    % function. This process straightens the lines that are bent by the radial 
    % distortion of the lens.
    I1 = undistortImage(I1, cameraParams);
    I2 = undistortImage(I2, cameraParams);

%     figure 
%     imshowpair(I1, I2, 'montage');
%     title('Undistorted Images');    
    
    %% Find Point Correspondences Between The Images - Dummy version
    matchedPoints1 = calculateCentroids(I1,19);
    matchedPoints2 = calculateCentroids(I2,20);

    %% Estimate the Essential Matrix
    % Use the |estimateEssenitalMatrix| function to compute the essential 
    % matrix and find the inlier points that meet the epipolar constraint.

    % Estimate the fundamental matrix
    [E, epipolarInliers] = estimateEssentialMatrix(...
        matchedPoints1, matchedPoints2, cameraParams, 'Confidence', 99.99);

    % Find epipolar inliers
    inlierPoints1 = matchedPoints1(epipolarInliers, :);
    inlierPoints2 = matchedPoints2(epipolarInliers, :);

    % Display inlier matches
%     figure
%     showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2);
%     title('Epipolar Inliers');

    %% Compute the Camera Pose
    % Compute the location and orientation of the second camera relative to the
    % first one. Note that |t| is a unit vector, because translation can only 
    % be computed up to scale.

    [orient, loc] = relativeCameraPose(E, cameraParams, inlierPoints1, inlierPoints2);

    %% Reconstruct the 3-D Locations of Matched Points
    % Re-detect points in the first image using lower |'MinQuality'| to get
    % more points. Track the new points into the second image. Estimate the 
    % 3-D locations corresponding to the matched points using the |triangulate|
    % function, which implements the Direct Linear Transformation
    % (DLT) algorithm [1]. Place the origin at the optical center of the camera
    % corresponding to the first image.

    % Detect dense feature points. Use an ROI to exclude points close to the
    % image edges.
            % roi = [30, 30, size(I1, 2) - 30, size(I1, 1) - 30];
            % imagePoints1 = detectMinEigenFeatures(rgb2gray(I1), 'ROI', roi, ...
            %     'MinQuality', 0.001);
            % 
            % % Create the point tracker
            % tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);
            % 
            % % Initialize the point tracker
            % imagePoints1 = imagePoints1.Location;
            % initialize(tracker, imagePoints1, I1);
            % 
            % % Track the points
            % [imagePoints2, validIdx] = step(tracker, I2);
            % matchedPoints1 = imagePoints1(validIdx, :);
            % matchedPoints2 = imagePoints2(validIdx, :);

    % Compute the camera matrices for each position of the camera
    % The first camera is at the origin looking along the Z-axis. Thus, its
    % rotation matrix is identity, and its translation vector is 0.
    camMatrix1 = cameraMatrix(cameraParams, eye(3), [0 0 0]);

    % Compute extrinsics of the second camera
    [R, t] = cameraPoseToExtrinsics(orient, loc);
    camMatrix2 = cameraMatrix(cameraParams, R, t);

    % Compute the 3-D points
    points3D = triangulate(matchedPoints1, matchedPoints2, camMatrix1, camMatrix2);

    % Get the color of each reconstructed point
   % Initialize video readers for two video files
vid1 = VideoReader('FHD0305.MOV');
vid2 = VideoReader('FHD0318.MOV');

% Display information about video files
whos vid1
whos vid2

% Main processing loop (currently disabled with while(false))
while (false)
    % Read frames from both videos
    I1 = readFrame(vid1);
    I2 = readFrame(vid2);
    
    % Load precomputed camera parameters
    load upToScaleReconstructionCameraParameters.mat
    
    % Remove lens distortion from both images
    I1 = undistortImage(I1, cameraParams);
    I2 = undistortImage(I2, cameraParams);
    
    % Find point correspondences between the images
    matchedPoints1 = calculateCentroids(I1,19);
    matchedPoints2 = calculateCentroids(I2,20);
    
    % Estimate the essential matrix
    [E, epipolarInliers] = estimateEssentialMatrix(matchedPoints1, matchedPoints2, cameraParams, 'Confidence', 99.99);
    
    % Find epipolar inliers
    inlierPoints1 = matchedPoints1(epipolarInliers, :);
    inlierPoints2 = matchedPoints2(epipolarInliers, :);
    
    % Compute the camera pose
    [orient, loc] = relativeCameraPose(E, cameraParams, inlierPoints1, inlierPoints2);
    
    % Compute camera matrices for each position of the camera
    camMatrix1 = cameraMatrix(cameraParams, eye(3), [0 0 0]);
    [R, t] = cameraPoseToExtrinsics(orient, loc);
    camMatrix2 = cameraMatrix(cameraParams, R, t);
    
    % Compute the 3-D points
    points3D = triangulate(matchedPoints1, matchedPoints2, camMatrix1, camMatrix2);
    
    % Get the color of each reconstructed point
    numPixels = size(I1, 1) * size(I1, 2);
    allColors = reshape(I1, [numPixels, 3]);
    colorIdx = sub2ind([size(I1, 1), size(I1, 2)], round(matchedPoints1(:,2)), round(matchedPoints1(:, 1)));
    color = allColors(colorIdx, :);
    
    % Create the point cloud
    ptCloud = pointCloud(points3D);
    
    % Display the 3-D point cloud
    cameraSize = 0.3;
    figure
    plotCamera('Size', cameraSize, 'Color', 'r', 'Label', '1', 'Opacity', 0);
    hold on
    grid on
    plotCamera('Location', loc, 'Orientation', orient, 'Size', cameraSize, 'Color', 'b', 'Label', '2', 'Opacity', 0);
    pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', 'MarkerSize', 450);
    camorbit(0, -30);
    camzoom(1.5);
    xlabel('x-axis');
    ylabel('y-axis');
    zlabel('z-axis');
    title('Up to Scale Reconstruction of the Scene');
    pause(20);       
end

function matchedPoints = calculateCentroids(I1, THRESH)
    % Convert the input image to grayscale
    GI1 = rgb2gray(I1);
    
    % Apply the specified threshold to create a binary image
    thresh = THRESH;
    BW1_TH = GI1 > thresh;
    
    % Detect blobs in the binary image
    BW1 = bwconncomp(BW1_TH); 
    stats = regionprops(BW1, 'Area', 'Eccentricity'); 
    idx = find([stats.Area] > 1); 
    BW1_BLOB = ismember(labelmatrix(BW1), idx);

    % Calculate centroids of the detected blobs
    CC1 = regionprops(BW1_BLOB, 'centroid');

    % Extract the centroids as matched points
    matchedPoints = single(cat(1, CC1.Centroid));

    % Display the image with the detected centroids
    imshow(I1);
    hold on
    plot(matchedPoints(:, 1), matchedPoints(:, 2), 'b*');
    hold off

    % Sort the matched points in ascending order based on the sum of X and Y values
    [rows, ~] = size(matchedPoints);
    tempMat = [];
    for i = 1:rows
        newRow = [(matchedPoints(i, 1) + matchedPoints(i, 2)) matchedPoints(i, 1) matchedPoints(i, 2)];
        tempMat = cat(1, tempMat, newRow); 
    end

    out = sortrows(tempMat);
    out(:, 1) = [];
    matchedPoints = out;
end
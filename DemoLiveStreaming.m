
% ==============================================================================

% % Replace url with your device specific stream
% url = 'rtsp://192.168.1.1/MJPG?W=640&H=360&Q=50&BR=3000000';
% cam = HebiCam(url);

% clear cam; % make sure device is not in use
% cam = HebiCam(1);
% imshow(cam.getsnapshot());
% 
% [image, frameNumber, timestamp] = getsnapshot(cam);
% 
% 
% figure();
% fig = imshow(getsnapshot(cam));
% cont = 0;
% while cont < 100
%     set(fig, 'CData', getsnapshot(cam)); 
%     drawnow;
%     cont = 1 + cont;
% end



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
    numPixels = size(I1, 1) * size(I1, 2);
    allColors = reshape(I1, [numPixels, 3]);
    colorIdx = sub2ind([size(I1, 1), size(I1, 2)], round(matchedPoints1(:,2)), ...
        round(matchedPoints1(:, 1)));
    color = allColors(colorIdx, :);

    % Create the point cloud
            % ptCloud = pointCloud(points3D, 'Color', color);
    ptCloud = pointCloud(points3D);

    %% Display the 3-D Point Cloud
    % Use the |plotCamera| function to visualize the locations and orientations
    % of the camera, and the |pcshow| function to visualize the point cloud.

    % Visualize the camera locations and orientations
    cameraSize = 0.3;
    figure
    plotCamera('Size', cameraSize, 'Color', 'r', 'Label', '1', 'Opacity', 0);
    hold on
    grid on
    plotCamera('Location', loc, 'Orientation', orient, 'Size', cameraSize, ...
        'Color', 'b', 'Label', '2', 'Opacity', 0);

    % Visualize the point cloud
    pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
        'MarkerSize', 450);

    % Rotate and zoom the plot
    camorbit(0, -30);
    camzoom(1.5);

    % Label the axes
    xlabel('x-axis');
    ylabel('y-axis');
    zlabel('z-axis')

    title('Up to Scale Reconstruction of the Scene');

    pause(20);       
%     disp(vid2.FrameRate);
end


function matchedPoints = calculateCentroids(I1,THRESH)
%% Segmentation with a threshold by KEVIN
% Convert to binary images with a threshold
GI1 = rgb2gray(I1);
% GI2 = rgb2gray(I2);
thresh = THRESH;
BW1_TH = GI1 > thresh ;
% BW2_TH = GI2 > thresh ;

% detect blobs from BW1_TH
BW1 = bwconncomp(BW1_TH); 
stats = regionprops(BW1, 'Area','Eccentricity'); 
idx = find([stats.Area] > 1); 
BW1_BLOB = ismember(labelmatrix(BW1), idx);
% detect blobs from BW2_TH
% BW2 = bwconncomp(BW2_TH); 
% stats = regionprops(BW2, 'Area','Eccentricity'); 
% idx = find([stats.Area] > 100); 
% BW2_BLOB = ismember(labelmatrix(BW2), idx);

% ploting binary images
% imshowpair(BW1_BLOB,BW2_BLOB,'montage');

% Calculate centroids
CC1 = regionprops(BW1_BLOB,'centroid');
% CC2 = regionprops(BW2_BLOB,'centroid');

% Calculate both matched point
matchedPoints = single(cat(1,CC1.Centroid));
% matchedPoints1(:,3) = [];
% matchedPoints2 = single(cat(1,CC2.Centroid));
% matchedPoints2(:,3) = [];
% 
imshow(I1);
hold on
plot(matchedPoints(:,1),matchedPoints(:,2),'b*');
hold off
% 
% imshow(I2);
% hold on
% plot(matchedPoints2(:,1),matchedPoints2(:,2),'b+');
% hold off


% asendent sort by adding its X and Y values
[rows, columns] = size(matchedPoints);
tempMat = [];
for i = 1:rows
    newRow = [(matchedPoints(i,1) + matchedPoints(i,2)) matchedPoints(i,1) matchedPoints(i,2)];
    tempMat = cat(1, tempMat, newRow); 
end

out = sortrows(tempMat);
out(:,1) =[];
matchedPoints = out;


end


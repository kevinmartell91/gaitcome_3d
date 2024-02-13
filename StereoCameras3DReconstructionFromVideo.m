% Clear previous values in memory
clc;
close all;
clear all;

%% Static values
% Define input type: 1 for Eken (video), 2 for ps3 (images)
TYPE_DATA_INPUT = 2;        
% Define image processing type: 1 for processing each frame, 2 for background subtraction
TYPE_IMG_PROC = 1;          
% Number of markers to track
N_MARKERS = 3;              
NUM_MARKERS = N_MARKERS;    
% Oscillation graph flag (true to show oscillation graphs)
OSCILATION = true;          

% Threshold values for binary image conversion
BINARY_THRES_A = 0.8;
BINARY_THRES_B = 0.8868;

% Area thresholds for marker detection
MIN_AREA = 2;
MAX_AREA = 60;

% Threshold values for image processing
THRESH_L = 230;
THRESH_R = 230 + 10;
% Multiplier to filter blob sizes detected
SURGE_MEAN_AREA = 1;      

% Image frame limits
MARGIN = 0;
TOP_LIM = MARGIN;
BOT_LIM = 720 - MARGIN;
LEF_LIM = MARGIN;
RIG_LIM = 1280 - MARGIN;

% Sync and blink detection flags
isSync = true;
outOfPhase_l = 0;             % Number of out of phase left frames
outOfPhase_r = 0;             % Number of out of phase right frames
isBlinking_l = false;
isBlinking_r = false;
SYNC_UMBRAL = 230;

% 3D Reconstruction flag
is3DEnable = true;

% Leg side flag (true for left, false for right)
isLeft = true;

% Frame skip settings
INI_SEC = 1/120;  % Start at second 0 for frame 1
INI_SEC_INITIAL_CONTACT_AFTER_FIRST_BLINK = 8.2;  % Initial contact X seconds after 1st blink
END_SEC = 9.5;  % End at second ?

% Initialize global variables for previous angles
global prev_S_ANG_HIP prev_S_ANG_PEL prev_S_ANG_KNE prev_S_ANG_ANK;
global prev_F_ANG_HIP prev_F_ANG_PEL prev_F_ANG_KNE prev_F_ANG_ANK;
global prev_T_ANG_HIP prev_T_ANG_PEL prev_T_ANG_KNE prev_T_ANG_ANK;

% Set initial values for previous angles to 0
prev_S_ANG_HIP = 0;
prev_S_ANG_PEL = 0;
prev_S_ANG_KNE = 0;
prev_S_ANG_ANK = 0;
prev_F_ANG_HIP = 0;
prev_F_ANG_PEL = 0;
prev_F_ANG_KNE = 0;
prev_F_ANG_ANK = 0;
prev_T_ANG_HIP = 0;
prev_T_ANG_PEL = 0;
prev_T_ANG_KNE = 0;
prev_T_ANG_ANK = 0;

%% Initiate vectors to save 3D raw coordinates, jsonResult, angles
idx_raw_data = 1;  % Index to manage vector position
% Initialize global variables for marker coordinates
global lbwt_x lbwt_y lbwt_z;
global lfwt_x lfwt_y lfwt_z;
global ltrc_x ltrc_y ltrc_z;
global lkne_x lkne_y lkne_z;
global lank_x lank_y lank_z;
global lhee_x lhee_y lhee_z;
global lteo_x lteo_y lteo_z;
lbwt_x = []; lbwt_y = []; lbwt_z = [];
lfwt_x = []; lfwt_y = []; lfwt_z = [];
ltrc_x = []; ltrc_y = []; ltrc_z = [];
lkne_x = []; lkne_y = []; lkne_z = [];
lank_x = []; lank_y = []; lank_z = [];
lhee_x = []; lhee_y = []; lhee_z = [];
lteo_x = []; lteo_y = []; lteo_z = [];

% Initialize arrays for angles
sagittal_hip_ang = [];
sagittal_plv_ang = [];
sagittal_kne_ang = [];
sagittal_ank_ang = [];
frontal_hip_ang = [];
frontal_plv_ang = [];
frontal_kne_ang = [];
frontal_ank_ang = [];
transversal_hip_ang = [];
transversal_plv_ang = [];
transversal_kne_ang = [];
transversal_ank_ang = [];
countLeft = [];
countRight = [];
countIndex = 1;

%% Instaciating Kinematic Analysis class
ka = CKinematicAnalysis; 

%% Create Video File Readers and the Video player_left

% Video file paths for left and right cameras
videoFileLeft  = '../ekenRawFiles/camera_a/test_17_video/FHD0063.MOV';  
videoFileRight = '../ekenRawFiles/camera_b/test_17_video/FHD0073.MOV';  

% Initialize video players for left and right videos
player_left = vision.DeployableVideoPlayer( ...
                'Location', [0, 0], ...
                'Size', 'Custom', ...
                'CustomSize', [750, 500], ...
                'Name', 'player_left' ...
              );

player_right = vision.DeployableVideoPlayer( ...
                'Location', [4000, 0], ...
                'Size', 'Custom', ...
                'CustomSize', [750, 500], ...
                'Name', 'player_right' ...
               );

% Oscillation - live data visualization
if OSCILATION
    % Initialize plots for oscillation in X, Y, and Z coordinates
    [HX, PX] = InitPlotOscilation(-0.05, 0.05,'X - Oscilation');
    [HY, PY] = InitPlotOscilation(-0.05, 0.05,'Y - Oscilation');
    [HZ, PZ] = InitPlotOscilation(0.03, 0.10,'Z - Oscilation');
    startTime = datetime('now');
end


%% Load cameras stereoParams 
load('./visionData/videoCalibration/stereoParams_test_pseye_01.mat');

%% Create a streaming point cloud viewer
% Adjusted for better visualization of oscillation in x, y, and z coordinates
player3D = pcplayer( ...
    [-0.05, 0.05], [-0.05, 0.05], [0, 0.3], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down', 'MarkerSize', 1000);

% Set labels for the axes of the point cloud viewer
xlabel(player3D.Axes,'x-axis (cm)');
ylabel(player3D.Axes,'y-axis (cm)');
zlabel(player3D.Axes,'z-axis (cm)');
  
%% Compute camera matrices for each position of the camera
camMatrixLeft = cameraMatrix( ...
        stereoParams_test_pseye_01.CameraParameters1, ...
        eye(3), [0 0 0] ...
    );
camMatrixRight = cameraMatrix( ...
        stereoParams_test_pseye_01.CameraParameters2, ...
        stereoParams_test_pseye_01.RotationOfCamera2, ...
        stereoParams_test_pseye_01.TranslationOfCamera2 ...
    );
                             
%% Initialize video file reading
movleft = VideoReader(videoFileLeft);
movRight = VideoReader(videoFileRight);

% Determine the number of frames to process based on input type
if TYPE_DATA_INPUT == 1
    iteFrames = isSync ? INI_SEC_INITIAL_CONTACT_AFTER_FIRST_BLINK * movleft.FrameRate : INI_SEC * movleft.FrameRate;
    endFrames = END_SEC * movRight.FrameRate;
elseif TYPE_DATA_INPUT == 2
    iteFrames = 0; % Manual setting for image-based input
    endFrames = 769; % Total number of images
end

%% Get background images from both cameras for background subtraction
if TYPE_IMG_PROC == 2
    backgroundLeft = read(movleft, 1); % Frame 1 from camera A
    backgroundRight = read(movRight, 1); % Frame 1 from camera B
end

SURGE = 1; % Initialize frame surge counter

%% iterating both video frames
while iteFrames < endFrames 

    % Sync process
    while ~isSync

         if TYPE_DATA_INPUT == 1
            % Retrieve video frames before syncing process
            frameLeft = read(movleft, iteFrames);
            frameRight = read(movRight, iteFrames); 
        elseif TYPE_DATA_INPUT == 2
            % Retrieve PS3 frames - no syncing process (test mode)
            ps3Index = iteFrames;
            frameLeft  = imread(['../ps3RawFiles/cam0/img', ...
                                 num2str(ps3Index +1), '.jpg']);
            frameRight = imread(['../ps3RawFiles/cam1/img', ...
                                 num2str(ps3Index), '.jpg']);  
        end

        % Display current frames
        step(player_left, frameLeft);
        step(player_right, frameRight);
        
        % Detect out-of-phase frames and blinking
        [isSync, outOfPhase_l, outOfPhase_r, isBlinking_l, isBlinking_r] = ...
                DetectOutOfPhaseFrames(frameLeft, frameRight, ...
                                       outOfPhase_l, outOfPhase_r, ...
                                       isBlinking_l, isBlinking_r, ...
                                       SYNC_UMBRAL, TOP_LIM, BOT_LIM, ...
                                       LEF_LIM, RIG_LIM);
        % Increment frame counter
        iteFrames = iteFrames +1;
        
        % If frames are synced, release video players
        if isSync
            release(player_left);
            release(player_right);
      
            % Adjust frame counter based on data input type
            if TYPE_DATA_INPUT == 1
                iteFrames = INI_SEC_INITIAL_CONTACT_AFTER_FIRST_BLINK * ...
                            movleft.FrameRate;
            elseif TYPE_DATA_INPUT == 2
                iteFrames = 0;
            end
        end
    end
    
    % Process frames after syncing
    if TYPE_DATA_INPUT == 1
        JUMP = 0;
        % Retrieve video frames after syncing
        frameLeft = read(movleft, iteFrames + outOfPhase_l + (SURGE * JUMP));
        frameRight = read(movRight, iteFrames + outOfPhase_r + (SURGE * JUMP)); 

        % Flip frames for right sagittal view emulation
        if ~isLeft
            frameLeft = flipdim(frameLeft, 2);
            frameRight = flipdim(frameRight, 2);
        end

        SURGE = SURGE +1;

    elseif TYPE_DATA_INPUT == 2
        % Retrieve PS3 frames - no syncing process (test mode)
        ps3Index = iteFrames;
        frameLeft  = imread(['../ps3RawFiles/cam0/img', ...
                             num2str(ps3Index)  - outOfPhase_l, '.jpg']);
        frameRight = imread(['../ps3RawFiles/cam1/img', ...
                             num2str(ps3Index) - outOfPhase_r, '.jpg']);  
    end

    %% Remove lens distortion
    frameLeft = undistortImage(frameLeft, ...
                               stereoParams_test_pseye_01.CameraParameters1);
    frameRight = undistortImage(frameRight, ...
                                stereoParams_test_pseye_01.CameraParameters2);

    % Marker recognition process
    if TYPE_IMG_PROC == 1
        % Process frames using kernel testing
        [frameLeftBinary, markersPosXY_Left] = ...
            TestKernels(frameLeft, THRESH_L, SURGE_MEAN_AREA, TOP_LIM, ...
                        BOT_LIM, LEF_LIM, RIG_LIM);
        [frameRightBinary, markersPosXY_Right] = ...
            TestKernels(frameRight, THRESH_R, SURGE_MEAN_AREA, TOP_LIM, ...
                        BOT_LIM, LEF_LIM, RIG_LIM);
    elseif TYPE_IMG_PROC == 2
        % Process frames using background subtraction
        [medFiltframeLeft, frameLeftBinary] = ...
            DetectMarkers_BackgroundSubtraction(backgroundLeft, frameLeft, ...
                                                BINARY_THRES_A);
        [medFiltframeRight, frameRightBinary] = ...
            DetectMarkers_BackgroundSubtraction(backgroundRight, frameRight, ...
                                                BINARY_THRES_B);

        % Extract marker positions
        markersPosXY_Left = ...
            ExtractMarkersPosXY(medFiltframeLeft, SURGE_MEAN_AREA);
        markersPosXY_Right = ...
            ExtractMarkersPosXY(medFiltframeRight, SURGE_MEAN_AREA);    
    end
    
    % 3D analysis if frames are synced and 3D is enabled
    if(isSync && is3DEnable) 
        numMarkersLeft = size(markersPosXY_Left,1);
        numMarkersRight = size(markersPosXY_Right,1);

        % Proceed with 3D reconstruction if the number of markers matches
        if numMarkersLeft == NUM_MARKERS && numMarkersRight == NUM_MARKERS
            % Label markers based on the number specified
            if NUM_MARKERS == 7   
                matchedPointsLeft = ...
                    LabelMarkers2DImages_7(DescendSort_YAxisValues(markersPosXY_Left),isLeft);
                matchedPointsRight = ...
                    LabelMarkers2DImages_7(DescendSort_YAxisValues(markersPosXY_Right), isLeft);
            elseif NUM_MARKERS == 9  
                matchedPointsLeft = ...
                    LabelMarkers2DImages_9(DescendSort_YAxisValues(markersPosXY_Left));
                matchedPointsRight = ...
                    LabelMarkers2DImages_9(DescendSort_YAxisValues(markersPosXY_Right));  
            elseif NUM_MARKERS == N_MARKERS 
                matchedPointsLeft = ...
                    AscendentSort_SumOfAxisValues(markersPosXY_Left);
                matchedPointsRight = ...
                    AscendentSort_SumOfAxisValues(markersPosXY_Right);
            end

            % Triangulate to compute 3D points
            points3D = triangulate(matchedPointsLeft, matchedPointsRight, ...
                                   camMatrixLeft, camMatrixRight);

            % Convert to meters and create a pointCloud object
            points3D = points3D ./1000; % Scale conversion

            % Calculate angles for 7 markers setup
            if NUM_MARKERS == 7
                % Convert Points3D into an array of dictionaries        
                [lbwt, lfwt, ltrc, lkne, lank, lhee, lteo] = GetArrayDicPoints3D(points3D);

                % Save Points3D for JSON encoding
                lbwt_x{idx_raw_data}= lbwt('x'); lbwt_y{idx_raw_data}= lbwt('y'); lbwt_z{idx_raw_data}= lbwt('z');
                lfwt_x{idx_raw_data}= lfwt('x'); lfwt_y{idx_raw_data}= lfwt('y'); lfwt_z{idx_raw_data}= lfwt('z');
                ltrc_x{idx_raw_data}= ltrc('x'); ltrc_y{idx_raw_data}= ltrc('y'); ltrc_z{idx_raw_data}= ltrc('z');
                lkne_x{idx_raw_data}= lkne('x'); lkne_y{idx_raw_data}= lkne('y'); lkne_z{idx_raw_data}= lkne('z');
                lank_x{idx_raw_data}= lank('x'); lank_y{idx_raw_data}= lank('y'); lank_z{idx_raw_data}= lank('z');
                lhee_x{idx_raw_data}= lhee('x'); lhee_y{idx_raw_data}= lhee('y'); lhee_z{idx_raw_data}= lhee('z');
                lteo_x{idx_raw_data}= lteo('x'); lteo_y{idx_raw_data}= lteo('y'); lteo_z{idx_raw_data}= lteo('z');

                % Decode JSON for calculated angles in three planes
                lstResThreePlanes = jsondecode(GetJsonOfCalculatedAnglesInThreePlanes(points3D));  

                % Store calculated angles
                sagittal_hip_ang(idx_raw_data) = lstResThreePlanes.S_ANG_HIP;
                sagittal_plv_ang(idx_raw_data) = lstResThreePlanes.S_ANG_PEL;
                sagittal_kne_ang(idx_raw_data) = lstResThreePlanes.S_ANG_KNE;
                sagittal_ank_ang(idx_raw_data) = lstResThreePlanes.S_ANG_ANK;
                frontal_hip_ang(idx_raw_data) = lstResThreePlanes.F_ANG_HIP;
                frontal_plv_ang(idx_raw_data) = lstResThreePlanes.F_ANG_PEL;
                frontal_kne_ang(idx_raw_data) = lstResThreePlanes.F_ANG_KNE;
                frontal_ank_ang(idx_raw_data) = lstResThreePlanes.F_ANG_ANK;
                transversal_hip_ang(idx_raw_data) = lstResThreePlanes.T_ANG_HIP;
                transversal_plv_ang(idx_raw_data) = lstResThreePlanes.T_ANG_PEL;
                transversal_kne_ang(idx_raw_data) = lstResThreePlanes.T_ANG_KNE;
                transversal_ank_ang(idx_raw_data) = lstResThreePlanes.T_ANG_ANK;

                % Save previous angles for continuity in data
                prev_S_ANG_HIP = lstResThreePlanes.S_ANG_HIP;
                prev_S_ANG_PEL = lstResThreePlanes.S_ANG_PEL;
                prev_S_ANG_KNE = lstResThreePlanes.S_ANG_KNE;
                prev_S_ANG_ANK = lstResThreePlanes.S_ANG_ANK;
                prev_F_ANG_HIP = lstResThreePlanes.F_ANG_HIP;
                prev_F_ANG_PEL = lstResThreePlanes.F_ANG_PEL;
                prev_F_ANG_KNE = lstResThreePlanes.F_ANG_KNE;
                prev_F_ANG_ANK = lstResThreePlanes.F_ANG_ANK;
                prev_T_ANG_HIP = lstResThreePlanes.T_ANG_HIP;
                prev_T_ANG_PEL = lstResThreePlanes.T_ANG_PEL;
                prev_T_ANG_KNE = lstResThreePlanes.T_ANG_KNE;
                prev_T_ANG_ANK = lstResThreePlanes.T_ANG_ANK;

                % Encode kinematic data and marker positions to JSON
                kinematicsAnalysisGaitAngles = ...
                    jsonencode(table(sagittal_hip_ang, sagittal_plv_ang, ...
                                     sagittal_kne_ang, sagittal_ank_ang, ...
                                     frontal_hip_ang, frontal_plv_ang, ...
                                     frontal_kne_ang, frontal_ank_ang, ...
                                     transversal_hip_ang, transversal_plv_ang, ...
                                     transversal_kne_ang, transversal_ank_ang));

                rawMarkersPositions = ...
                    jsonencode(table(lbwt_x, lbwt_y, lbwt_z, ...
                                     lfwt_x, lfwt_y, lfwt_z, ...
                                     ltrc_x, ltrc_y, ltrc_z, ...
                                     lkne_x, lkne_y, lkne_z, ...
                                     lank_x, lank_y, lank_z, ...
                                     lhee_x, lhee_y, lhee_z, ...
                                     lteo_x, lteo_y, lteo_z));

                kinematicDataToServerAsJson = ...
                    cookKinematicData(kinematicsAnalysisGaitAngles, rawMarkersPositions); 
            end    

            % Increment raw data index
            idx_raw_data = idx_raw_data +1; 

            %% Create and visualize the point cloud
            ptCloud = pointCloud(points3D);
            view(player3D, ptCloud);

            % Update plots for marker oscillation if enabled
            if OSCILATION
                UpdatePlotOscilation(HX, PX, points3D(3,1), startTime,1);
                UpdatePlotOscilation(HY, PY, points3D(3,2), startTime,2);
                UpdatePlotOscilation(HZ, PZ, points3D(3,3), startTime,3);
            end

        else
            % If marker count mismatches, use previous angles
            sagittal_hip_ang(idx_raw_data) = prev_S_ANG_HIP;
            sagittal_plv_ang(idx_raw_data) = prev_S_ANG_PEL;
            sagittal_kne_ang(idx_raw_data) = prev_S_ANG_KNE;
            sagittal_ank_ang(idx_raw_data) = prev_S_ANG_ANK;
            frontal_hip_ang(idx_raw_data) = prev_F_ANG_HIP;
            frontal_plv_ang(idx_raw_data) = prev_F_ANG_PEL;
            frontal_kne_ang(idx_raw_data) = prev_F_ANG_KNE;
            frontal_ank_ang(idx_raw_data) = prev_F_ANG_ANK;
            transversal_hip_ang(idx_raw_data) = prev_T_ANG_HIP;
            transversal_plv_ang(idx_raw_data) = prev_T_ANG_PEL;
            transversal_kne_ang(idx_raw_data) = prev_T_ANG_KNE;
            transversal_ank_ang(idx_raw_data) = prev_T_ANG_ANK;

            % Increment raw data index
            idx_raw_data = idx_raw_data +1; 
        end
    end

    % Display binary frames
    step(player_left, frameLeftBinary);
    step(player_right, frameRightBinary);

    % Increment frame counter
    iteFrames = iteFrames + 1;
end

% Clean up by resetting video players
reset(player_left);
reset(player_right);

%% Oscilation Plot function (x,y,z)
% Initializes plot for oscillation visualization
function [H, P] = InitPlotOscilation(L_I, L_S, TITLE)
    figure
    h = animatedline; % Create an animated line
    px = gca; % Get current axis
    px.YGrid = 'on'; % Enable Y grid
    px.YLim = [L_I, L_S]; % Set Y limits
    title(TITLE); % Set plot title
    xlabel('num frames'); % Label X-axis
    ylabel('meters'); % Label Y-axis
    
    H = h; % Return handle to animated line
    P = px; % Return handle to plot
end

% Updates oscillation plot with new data
function UpdatePlotOscilation(H, P, PTCLOUD, STARTIME, POS_POT)
    t = datetime('now') - STARTIME; % Calculate elapsed time
    pos = double(PTCLOUD); % Convert position to double
    addpoints(H, datenum(t), pos); % Add new point to animated line
    P.XLim = datenum([t-seconds(15), t]); % Update X-axis limits
    datetick('x', 'keeplimits') % Keep limits while updating ticks
    drawnow % Force drawing of graphics
end

%% Marker tracking using background subtraction
function [medFiltImage, binaryImage] = DetectMarkers_BackgroundSubtraction(backgroundLeft, frameLeft, BINARY_THRES)
    [BackgroundLeft_hsv] = round(rgb2hsv(backgroundLeft)); % Convert background to HSV
    [FrameLeft_hsv] = round(rgb2hsv(frameLeft)); % Convert frame to HSV
    
    Out = bitxor(BackgroundLeft_hsv, FrameLeft_hsv); % XOR to find differences
    subplot(2, 2, 1), imshow(Out), title('diff - bitxor');
    
    Out = rgb2gray(Out); % Convert to grayscale
    subplot(2, 2, 2), imshow(Out), title('diff Gray scale');
    
    [rows, columns] = size(Out); % Get image dimensions

    % Convert to binary image based on threshold
    for i = 1:rows
        for j = 1:columns
            if Out(i, j) > BINARY_THRES
                binaryImage(i, j) = 1;
            else
                binaryImage(i, j) = 0;
            end
        end
    end
    subplot(2, 2, 3), imshow(binaryImage), title('binaryImage');
    
    medFiltImage = medfilt2(binaryImage); % Apply median filter
    subplot(2, 2, 4), imshow(medFiltImage), title('MedFilteredImage');
end

% Applies Gaussian filter to an image
function img = Gaussianfilter(FILENAME)
    I = FILENAME; % Use input image
    Iblur = imgaussfilt(I, 6); % Apply Gaussian filter
    
    subplot(1, 2, 1), imshow(I), title('Original Image');
    subplot(1, 2, 2), imshow(Iblur)
    title('Gaussian filtered image, \sigma = 1')
    
    img = Iblur; % Return filtered image
end

% Dummy version of feature point detection
function matchedPoints = DetectFeaturedPoints_DummyVersion_Sum_X_Y(I1, THRESH, MIN_AREA, MAX_AREA)
    GI1 = rgb2gray(I1); % Convert to grayscale
    BW1_TH = GI1 > THRESH; % Apply threshold

    BW1 = bwconncomp(BW1_TH); % Find connected components
    stats = regionprops(BW1, 'Area', 'Eccentricity'); % Get properties
    outOfPhase = find([stats.Area] > MIN_AREA & [stats.Area] < MAX_AREA); 
    BW1_BLOB = ismember(labelmatrix(BW1), outOfPhase); % Filter blobs

    CC1 = regionprops(BW1_BLOB, 'centroid'); % Find centroids
    matchedPoints = single(cat(1, CC1.Centroid)); % Extract points

    % Sort points by sum of X and Y values
    if length(matchedPoints) > 1
        [rows, ~] = size(matchedPoints);
        tempMat = [];
        for i = 1:rows
            newRow = [(matchedPoints(i, 1) + matchedPoints(i, 2)), matchedPoints(i, 1), matchedPoints(i, 2)];
            tempMat = cat(1, tempMat, newRow);
        end

        out = sortrows(tempMat);
        out(:,1) =[];
        matchedPoints = out;    
    end
end

function [conv2Img BinaryImage markerPoints] =ExtractPositions(I1, THRESH_MIN, THRESH_MAX, MIN_AREA, MAX_AREA)
    figure
    subplot(2,3,1), imshow(I1), title('color img');
    
    % Convert to grayscale
    GI1 = rgb2gray(I1);
    subplot(2,3,2), imshow(GI1), title('gray img');
    
    % Apply convolution to highlight features
    s = [-1.0  -1.0  -1.0; 
         -1.0  8.0  -1.0; 
         -1.0  -1.0  -1.0];
    conv2Img = conv2(GI1, s, 'same');
    
    subplot(2,3,3), imshow(conv2Img), title('conv2 img');
    
    % Apply threshold
    BW1_TH = conv2Img > THRESH_MIN & conv2Img < THRESH_MAX;
    subplot(2,3,4), imshow(BW1_TH), title('Thresholded Image');
    
    % Detect connected components
    BW1 = bwconncomp(BW1_TH); 
    
    % Filter blobs by area
    stats = regionprops(BW1, 'Area', 'Eccentricity'); 
    outOfPhase = find([stats.Area] > MIN_AREA & [stats.Area] < MAX_AREA); 
    BW1_BLOB = ismember(labelmatrix(BW1), outOfPhase);
    subplot(2,3,6), imshow(BW1_BLOB), title('Filtered Blobs');
    
    % Calculate centroids of filtered blobs
    CC1 = regionprops(BW1_BLOB, 'centroid');
    
    % Return binary image and marker positions
    BinaryImage = BW1_BLOB;
    markerPoints = single(cat(1, CC1.Centroid));
end

% Applies a custom technique for image processing
function res = myTechnique(RGB)
    GAUS = imgaussfilt(RGB, 2); % Apply Gaussian filter
    
    imjt = imadjust(GAUS, [.5 .4 0; .6 .7 1], []); % Adjust image intensity
    
    HSV = rgb2hsv(imjt); % Convert to HSV
    hsv(:,:,1) = ones(720, 1280);
    hsv(:,:,2) = HSV(:,:,2);
    hsv(:,:,3) = ones(720, 1280);
    
    RGB2 = hsv2rgb(hsv); % Convert back to RGB
    
    gray = rgb2gray(RGB2); % Convert to grayscale
    
    sharpen = imsharpen(gray); % Sharpen image
    
    markers = sharpen > 0.5; % Threshold to get markers
    
    bw = imbinarize(sharpen); % Binarize image
    
    res = bw; % Return result
end

function [markerImage, markerPoints] = TestKernels(I1, THRESH_MIN, SURGE_MEAN_AREA, TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM)
    % Define kernel matrices for image processing
    outline = [-1.0, -1.0, -1.0; 
               -1.0,  8.0, -1.0; 
               -1.0, -1.0, -1.0];

    blur = [0.0625, 0.125, 0.0625; 
            0.125,   0.25,  0.125; 
            0.0625, 0.125, 0.0625];

    sharpen = [0.0, -1.0,  0.0; 
              -1.0,  7.0, -1.0; 
               0.0, -1.0,  0.0];

    % Convert image to grayscale and apply threshold
    BW1_TH = THRESH_MIN < rgb2gray(I1);

    % Detect connected components based on the thresholded image
    BW1 = bwconncomp(BW1_TH);

    % Initialize an empty array for blob IDs
    ids = [];
    % Calculate properties of connected components
    stats = regionprops(BW1, 'Area', 'Eccentricity');
    if not (size(stats, 1) < 4)
        % Calculate mean area of blobs
        mean_area = mean([stats.Area]);
        % Find blobs larger than a surge mean area
        ids = find([stats.Area] > mean_area * SURGE_MEAN_AREA);

        % If one blob is removed due to its smallness, adjust the criteria
        if (size(ids) < 3)
            sort_stats = table2struct(sortrows(struct2table(stats), 'Area', 'descend'));
            min_area = sort_stats(3, 1).Area;
            ids = find([stats.Area] >= min_area);
        end
    elseif (size(stats, 1) == 3) % If there are exactly 3 blobs
        ids = [1; 2; 3];
    end

    % Filter blobs based on identified IDs
    marker_blobs = ismember(labelmatrix(BW1), ids);

    % Calculate centroids of the filtered blobs
    CC1 = regionprops(marker_blobs, 'centroid');

    % Return binary image of marker blobs and their centroids
    markerImage = marker_blobs;
    markerPoints = single(cat(1, CC1.Centroid));
end

function trackedPoints = ExtractMarkersPosXY(IMAGE, SURGE_MEAN_AREA)
    % Detect connected components from the binary image
    BW1 = bwconncomp(IMAGE);
    % Calculate properties of connected components
    stats = regionprops(BW1, 'Area', 'Eccentricity');
    % Calculate mean area of blobs
    mean_area = mean([stats.Area]);
    % Find blobs larger than a surge mean area and smaller than a threshold
    ids = find([stats.Area] > mean_area * SURGE_MEAN_AREA);
    ids = find([stats.Area] < mean_area * (1 + (1 - SURGE_MEAN_AREA)));
    % Filter blobs based on identified IDs
    blobs = ismember(labelmatrix(BW1), ids);

    % Calculate centroids of the filtered blobs
    CC1 = regionprops(blobs, 'centroid');

    % Return list of centroids of the tracked markers
    trackedPoints = single(cat(1, CC1.Centroid));
end

%% Marker sorts

function sortList = AscendentSort_SumOfAxisValues(MARKER_POINTS)
    % Check if there are more than one marker point
    if length(MARKER_POINTS) > 1
        % Ascending sort by the sum of X and Y values
        [rows, ~] = size(MARKER_POINTS);
        tempMat = [];
        for i = 1:rows
            newRow = [(MARKER_POINTS(i,1) + MARKER_POINTS(i,2)) ...
                       MARKER_POINTS(i,1) ...
                       MARKER_POINTS(i,2)];
            tempMat = cat(1, tempMat, newRow); 
        end

        out = sortrows(tempMat); % Sort rows based on the sum
        out(:,1) =[]; % Remove the sum column
        sortList = out; % Return the sorted list    
    end
end

function sortList = DescendSort_YAxisValues(MARKER_POINTS)
    % Check if there are more than two marker points
    if length(MARKER_POINTS) > 2
        % Descending sort by Y-axis values
        sortList = sortrows(MARKER_POINTS,-2); 
    end
end

%% Label markers
function labeledMarkers = LabelMarkers2DImages_7(SORTED_LIST_Y, IS_LEFT)
    % Label the last three markers: CALCANIUM, FOOT, and MALLEOLE EXTERN

    matSum = []; 
    % Calculate the distance between the three markers and store in matSum
    for i = 1:3
        sumDist = 0;
        for j = 1:3
            if i ~= j
                dist = sqrt(((SORTED_LIST_Y(i,1) - SORTED_LIST_Y(j,1)) ^ 2) + ...
                            ((SORTED_LIST_Y(i,2) - SORTED_LIST_Y(j,2)) ^ 2));
                sumDist = sumDist + dist; 
            end
        end
        newRow = [i, sumDist];
        matSum = cat(1, matSum, newRow);
    end
    
    % Sort matSum to identify markers based on distance
    matSumOrdered = sortrows(matSum,-2);

    % Copy SORTED_LIST_Y to a temporary list
    temList = SORTED_LIST_Y;
     
    % Assign labels based on distance
    % Highest sum: FOOT MARKER
    SORTED_LIST_Y(1,:) = temList(matSumOrdered(1,1), :);
    % Intermediate sum: CALCANIUM MARKER
    SORTED_LIST_Y(2,:) = temList(matSumOrdered(2,1), :);
    % Lowest sum: MALLEOLE MARKER
    SORTED_LIST_Y(3,:) = temList(matSumOrdered(3,1), :);
    
    % KNEE and TROCHANTER are already correctly positioned by y-axis order
    
    % Compare X values for ASIS and PSIS
    if SORTED_LIST_Y(6,1) > SORTED_LIST_Y(7,1)
        if IS_LEFT
            % Swap if left side
            SORTED_LIST_Y(6,1) = temList(7,1); % ASIS
            SORTED_LIST_Y(7,1) = temList(6,1); % PSIS
        else
            % Keep order if right side
            SORTED_LIST_Y(6,1) = temList(6,1); % ASIS
            SORTED_LIST_Y(7,1) = temList(7,1); % PSIS
        end
    else
        if IS_LEFT
            % Keep order if left side
            SORTED_LIST_Y(6,1) = temList(6,1); % ASIS
            SORTED_LIST_Y(7,1) = temList(7,1); % PSIS
        else
            % Swap if right side
            SORTED_LIST_Y(6,1) = temList(7,1); % ASIS
            SORTED_LIST_Y(7,1) = temList(6,1); % PSIS
        end
    end

    labeledMarkers = SORTED_LIST_Y; % Return labeled markers
end

function labeledMarkers = LabelMarkers2DImages_9(SORTED_LIST_Y)

    % This function labels the last three markers in the list, which
    % represent the ankle, feet, and malleole extern. It calculates the
    % distance in space between these markers to assist in labeling.

    % Initialize matrix to store sum of distances for each marker
    matSum = []; 
    
    % Iterate through each of the last three markers
    for i = 1:3
        sumDist = 0; % Initialize sum of distances for current marker
        % Calculate distance from current marker to the other two
        for j = 1:3
            if i ~= j
                dist = sqrt(((SORTED_LIST_Y(i,1) - SORTED_LIST_Y(j,1)) ^ 2) + ...
                            ((SORTED_LIST_Y(i,2) - SORTED_LIST_Y(j,2)) ^ 2));
                sumDist = sumDist + dist; 
            end
        end
        % Store marker index and its sum of distances
        newRow = [i, sumDist];
        matSum = cat(1, matSum, newRow);
    end
    
    % Sort markers by descending sum of distances
    matSumOrdered = sortrows(matSum, -2);

    % Copy sorted list to a temporary list for manipulation
    temList = SORTED_LIST_Y;
     
    % Assign labels based on sorted sum of distances
    % Highest sum indicates the FEET MARKER
    SORTED_LIST_Y(1,:) = temList(matSumOrdered(1,1), :);
    % Intermediate sum indicates the CALCANIUM MARKER
    SORTED_LIST_Y(2,:) = temList(matSumOrdered(2,1), :);
    % Lowest sum indicates the MALLEOLE MARKER
    SORTED_LIST_Y(3,:) = temList(matSumOrdered(3,1), :);
    
    % Additional markers between foot and hip are to be labeled considering
    % the knee position for Thigh and Shank markers. The sorted list by
    % y-axis assists in this labeling.

    % Compare X values to label ANTERIOR ILIAC SPINE CREST (ASIS) and
    % POSTERIOR SUPERIOR ILIAC CREST (PSIS)
    if SORTED_LIST_Y(8,1) > SORTED_LIST_Y(9,1)
        % Swap if necessary to correctly label ASIS and PSIS
        SORTED_LIST_Y(9,1) = temList(8,1);
        SORTED_LIST_Y(8,1) = temList(9,1);
    end
    
    % Return the labeled markers
    labeledMarkers = SORTED_LIST_Y;
end


%% Methods that turn 3DPoints into dictionaries
function newMapDicPoint3D = GetDicPoint3D(POINT3D)
    % Converts a 3D point into a dictionary with keys 'x', 'y', 'z'.
    % POINT3D: A 1x3 vector representing a 3D point.
    % Returns a dictionary mapping 'x', 'y', 'z' to their respective values.

    [m, n] = size(POINT3D); % Get size of POINT3D
    keySet = {'x', 'y', 'z'}; % Define keys for the dictionary

    for i = 1:n
        valueSet(i) = POINT3D(1,i); % Assign values from POINT3D to valueSet
    end
    newMapDicPoint3D = containers.Map(keySet, valueSet); % Create dictionary
end

function [lbwt, lfwt, ltrc, lkne, lank, lhee, lteo] = GetArrayDicPoints3D(POINTS3D)
    % Converts an array of 3D points into dictionaries for each point.
    % POINTS3D: A Nx3 matrix where each row represents a 3D point.
    % Returns dictionaries for each anatomical landmark.

    lteo = GetDicPoint3D(POINTS3D(1,:)); % Toe
    lhee = GetDicPoint3D(POINTS3D(2,:)); % Heel
    lank = GetDicPoint3D(POINTS3D(3,:)); % Ankle
    lkne = GetDicPoint3D(POINTS3D(4,:)); % Knee
    ltrc = GetDicPoint3D(POINTS3D(5,:)); % Thigh
    lfwt = GetDicPoint3D(POINTS3D(6,:)); % Waist Front
    lbwt = GetDicPoint3D(POINTS3D(7,:)); % Waist Back
end

%% Methods that calculate angles

function listResAngles = GetJsonOfCalculatedAnglesInThreePlanes(POINTS3D)
    % Calculates various angles in sagittal, frontal, and transversal planes
    % from given 3D points and encodes the results into a JSON string.
    % POINTS3D: A Nx3 matrix of 3D points representing anatomical landmarks.
    % Returns a JSON string of calculated angles in three planes.

    [lbwt, lfwt, ltrc, lkne, lank, lhee, lteo] = GetArrayDicPoints3D(POINTS3D);

    % Calculate angles in sagittal plane
    S_ANG_HIP = CalculateHipAnglesSagittal(lbwt, lfwt, ltrc, lkne);
    S_ANG_PEL = CalculatePelvicAnglesSagittal(lbwt, lfwt);
    S_ANG_KNE = CalculateKneeAnglesSagittal(ltrc, lkne, lank);
    S_ANG_ANK = CalculateAnkleAnglesSagittal(lkne, lank, lhee, lteo);

    % Frontal and transversal angles are calculated but currently use
    % sagittal calculations as placeholders
    F_ANG_HIP = S_ANG_HIP;
    F_ANG_PEL = S_ANG_PEL;
    F_ANG_KNE = S_ANG_KNE;
    F_ANG_ANK = S_ANG_ANK;

    T_ANG_HIP = S_ANG_HIP;
    T_ANG_PEL = S_ANG_PEL;
    T_ANG_KNE = S_ANG_KNE;
    T_ANG_ANK = S_ANG_ANK;

    % Encode angles into a JSON string
    listResAngles = jsonencode(containers.Map({
        'S_ANG_HIP', 'S_ANG_PEL', 'S_ANG_KNE', 'S_ANG_ANK',
        'F_ANG_HIP', 'F_ANG_PEL', 'F_ANG_KNE', 'F_ANG_ANK',
        'T_ANG_HIP', 'T_ANG_PEL', 'T_ANG_KNE', 'T_ANG_ANK'
    }, {
        S_ANG_HIP, S_ANG_PEL, S_ANG_KNE, S_ANG_ANK,
        F_ANG_HIP, F_ANG_PEL, F_ANG_KNE, F_ANG_ANK,
        T_ANG_HIP, T_ANG_PEL, T_ANG_KNE, T_ANG_ANK
    }));
end

function ang = HCCalculateHipAnglesSagittal()
    % Calculates hip angles in the sagittal plane using hardcoded vectors.
    % Returns the calculated angle adjusted to represent the hip angle.

    % Define vectors for top and bottom segments
    v_top = [10, 0] - [-10, 0];
    v_bott  = [0, -1] - [10, -1];
    
    % Inverted vectors for alternative calculation (unused)
    v_topI = [-10, 0] - [10, 0];
    v_bottI  = [10, -1] - [0, -10];
    
    % Calculate angles between vectors
    ang = CalculateAngles2Vectors(v_top, v_bott, 'Absolute');
    angI = CalculateAngles2Vectors(v_topI, v_bottI, 'Absolute'); % Unused
    
    % Adjust angle to represent hip angle in sagittal plane
    ang = 90 - ang; % Adjust for perpendicular to spine
end
function absAnglekevin = CalculateAbsoluteAngle(leftPoint,rightPoint)
    % Calculates the absolute angle between two points with respect to a
    % horizontal line. This function uses the 'CalculateAbsoluteAnglekevinVersion'
    % for the actual calculation and returns the absolute angle.
    %
    % Parameters:
    % leftPoint: The origin point (dictionary with 'x' and 'y' keys).
    % rightPoint: The target point (dictionary with 'x' and 'y' keys).
    %
    % Returns:
    % absAnglekevin: The absolute angle between the two points.

    % Calculate opposite and adjacent sides of the triangle
    opp = rightPoint('y') - leftPoint('y'); % Opposite side
    adj = rightPoint('x') - leftPoint('x'); % Adjacent side
    
    % Calculate angle in radians
    tethaRad = atan(opp/adj);
    
    % Convert radians to degrees
    ang = RadToDeg(tethaRad);
    
    % Extract coordinates for direct method comparison
    x1 = leftPoint('x');
    y1 = leftPoint('y');    
    x2 = rightPoint('x');
    y2 = rightPoint('y');

    % Use the 'CalculateAbsoluteAnglekevinVersion' for calculation
    [absAnglekevin, ~] = CalculateAbsoluteAnglekevinVersion(x1, y1, x2, y2);
end
                                    
function [absAnglekevin, angleNormal] = CalculateAbsoluteAnglekevinVersion(x1, y1, x2, y2)
    % Calculates the absolute angle between two points considering the
    % quadrant in which the angle is located. This method ensures correct
    % angle calculation in all quadrants.
    %
    % Parameters:
    % x1, y1: Coordinates of the first point (Master).
    % x2, y2: Coordinates of the second point (Slave).
    %
    % Returns:
    % absAnglekevin: The absolute angle considering the quadrant.
    % angleNormal: The angle without quadrant consideration.

    % Calculate opposite and adjacent sides of the triangle
    opp = y2 - y1; % Opposite side
    adj = x2 - x1; % Adjacent side
    
    % Calculate angle in radians and then convert to degrees
    tethaRad = atan(opp/adj);
    ang = RadToDeg(tethaRad);

    % Initial angle assignment
    angleNormal = ang;
    absAnglekevin = ang; 

    % Adjust angle based on quadrant to ensure correctness
    if ang < 0 && y1 > y2
        absAnglekevin = -1 * ang; % Quadrant I
    elseif ang > 0 && y1 > y2 
        absAnglekevin = 180 - ang; % Quadrant II
    elseif ang < 0 && y1 < y2
        absAnglekevin = 180 + ang; % Quadrant III
    elseif ang > 0 && y1 < y2 
        absAnglekevin = ang; % Quadrant IV
    end
end

function angConv = RadToDeg(rad)
    % Converts radians to degrees.
    %
    % Parameters:
    % rad: Angle in radians.
    %
    % Returns:
    % angConv: Angle in degrees.

    angConv = rad * 180 / pi; % Conversion formula
end

%% Sagittal Angles calculation
% theorically done OK
function ang = CalculatePelvicAnglesSagittal(LBWT, LFWT)
    % Pelvic obliquity
    % Absolute
    
    % Pelvic tilt is normally calculated about the laboratory's transverse axis.
    % If the subject's direction of forward progression is closer to the 
    % laboratory's sagittal axis, however, then pelvic tilt is measured about
    % this axis. The sagittal pelvic axis, which lies in the pelvis transverse 
    % plane, is normally projected into the laboratory sagittal plane. Pelvic 
    % tilt is measured as the angle in this plane between the projected sagittal
    % pelvic axis and the sagittal laboratory axis. A positive value (up) 
    % corresponds to the normal situation in which the PSIS is higher than 
    % the ASIS.

    lfwtAbs = CalculateAbsoluteAngle(LFWT, LBWT);
    ang = lfwtAbs;
end

% theorically done OK, try absolute values
function ang = CalculateHipAnglesSagittal(LBWT, LFWT,LTRC, LKNE)
    % Hip flexion/extension
    % Relative

    % Hip flexion is calculated about an axis parallel to the pelvic transverse
    % axis which passes through the hip joint centre. The sagittal thigh axis 
    % is projected onto the plane perpendicular to the hip flexion axis. Hip 
    % flexion is then the angle between the projected sagittal thigh axis and
    % the sagittal pelvic axis. A positive (Flexion) angle value corresponds to 
    % the situation in which the knee is in front of the body.

    % the fact that is left kinematic analysis means that both vectors 
    % should point from left to rigth (->) to get the intenral angle
    
    sagittalPelvicAxis = [LFWT('x'), LFWT('y')] - [LBWT('x'), LBWT('y')];
    thighAxis  = [LKNE('x'), LKNE('y')] - [LTRC('x'), LTRC('y')];  
    
    ang = CalculateAngles2Vectors(sagittalPelvicAxis, thighAxis, 'relative');
    
    % then 90 degress must be substracted to get the rigth angle which is
    % formed beteen the thighAxis and the perpendicular axis to sagittalPelvicAxis 
    ang = 90 - ang;

    lkneAbs = CalculateAbsoluteAngle(LKNE,LTRC);
    lfwtAbs = CalculateAbsoluteAngle(LFWT, LBWT);

    hipRef = ( 90 - lkneAbs) + lfwtAbs;
    ang = hipRef;

end

% theorically done OK, try both methods vec  &  3 points
function ang = CalculateKneeAnglesSagittal(LTRC, LKNE, LANK)
    % Knee flexion/extension
    % Relative

    % The sagittal shank axis is projected into the plane perpendicular to 
    % the knee flexion axis. Knee flexion is the angle in that plane between 
    % this projection and the sagittal thigh axis. The sign is such that a 
    % positive angle corresponds to a flexed knee.

    lkneAbs = CalculateAbsoluteAngle(LKNE,LTRC);
    lankAbs = CalculateAbsoluteAngle(LANK,LKNE);
    lankRel = lankAbs - lkneAbs;
    ang = lankRel;

end

% theorically done,
function ang = CalculateAnkleAnglesSagittal(LKNE, LANK, LHEE, LTOE)
    % Ankle dorsi/plantar flexion
    % Relative
    
    % The foot vector is projected into the foot sagittal plane. The angle 
    % between the foot vector and the sagittal axis of the shank is the foot
    % dorsi/plantar flexion. A positive number corresponds to dorsiflexion.

    % the fact that It is left kinematic analysis means that both vectors 
    % should point from left to rigth (->) to get the internal angle

    shankAxis  = [LANK('x'), LANK('y')] - [LKNE('x'), LKNE('y')]; 
    retropieAxis  = [LTOE('x'), LTOE('y')] - [LHEE('x'), LHEE('y')]; 
    
    ang = CalculateAngles2Vectors(shankAxis, retropieAxis, 'Absolute');
    ang = 90 - ang; % sin perpendicular to Eilic spain

    lheeAbs = CalculateAbsoluteAngle(LHEE,LTOE);
    lankAbs = CalculateAbsoluteAngle(LANK,LKNE);

    ang = lheeAbs - (90 + lankAbs);

end

%% Frontal Angles Calculation

% theorically done
function ang = CalculatePelvicAnglesFrontal(LFWT, RFWT)
    % Pelvic obliquity
    % Absolute
    
    % Pelvic obliquity is measured about an axis of rotation perpendicular to
    % the axes of the other two rotations. This axis does not necessarily
    % correspond with any of the laboratory or pelvic axes. Pelvic obliquity
    % is measured in the plane of the laboratory transverse axis and the pelvic
    % frontal axis. The angle is measured between the projection into the plane
    % of the transverse pelvic axis and projection into the plane of the 
    % laboratory transverse axis (the horizontal axis perpendicular to the
    % subject's axis of progression). A negative pelvic obliquity value (down)
    % relates to the situation in which the opposite side of the pelvis is lower.

    % the fact that is left kinematic analysis means that both vectors 
    % should point from left to right (->) to get the internal angle
    
    transversePelvicAxis = [LFWT('z'), LFWT('y')] - [RFWT('z'), RFWT('y')];
    labTransverseAxis  = [LFWT('z'), LFWT('y')] - [9999, LFWT('y')];  
    
    ang = CalculateAngles2Vectors(transversePelvicAxis, labTransverseAxis, 'relative');
end

% theorically done 
function ang = CalculateHipAnglesFrontal(LFWT, RFWT, LTRC, LKNE)
    % Hip ab/adduction
    % Relative
    
    % Hip adduction is measured in the plane of the hip flexion axis and the 
    % knee joint centre. The angle is calculated between the long axis of the 
    % thigh and the frontal axis of the pelvis projected into this plane. 
    % A positive number corresponds to an adducted (inwardly moved) leg.   
    
    pelvicFrontalAxis = [LFWT('z'), LFWT('y')] - [LFWT('z'), 9999];
    thighAxis  = [LTRC('z'), LTRC('y')] - [LKNE('z'), LKNE('y')];  
    
    ang = CalculateAngles2Vectors(pelvicFrontalAxis, thighAxis, 'relative');
end

% theorically done, add an implement to LANK marker
function ang = CalculateKneeAnglesFrontal(LTRC, LKNE, LANK)
    % Knee ab/adduction (Knee valgus/varus)
    % Relative

    % This is measured in the plane of the knee flexion axis and the ankle 
    % center, and is the angle between the long axis of the shank and the long 
    % axis of the thigh projected into this plane.
    % A positive number corresponds to varus (outward bend of the knee).

    P_TOP  = [LTRC('z'), LTRC('y')];  
    P_CENT = [LKNE('z'), LKNE('y')];

    % LANK marker will not be visible by frontal IR cameras (add a bar with
    % markers on each extream) =>.  (O)=====o=====(O)
    P_BOTT = [LANK('z'), LANK('y')];  
    
    % or try with vevtors
    ang = CalculateAngle3Points(P_TOP, P_CENT, P_BOTT, 'relative');
end

% therically done, check the signal of (+/-)9999 hardcode value
function ang = CalculateAnkleAnglesFrontal(LHEE, LTOE)
    % Assessing the foot progression angle (FPA) during gait is an important 
    % part of a clinician's examination. The FPA is defined as the angle made
    % by the long axis of the foot from the heel to 2nd metatarsal and the line 
    % of progression of gait. A negative FPA indicates in-toeing and a positive
    % FPA out-toeing.
    
    pregressionAxis  = [LHEE('z'), LHEE('y')] - [LKNE('z'), -9999]; 
    retropieAxis  = [LHEE('z'), LHEE('y')] - [LTOE('z'), LTOE('y')]; 
    
    ang = CalculateAngles2Vectors(pregressionAxis, retropieAxis, 'Absolute');
end

%% Transversal Angles
% therically done, check the signal of (+/-)9999 hardcode value
function ang = CalculateAnkleAnglesRotation_v1(LHEE, LTOE)
    % Foot progression
    % Absolute

    % This is the angle between the foot vector (projected into the laboratory's 
    % transverse plane) and the sagittal laboratory axis. A positive number 
    % corresponds to an internally rotated foot.
    
    sagittallLabAxis  = [LHEE('z'), LHEE('x')] - [LHEE('z'), -9999]; 
    footVector  = [LHEE('z'), LHEE('x')] - [LTOE('z'), LTOE('x')]; 
    
    ang = CalculateAngles2Vectors(sagittallLabAxis, footVector, 'Absolute');
end

% therically done
function ang = CalculateAnkleAnglesRotation_v2(LKNE, LANK,LHEE, LTOE)
    % Foot rotation
    % Relative

    % This is measured about an axis perpendicular to the foot vector and the 
    % ankle flexion axis. It is the angle between the foot vector and the 
    % sagittal axis of the shank, projected into the foot transverse plane. 
    % A positive number corresponds to an internal rotation.
    
    sagittallAxisShank  = [LANK('z'), LANK('x')] - [LKNE('z'), LKNE('x')]; 
    footVector  = [LHEE('z'), LHEE('x')] - [LTOE('z'), LTOE('x')]; 
    
    ang = CalculateAngles2Vectors(sagittallAxisShank, footVector, 'Absolute');
end

% therically done
function ang = CalculateKneeAnglesRotation(LTRC, LKNE, LANK)
    % Knee rotation
    % Relative

    % Knee rotation is measured about the long axis of the shank. It is 
    % measured as the angle between the sagittal axis of the shank and 
    % the sagittal axis of the thigh, projected into a plane perpendicular
    % to the long axis of the shank. The sign is such that a positive 
    % angle corresponds to internal rotation. If a tibial torsion value 
    % is present in the Session form, it is subtracted from the calculated
    % knee rotation value. A positive tibial torsion value therefore has 
    % the effect of providing a constant external offset to knee rotation.

    P_TOP  = [LTRC('z'), LTRC('x')];  
    P_CENT = [LKNE('z'), LKNE('x')];

    % LANK marker will not be visible by frontal IR cameras (add a bar with
    % markers on each extream) =>.  (O)=====o=====(O)
    P_BOTT = [LANK('z'), LANK('x')];  
    
    % or try with vevtors
    ang = CalculateAngle3Points(P_TOP, P_CENT, P_BOTT, 'relative');
end

% theorically, not sure & check the signal of (+/-)9999 hardcode value
function ang = CalculateHipAnglesRotation(LFWT, RFWT, LTRC, LKNE)
    % Hip rotation
    % Relative
    
    % Hip rotation is measured about the long axis of the thigh segment 
    % and is calculated between the sagittal axis of the thigh and the 
    % sagittal axis of the pelvis projected into the plane perpendicular 
    % to the long axis of the thigh. The sign is such that a positive
    % hip rotation corresponds to an internally rotated thigh.
    
    pelvicFrontalAxis = [LFWT('z'), LFWT('x')] - [LFWT('z'), 9999];
    thighAxis  = [LTRC('z'), LTRC('x')] - [LKNE('z'), LKNE('x')];  
    
    ang = CalculateAngles2Vectors(pelvicFrontalAxis, thighAxis, 'relative');
end

% theorically done, check the signal of (+/-)9999 hardcode value
function ang = CalculatePelvicAnglesRotation(LFWT, RFWT)
    % Pelvic rotation
    % Absolute
    
    % Pelvic rotation is calculated about the frontal axis of the pelvic 
    % co-ordinate system. It is the angle measured between the sagittal 
    % axis of the pelvis and the sagittal laboratory axis (axis closest 
    % to subject's direction of progression) projected into the pelvis 
    % transverse plane. A negative (external) pelvic rotation value means 
    % the opposite side is in front

    % the fact that is left kinematic analysis means that both vectors 
    % should point from left to right (->) to get the internal angle
    
    sagittalAxisPlevis = [LFWT('z'), LFWT('x')] - [RFWT('z'), RFWT('x')];
    sagittalLabAxis  = [LFWT('z'), LFWT('x')] - [LFWT('y') , 9999];  
    
    ang = CalculateAngles2Vectors(sagittalAxisPlevis, sagittalLabAxis, 'relative');

    ang = ang - 90;
end

% source => angles desciption
% https://docs.vicon.com/display/Nexus25/Plug-in+Gait+kinematic+variables#Plug-inGaitkinematicvariables-Completepelvispositiondescription

%% Raw angles calculation
function angle = CalculateAngle3Points(P_TOP, P_CENT, P_BOTT, TYPE)

    % calculates the angle between the lines from P0 to P1 and P0 to P2.
    % P0 = [x0, y0];  
    % P1 = [x1, y1];
    % P2 = [x2, y2];
    n1 = (P_BOTT - P_CENT) / norm(P_BOTT - P_CENT);  % Normalized vectors
    n2 = (P_TOP - P_CENT) / norm(P_TOP - P_CENT);
    % angle1 = acos(dot(n1, n2));           % Instable at (anti-)parallel n1 and n2
    % angle2 = asin(norm(cropss(n1, n2));   % Instable at perpendiculare n1 and n2
    angle = atan2(norm(det([n2; n1])), dot(n1, n2)) * 180/pi ;  % Stable
    if strcmp (TYPE,'relative') 
        angle = 180 - angle ;  % Stable
    end
end


function ang = CalculateAngles2Vectors(V_TOP, V_BOTT, TYPE)
    % CalculateAngles2Vectors - Calculates the angle between two vectors.
    % Inputs:
    %   V_TOP - The first vector.
    %   V_BOTT - The second vector.
    %   TYPE - The type of calculation (not used in current implementation).
    % Outputs:
    %   ang - The angle between the two vectors in degrees.
    ang = (acos(sum(V_TOP.*V_BOTT)/(norm(V_TOP)*norm(V_BOTT))))  * 180/pi;
end

function [ Result ] = Factorial2( Value )
    % Factorial2 - Recursively calculates the factorial of a number.
    % Inputs:
    %   Value - The number to calculate the factorial of.
    % Outputs:
    %   Result - The factorial of the input number.
    if Value > 1
        Result = Factorial2(Value - 1) * Value;
    else
        Result = 1;
    end
end

function img = filterRGBChannels(IMG)
    % filterRGBChannels - Filters RGB channels to highlight certain features.
    % Inputs:
    %   IMG - The input RGB image.
    % Outputs:
    %   img - The output image after filtering.
    r = IMG(:,:,1);
    g = IMG(:,:,2);
    b = IMG(:,:,3);

    % Example of creating a white image placeholder
    whiteImage = 255 * ones(720,1280, 'uint8');

    % Example of filtering based on green channel
    img = 100 < g;

    % Convert to grayscale and apply threshold
    GI1 = rgb2gray(IMG);
    BW1_TH = 160 < GI1;

    % Fuse images for visualization
    k1 = imfuse(BW1_TH, 50 < g, 'montage');
    imshow(k1); 

    % Placeholder variable (not used)
    INT = 1;
end

function res = postTest_weird_error (DATA)
    % postTest_weird_error - Sends a POST request with JSON data.
    % Inputs:
    %   DATA - The data to be sent in the request.
    % Outputs:
    %   res - The response from the server (not explicitly returned).

    % Define the request method
    method = matlab.net.http.RequestMethod.POST;

    % Set content type and token headers
    contentType = matlab.net.http.HeaderField('ContentType','application/json');
    token = matlab.net.http.HeaderField('x-access-token','xxxxxxxx');
    header = [contentType token];

    % Prepare the data for sending
    input = struct('kinematics_analysis_id','58327f939d4fe93d29435260');
    paramsInput = struct('params', input);

    % Encode the data as JSON
    paramsInput = jsonencode(DATA);
 
    % Create the message body
    body = {json, paramsInput};
    body = matlab.net.http.MessageBody(body);
 
    % Display the prepared body (for debugging)
    disp(body)

    % Create and send the request
    request = matlab.net.http.RequestMessage(method, header, body);
    uri = matlab.net.URI('URL');
    response = send(request, uri);

    % Display the response (for debugging)
    show(response)
    disp(response)
end

function call_get_method_test_OK()
    % call_get_method_test_OK - Sends a GET request to a specified URI.
    % This function demonstrates how to send a simple GET request using
    % MATLAB's HTTP interface. It sets up the request headers, including
    % content type and a placeholder token, and sends the request to a
    % predefined URI.

    % Define the request method as GET
    method = matlab.net.http.RequestMethod.GET;

    % Set up the request headers
    contentType = matlab.net.http.HeaderField('ContentType','application/json');
    token       = matlab.net.http.HeaderField('x-access-token','xxxxxxxx');
    header = [contentType token];

    % Create the request message
    request = matlab.net.http.RequestMessage(method,header);

    % Specify the URI to send the request to
    uri = matlab.net.URI('http://52.89.123.49:8080/api/kinematics_analysis_matlab');
    
    % Send the request and receive the response
    response = send(request,uri);
end

function call_put_method_from_onlineMatlab_NO_RESPONSE()
    % call_put_method_from_onlineMatlab_NO_RESPONSE - Sends a PUT request
    % with JSON data to a specified URI but does not receive a response.
    % This function is intended to demonstrate sending a PUT request with
    % a JSON payload. It includes a placeholder for the data to be sent
    % and demonstrates setting up the request headers.

    % Define the URI to send the request to
    uri = matlab.net.URI('http://52.89.123.49:8080/api/kinematics_analysis_matlab/5b43c89ecb275f3fed2e3cde');

    % Set up the request headers
    contentType = matlab.net.http.HeaderField('ContentType','application/json');
    token       = matlab.net.http.HeaderField('x-access-token','xxxxxxxx');
    data = '{"frontal_ank_ang":[[0,36.6239548],[25,36.4931221],[50,37.341877],[75,36.720871],[100,37.5084839]],...}';
    data = matlab.net.http.HeaderField('data',data);
    header = [contentType token data];
    
    % Create and send the PUT request
    request=matlab.net.http.RequestMessage('put',header,matlab.net.http.MessageBody('useless'));
    response=request.send(uri);
end

%% Methods for cooking data to create a json which will be send to gaitcome.con server

function  res = cookKinematicData(LIST_ANGLES,LIST_RAW_POINTS)
    % cookKinematicData - Encodes kinematic data and raw marker positions
    % into a JSON string. This function takes lists of angles and raw
    % marker positions, processes them, and encodes the result into a JSON
    % string that can be sent to a server or used for further processing.

    % Process the list of angles to create a gait cycle percentage object
    lstResAngles = createGaitCycleInPercentageObj(LIST_ANGLES);

    % Decode the list of raw points from JSON
    lstRawPoints = jsondecode(LIST_RAW_POINTS);
    obj = lstRawPoints(1,1);

    % Encode the processed data into a JSON string
    res  = jsonencode(containers.Map({...}, {...}));
end

function res = cookRawData_MarkerPostions(DATA)
    % cookRawData_MarkerPostions - Encodes raw marker positions into a JSON
    % string. This function takes raw marker position data, decodes it from
    % JSON, and re-encodes it into a new JSON string that organizes the
    % data into a specific structure for further use.

    % Decode the raw data from JSON
    data = jsondecode(DATA);
    obj = data(1,1);

    % Encode the data into a new JSON string
    res  = jsonencode(containers.Map({...}, {...}));
end

function  res = cookAnglesWithPercentageProgress(LIST_ANGLES)
    % Converts angle data into a JSON string with percentage progress.
    % This function takes a list of angles for different gait cycles and
    % encodes them into a JSON string. The angles are organized by their
    % anatomical location and phase in the gait cycle.

    % Generate a list of angles with their gait cycle percentage
    lstResAngles = createGaitCycleInPercentageObj(LIST_ANGLES);
    
    % Encode the structured data into a JSON string
    res  = jsonencode( ...
            containers.Map( ...
                {   % Keys representing different angles
                    'sagittal_hip_ang', ...
                    'sagittal_plv_ang', ...
                    'sagittal_kne_ang', ...
                    'sagittal_ank_ang', ...
                    'frontal_hip_ang', ...
                    'frontal_plv_ang', ...
                    'frontal_kne_ang', ...
                    'frontal_ank_ang', ...
                    'transversal_hip_ang', ...
                    'transversal_plv_ang', ...
                    'transversal_kne_ang', ...
                    'transversal_ank_ang' ...
                }, ...
                {   % Values are lists of angles for each key
                    lstResAngles{1}, ...
                    lstResAngles{2}, ...
                    lstResAngles{3}, ...
                    lstResAngles{4}, ...
                    lstResAngles{1}, ...
                    lstResAngles{2}, ...
                    lstResAngles{3}, ...
                    lstResAngles{4}, ...
                    lstResAngles{1}, ...
                    lstResAngles{2}, ...
                    lstResAngles{3}, ...
                    lstResAngles{4} ...
                } ...
            ) ...
        );
end

function lstRes = createGaitCycleInPercentageObj(MULTI_LIST)
    % Creates a structured object from a list of angles with their
    % corresponding gait cycle percentages.
    % This function decodes a JSON string into a list, then processes
    % each element to match it with its gait cycle percentage.

    % Decode the JSON string into a list
    mul_list = jsondecode(MULTI_LIST);
    mul_list = struct2cell(mul_list);

    % Process each list to calculate gait cycle percentages
    lenLst = size(mul_list,1);
    for ii=1:lenLst
        lstRes{ii} = getHighChartsAngleObject_DATA(mul_list{ii});
    end
end

function res = getHighChartsAngleObject_DATA(LIST)
    % Generates data points for HighCharts from a list of angles.
    % This function calculates the gait cycle percentage for each angle
    % and formats it for visualization with HighCharts.

    lenLst = size(LIST,1);
    gaitIntervals = 100/(lenLst-1);  % Calculate interval percentage
  
    for ii=1:lenLst
        % Calculate percentage progress and corresponding angle
        percentualGaitProgress = (ii-1) * gaitIntervals;
        gaitAngle = LIST(ii,1);

        % Store the result in a structured format
        res{ii} = [percentualGaitProgress, gaitAngle];
    end
end

function testPUTPUT(DATA) %OK It sends through online Matlab, since from here has a weird problem
    % Define the URI for the PUT request
    uri = matlab.net.URI('http://52.89.123.49:8080/api/kinematics_analysis_matlab/58327fa19d4fe93d29435261');

    % Set up the content type and token for the request header
    contentType = matlab.net.http.HeaderField('ContentType','application/json');
    token       = matlab.net.http.HeaderField('x-access-token','xxxxxxxx');
    
    % Define the data to be sent in the request
    data = '{"frontal_ank_ang":[[0,36.6239548],[25,36.4931221],[50,37.341877],[75,36.720871],[100,37.5084839]],...}';
    data = matlab.net.http.HeaderField('data',data);
    
    % Combine all header fields
    header = [contentType token data];
    
    % Create the PUT request with the specified header and body
    request=matlab.net.http.RequestMessage('put',header,matlab.net.http.MessageBody('useless'));
    
    % Send the request and store the response
    response=request.send(uri);
end

function testAdjustContrust() 
    % Read an image file for contrast adjustment
    RGB = imread('./visionData/removeBackground/toCalibrate.png');
    imshow(RGB); % Display the original image

    % Apply local Laplacian filtering for contrast adjustment
    sigma = 0.4;
    alpha = 0.8;
    B = locallapfilt(RGB, sigma, alpha);
    
    % Display the original and adjusted images side by side
    imshowpair(RGB, B, 'montage');
end

%% Sync methods
function AnalyzeAudio()
    % Define paths to audio files from two cameras
    a = '../ekenRawFiles/camera_a/test_14_video/FHD0001.MOV';
    b = '../ekenRawFiles/camera_b/test_14_video/FHD0002.MOV'; 

    % Read audio data from files
    a = audioread('audio_test_14_a.WAV');
    b = audioread('audio_test_14_b.WAV');
    
    % Perform FFT on the audio signals
    Y = fft(abs(a));
    Z = fft(abs(b));
    
    % Plot the original audio signals
    subplot(3,1,1), plot(a), title('Audio from Camera A');
    subplot(3,1,2), plot(b), title('Audio from Camera B');
end

function [isSync, outOfPhase_l, outOfPhase_r, isBlinking_l, isBlinking_r] = ...
    DetectOutOfPhaseFrames(FRAME_LEFT, FRAME_RIGHT, OUT_OF_PHASE_L, OUT_OF_PHASE_R, IS_BLINKING_LEFT, IS_BLINKING_RIGHT, SYNC_UMBRAL, TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM)
    % Initialize synchronization status as false
    isSync = false;

    % Check if both left and right frames are blinking
    if(IS_BLINKING_LEFT && IS_BLINKING_RIGHT)
        isSync = true;
    else
        % Detect first blink in the left frame if not already blinking
        if ~IS_BLINKING_LEFT
            [isBlinking_l, blobImg_l] = DetectFirstBlink(FRAME_LEFT, SYNC_UMBRAL, 50, TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM);
            IS_BLINKING_LEFT = isBlinking_l;
            OUT_OF_PHASE_L = OUT_OF_PHASE_L + 1; % Increment out-of-phase counter
        end 

        % Detect first blink in the right frame if not already blinking
        if ~IS_BLINKING_RIGHT
            [isBlinking_r, blobImg_r] = DetectFirstBlink(FRAME_RIGHT, SYNC_UMBRAL, 50, TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM);
            IS_BLINKING_RIGHT = isBlinking_r;
            OUT_OF_PHASE_R = OUT_OF_PHASE_R + 1; % Increment out-of-phase counter
        end
    end

    % Return the updated synchronization status and counters
    outOfPhase_l = OUT_OF_PHASE_L;
    outOfPhase_r = OUT_OF_PHASE_R;
    isBlinking_l = IS_BLINKING_LEFT;
    isBlinking_r = IS_BLINKING_RIGHT;

    % Final determination of synchronization based on blinking detection
end

% Detects the first blink in an image based on threshold and area
function [isBlinking, blobImage] = DetectFirstBlink(I1, THRESH_MIN, MIN_AREA, TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM)
    isBlinking = false; % Initialize blinking flag

    GI1 = rgb2gray(I1); % Convert image to grayscale
    BW1_TH = GI1 > THRESH_MIN; % Apply threshold
    BW1 = bwconncomp(BW1_TH); % Detect connected components
    
    stats = regionprops(BW1, 'Area', 'Eccentricity'); % Get blob properties
    mean_area = mean([stats.Area]); % Calculate mean area
    ids = find([stats.Area] >= MIN_AREA); % Filter blobs by area
    blobImage = ismember(labelmatrix(BW1), ids); % Create binary image of blobs
    
    % Calculate centroids of blobs
    CC1 = regionprops(blobImage, 'centroid');
    markerPoints = single(cat(1, CC1.Centroid));
    
    % Check if any centroids are within specified limits
    for i = 1:size(markerPoints, 1)
       if markerPoints(i, 2) > TOP_LIM && markerPoints(i, 2) < BOT_LIM && ...
          markerPoints(i, 1) > LEF_LIM && markerPoints(i, 1) < RIG_LIM
           isBlinking = true; % Set blinking flag if conditions are met
           break; % Exit loop if a blinking is detected
       end
    end
end

% Determines if an image contains blinking based on threshold
function res = IsBlinking(I1, THRESH_MIN)
    BW1_TH = THRESH_MIN < rgb2gray(I1); % Apply threshold to grayscale image

    BW1 = bwconncomp(BW1_TH); % Detect connected components

    stats = regionprops(BW1, 'Area', 'Eccentricity'); % Get blob properties
    mean_area = mean([stats.Area]); % Calculate mean area
    ids = find([stats.Area] > mean_area); % Filter blobs by area
    
    res = size(ids, 1) >= 1; % Return true if any blobs are found
end

% Saves synchronization data to a text file
function SaveTxtFile(countLeft, countRight)
    fileId = fopen('compareSync.txt', 'w'); % Open file for writing
    fprintf(fileId, 'Compare L - R \n\n'); % Write header
    length = size(countLeft, 2); % Get number of elements
    for i = 1:1:length
        fprintf(fileId, '%f ', countLeft{i}); % Write left count
        fprintf(fileId, '%f \n', countRight{i}); % Write right count
    end
    fclose(fileId); % Close file
end

% Plots blink counts for left and right sides
function PlotBlinks(countLeft, countRight)
    joinMat = [countLeft{:}; countRight{:}]; % Join left and right counts
    
    subplot(3, 1, 1), plot([countLeft{:}]), title('left'); % Plot left counts
    subplot(3, 1, 2), plot([countRight{:}]), title('right'); % Plot right counts
    subplot(3, 1, 3), plot(joinMat'), title('both'); % Plot both counts
end


% Visualizes labels on images
function VisualizeLabels(frameLeft, frameLeftBinary, frameRightBinary, markerSortPoints)
    % Analyze blobs in binary images
    blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
        'AreaOutputPort', false, 'CentroidOutputPort', false, ...
        'MinimumBlobArea', 30);

    % Get bounding boxes for left and right binary images
    bboxLeft = step(blobAnalysis, frameLeftBinary);
    bboxRight = step(blobAnalysis, frameRightBinary);

    % Insert bounding boxes into the left frame image
    result = insertShape(frameLeft, 'Rectangle', bboxLeft, 'LineWidth', 5, 'Color', 'red');

    % Insert number of markers detected
    numMarkers = size(bboxLeft, 1);
    result = insertText(result, [10 10], numMarkers, 'BoxOpacity', 1, 'FontSize', 24);
    

    figure; imshow(result); title('Detected markers');
end




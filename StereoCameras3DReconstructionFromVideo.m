%% clear previou values in memory
clc;
close all;
clear all;

% TRY this http://setosa.io/ev/image-kernels/
% convolution  => linear operations

% Try this
% https://stackoverflow.com/questions/35947438/control-frame-rate-of-video-player-in-vision-system-toolbox



%% Static values
TYPE_PROC = 1;              % 1 or 2 values
N_MARKERS = 7;              % dynamic number markers
NUM_MARKERS = N_MARKERS;    % (N marker), (7 marker) or (9 marker) values
OSCILATION = 0;             % 0 (OFF ) , 1(ON) | It show osilation graphs

% TYPE_PROC = 2
BINARY_THRES_A = 0.4;
BINARY_THRES_B = 0.4868;

MIN_AREA = 2;
MAX_AREA = 60;

% TYPE_PROC = 1
THRESH_L = 100;
THRESH_R = 100;
% THRESH_L = 250;
% THRESH_R = 250;
SURGE_MEAN_AREA = 1;      % multiplier to filter blob sizes detected

% Limits
TOP_LIM = 70;
BOT_LIM = 720 - 70;
LEF_LIM = 70;
RIG_LIM = 1280 - 70;

% Out of Phase frames
isSync = true;
outOfPhase_l =0;             % number of out of phase left frames
outOfPhase_r = 0;             % number of out of phase right frames
isSyncBlob_left_detecteddd = false;
isSyncBlob_right_detecteddd = false;
SYNC_UMBRAL = 250;


% Skip frames 
INI_SEC = 8 ;           % init at second ?
END_SEC = 8.3 ;           % init at second ?


%% Initiate vectors to save 3D raw coordinates, jsonResult, angles
idx_raw_data= 1;                     % index to manage vector position
global lbwt_x; lbwt_x = []; global lbwt_y; lbwt_y = []; global lbwt_z; lbwt_z = [];
global lfwt_x; lfwt_x = []; global lfwt_y; lfwt_y = []; global lfwt_z; lfwt_z = [];
global ltrc_x; ltrc_x = []; global ltrc_y; ltrc_y = []; global ltrc_z; ltrc_z = [];
global lkne_x; lkne_x = []; global lkne_y; lkne_y = []; global lkne_z; lkne_z = [];
global lank_x; lank_x = []; global lank_y; lank_y = []; global lank_z; lank_z = [];
global lhee_x; lhee_x = []; global lhee_y; lhee_y = []; global lhee_z; lhee_z = [];
global lteo_x; lteo_x = []; global lteo_y; lteo_y = []; global lteo_z; lteo_z = [];
% sagittal angels
sagittal_hip_ang = [];
sagittal_pel_ang = [];
sagittal_kne_ang = [];
sagittal_ank_ang = [];
% sagittal angels
frontal_hip_ang = [];
frontal_pel_ang = [];
frontal_kne_ang = [];
frontal_ank_ang = [];
% sagittal angels
transversal_hip_ang = [];
transversal_pel_ang = [];
transversal_kne_ang = [];
transversal_ank_ang = [];

%% Test area
ka = CKinematicAnalysis; 
% img = imread('./visionData/videoCalibration/camera_b/snap_to_webImgeKernel/8711.png');

%% Create Video File Readers and the Video player_left

% Video file reader
videoFileLeft  = ... % camera A
    '../ekenRawFiles/camera_a/test_11_video/FHD0629.MOV'; 
videoFileRight = ... % camera B
    '../ekenRawFiles/camera_b/test_11_video/FHD0623.MOV'; 

% Image file reader
backgroundLeft  = ... % camera A
    imread('./visionData/videoCalibration/camera_b/snap/120.png');          
backgroundRight = ... % camera B
    imread('./visionData/videoCalibration/camera_a/snap/120.png');

% Video players
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


% Oscilation - live data
if OSCILATION == 1
    
    [HX PX] = InitPlotOscilation(-0.3, 0.5,'X - Oscilation');
    [HY PY] = InitPlotOscilation(-0.1, 0.09,'Y - Oscilation');
    [HZ PZ] = InitPlotOscilation(2.9, 3.1,'Z - Oscilation');
    startTime = datetime('now');

end




%% Load cameras stereoParams 
load('./visionData/videoCalibration/stereoParams_test_10_total_error.mat');
% load './visionData/videoCalibration/stereoParams_test_10_total_error';

%% Create a streaming point cloud viewer
player3D =  pcplayer( ...
                [-4, 3], [-1,1], [-0.5, 7], 'VerticalAxis', 'y', ...
                'VerticalAxisDir', 'down', 'MarkerSize', 1300 ...
            );
% player3D = pcplayer( ...
%     [-8, 6], [-10,10], [-4.5, 11], 'VerticalAxis', 'y', ...
%     'VerticalAxisDir', 'down', 'MarkerSize', 2700);
xlabel(player3D.Axes,'x-axis (m)');
ylabel(player3D.Axes,'y-axis (m)');
zlabel(player3D.Axes,'z-axis (m)');
  
%% Reconstruct the 3D locations of matched Points	
% compute camera matrices for each position of the camera	
camMatrixLeft = ...
    cameraMatrix( ...
        stereoParams_test_10_total_error.CameraParameters1, ...
        eye(3), [0 0 0] ...
    );
camMatrixRight = ... 
    cameraMatrix( ...
        stereoParams_test_10_total_error.CameraParameters2, ...
        stereoParams_test_10_total_error.RotationOfCamera2, ...
        stereoParams_test_10_total_error.TranslationOfCamera2 ...
    );
                             
%% Skip frames
%reading a video file
movleft = VideoReader(videoFileLeft);
movRight = VideoReader(videoFileRight);


%getting no of frames
iteFrames = INI_SEC * movleft.FrameRate;
endFrames = END_SEC * movRight.FrameRate;

%% Read pair of frames from video in while loop
while iteFrames < endFrames 

    % Sync process
    while ~isSync

%         retrieve video frames
        frameLeft = read(movleft, iteFrames);
        frameRight = read(movRight, iteFrames );       

        step(player_left, frameLeft);
        step(player_right, frameRight);
        
        [ isSync outOfPhase_l outOfPhase_r isSyncBlob_left_detecteddd isSyncBlob_right_detecteddd ] = ...
                SyncBothFrames(...
                    frameLeft, ...
                    frameRight, ...
                    outOfPhase_l, ...
                    outOfPhase_r, ...
                    isSyncBlob_left_detecteddd, ...
                    isSyncBlob_right_detecteddd, ...
                    SYNC_UMBRAL, ...
                    TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM ...
                )
                    
        iteFrames = iteFrames +1;
        
        if isSync
            release(player_left);
            release(player_right);
        end
    end

    % continue with the process of markers recognition
    % Type of proccessing
    if TYPE_PROC == 1

        frameLeft = read(movleft, iteFrames - outOfPhase_l);
        frameRight = read(movRight, iteFrames - outOfPhase_r); 
        
    elseif TYPE_PROC == 2

        frameLeft_ = read(movleft, iteFrames - outOfPhase_l);
        frameRight_ = read(movRight, iteFrames - outOfPhase_r); 
        
        % % show
        % figure
        % imshowpair(frameLeft,frameRight,'montage');
        % title('L         Original Image         R');

        frameRight = BackgroundSubtraction(backgroundRight, frameRight, BINARY_THRES_B);
        frameLeft = BackgroundSubtraction(backgroundLeft, frameLeft, BINARY_THRES_A);
    end
      


    if TYPE_PROC == 1
        
        [conv2Left frameLeftBinary markersPositionLeft] = ...
            TestKernels(frameLeft,THRESH_L,SURGE_MEAN_AREA, TOP_LIM, ...
            BOT_LIM, LEF_LIM, RIG_LIM);
        [conv2Right frameRightBinary markersPositionRight] = ...
            TestKernels(frameRight,THRESH_R,SURGE_MEAN_AREA, TOP_LIM, ...
            BOT_LIM, LEF_LIM, RIG_LIM);
        
    elseif TYPE_PROC == 2 
        
        markersPositionLeft = ...
            ExtractPositionFromFilteredImage(frameLeft,MIN_AREA,MAX_AREA);
        markersPositionRight = ...
            ExtractPositionFromFilteredImage(frameRight,MIN_AREA,MAX_AREA);
        
    end
      


    %% Remove lens distortion
    
    frameLeft = ...
        undistortImage( ...
            frameLeft, ...
            stereoParams_test_10_total_error.CameraParameters1 ...
        );
    frameRight = ...
        undistortImage( ...
            frameRight, ...
            stereoParams_test_10_total_error.CameraParameters2 ...
        );
    
    % show
    % figure
    % imshowpair(frameLeft,frameRight,'montage');
    % title('L         Undistorted Images         R');

    
    

    if size(markersPositionLeft,1) == NUM_MARKERS && ...
        size(markersPositionRight,1) == NUM_MARKERS
       
        if NUM_MARKERS == N_MARKERS      %   1 or more markers tracked

            matchedPointsLeft = ...
                SortAscendentBySumOfAxisValues(markersPositionLeft);
            matchedPointsRight = ...
                SortAscendentBySumOfAxisValues(markersPositionRight);

        elseif NUM_MARKERS == 7   %   7 markers tracked only

            matchedPointsLeft = ...
                labelMarkers2DImages_7( ...
                    SortDescendByYAxisValues(markersPositionLeft) ...
                );
            matchedPointsRight = ...
                labelMarkers2DImages_7( ...
                    SortDescendByYAxisValues(markersPositionRight) ...
                );
            
        elseif NUM_MARKERS == 9   % 9 markers tracked only

            matchedPointsLeft = ...
                labelMarkers2DImages_9( ...
                    SortDescendByYAxisValues(markersPositionLeft) ...
                );
            matchedPointsRight = ...
                labelMarkers2DImages_9( ...
                    SortDescendByYAxisValues(markersPositionRight) ...
                );     
        end

        
        % % visualize correspondance points
        % figure
        % showMatchedFeatures(frameLeft, frameRight, matchedPointsLeft, matchedPointsRight);
        % title('Traked features');

       
    
        %% Estimate essential matrix
        % FundamentalMatrix is precalculated and stimated by stereoParams_morning 

        % % Estimate the fundamental matrix
        % [E, epipolarInliers] = estimateEssentialMatrix(...
        %     matchedPointsLeft, matchedPointsRight, stereoParams_morning.CameraParameters1, 'Confidence', 99.99);
        % 
        % % Find epipolar inliers
        % inlierPoints1 = matchedPointsLeft(epipolarInliers, :);
        % inlierPoints2 = matchedPointsRight(epipolarInliers, :);


        %% Compute the camera Pose 	
        % RotationOfCamera2 is done by stereoParams_morning
        % TraslationOfCamera2 is done by stereoParams_morning

        % [orient, loc] = relativeCameraPose(E, stereoParams_morning.CameraParameters1, inlierPoints1, inlierPoints2);
        
        %% Reconstruct the 3D locations of matched Points	
        % compute camera matrices for each position of the cameras	
        
        %     if ~isempty(matchedPointsLeft) && ~isempty(matchedPointsRight) && ...
        %         (size(matchedPointsLeft,1) == size(matchedPointsRight,1))
        %     
                % visualize correspondance
        %         figure
        %         showMatchedFeatures(frameLeft, frameRight, matchedPointsLeft, matchedPointsRight);
        %         title('Traked features');

        % compute 3-D points
        points3D = triangulate(matchedPointsLeft, matchedPointsRight, ...
                               camMatrixLeft, camMatrixRight);

        % Convert to meters and create a pointCloud object
        points3D = points3D ./ 1000;
        
%%      Turn Points3D into an array of dictiories        
       
        [lbwt lfwt ltrc lkne lank lhee lteo] = GetArrayDicPoints3D(points3D);

%%     SavePoints3D - cooking data to be used in JsonEncode
        lbwt_x{idx_raw_data}= lbwt('x'); lbwt_y{idx_raw_data}= lbwt('y'); lbwt_z{idx_raw_data}= lbwt('z');
        lfwt_x{idx_raw_data}= lfwt('x'); lfwt_y{idx_raw_data}= lfwt('y'); lfwt_z{idx_raw_data}= lfwt('z');
        ltrc_x{idx_raw_data}= ltrc('x'); ltrc_y{idx_raw_data}= ltrc('y'); ltrc_z{idx_raw_data}= ltrc('z');
        lkne_x{idx_raw_data}= lkne('x'); lkne_y{idx_raw_data}= lkne('y'); lkne_z{idx_raw_data}= lkne('z');
        lank_x{idx_raw_data}= lank('x'); lank_y{idx_raw_data}= lank('y'); lank_z{idx_raw_data}= lank('z');
        lhee_x{idx_raw_data}= lhee('x'); lhee_y{idx_raw_data}= lhee('y'); lhee_z{idx_raw_data}= lhee('z');
        lteo_x{idx_raw_data}= lteo('x'); lteo_y{idx_raw_data}= lteo('y'); lteo_z{idx_raw_data}= lteo('z');

                
        % %%      Calulate knee angles
        if NUM_MARKERS == 7
            lstResThreePlanes = jsondecode(CalculateAnglesPerImage(points3D));  

            sagittal_hip_ang(idx_raw_data) = lstResThreePlanes.S_ANG_HIP;
            sagittal_pel_ang(idx_raw_data) = lstResThreePlanes.S_ANG_PEL;
            sagittal_kne_ang(idx_raw_data) = lstResThreePlanes.S_ANG_KNE;
            sagittal_ank_ang(idx_raw_data) = lstResThreePlanes.S_ANG_ANK
            frontal_hip_ang(idx_raw_data) = lstResThreePlanes.F_ANG_HIP;
            frontal_pel_ang(idx_raw_data) = lstResThreePlanes.F_ANG_PEL;
            frontal_kne_ang(idx_raw_data) = lstResThreePlanes.F_ANG_KNE;
            frontal_ank_ang(idx_raw_data) = lstResThreePlanes.F_ANG_ANK
            transversal_hip_ang(idx_raw_data) = lstResThreePlanes.T_ANG_HIP;
            transversal_pel_ang(idx_raw_data) = lstResThreePlanes.T_ANG_PEL;
            transversal_kne_ang(idx_raw_data) = lstResThreePlanes.T_ANG_KNE;
            transversal_ank_ang(idx_raw_data) = lstResThreePlanes.T_ANG_ANK

  
            if idx_raw_data > 4

                kinematicsAnalysisGaitAngles = ...
                    jsonencode(table( ...
                                    sagittal_hip_ang, ...
                                    sagittal_pel_ang, ...
                                    sagittal_kne_ang, ...
                                    sagittal_ank_ang, ...
                                    frontal_hip_ang, ...
                                    frontal_pel_ang, ...
                                    frontal_kne_ang, ...
                                    frontal_ank_ang, ...
                                    transversal_hip_ang, ...
                                    transversal_pel_ang, ...
                                    transversal_kne_ang, ...
                                    transversal_ank_ang ...
                                ));


                rawMarkersPositions = ...
                    jsonencode(table(lbwt_x, lbwt_y, lbwt_z, ...
                        lfwt_x, lfwt_y, lfwt_z, ...
                        ltrc_x, ltrc_y, ltrc_z, ...
                        lkne_x, lkne_y, lkne_z, ...
                        lank_x, lank_y, lank_z, ...
                        lhee_x, lhee_y, lhee_z, ...
                        lteo_x, lteo_y, lteo_z));

                kinematicDataToServerAsJson = ...
                    cookKinematicData(kinematicsAnalysisGaitAngles,rawMarkersPositions); 
%               call_put_method_from_onlineMatlab_OK(kinematicDataToServerAsJson);
                testPUTPUT(kinematicDataToServerAsJson);
                kevin='luya';
            end    

        end       
        
        idx_raw_data = idx_raw_data +1; 
        
        %% create the point cloud
        ptCloud = pointCloud(points3D);

        % Visualize the point cloud / Update points
        view(player3D, ptCloud);
        
        if OSCILATION == 1
            % subplot(3,1,1);
            UpdatePlotOscilation(HX, PX, points3D(1,1), startTime,1);
            % subplot(3,1,2);
            UpdatePlotOscilation(HY, PY, points3D(1,2), startTime,2);
            % subplot(3,1,3);
            UpdatePlotOscilation(HZ, PZ, points3D(1,3), startTime,3);

        end
       
    end

    % Display the frame.
    step(player_left, frameLeftBinary);
    step(player_right, frameRightBinary);
   
    % gl = frameLeft(:,:,2) -50;
    % br = frameRight(:,:,3);
    %  step(player_left, gl);
    %  step(player_right, gl + br);

%     step(player_left, frameLeft - frameRight);
%     step(player_right, frameLeftBinary - frameRightBinary);



    iteFrames = iteFrames + 1;
end


%  Clean up.
reset(player_left);
reset(player_right);

%% Oscilation Plot function (x,y,z)

function [H P] = InitPlotOscilation(L_I, L_S,TITLE)

    figure
    h = animatedline;
    px = gca;
    px.YGrid = 'on';
    px.YLim = [L_I L_S];
    title(TITLE);
    xlabel('num frames');
    ylabel('meters');
    
    H = h;
    P = px;
end

function UpdatePlotOscilation(H, P, PTCLOUD, STARTIME,POS_POT)
        
        % subplot(3,1,POS_POT);
        % Get current time
        t =  datetime('now') - STARTIME;
        pos = double(PTCLOUD);
        % Add points to animation
        addpoints(H,datenum(t),pos);
        % Update axes
        P.XLim = datenum([t-seconds(15) t]);
        datetick('x','keeplimits')
        drawnow
        
end

%% Makers traking
function ans = BackgroundSubtraction(backgroundLeft,frameLeft,BINARY_THRES)
    %   Convert RGB 2 HSV Color conversion
    [BackgroundLeft_hsv]=round(rgb2hsv(backgroundLeft));
    [FrameLeft_hsv]=round(rgb2hsv(frameLeft));
    
    Out = bitxor(BackgroundLeft_hsv,FrameLeft_hsv);
    %     subplot(2,2,1), imshow(Out), title('diff - bitxor');
    
    Out = rgb2gray(Out);
    %     subplot(2,2,2), imshow(Out), title('diff Gray scale');
    
    %Read Rows and Columns of the Image
    [rows columns]=size(Out);

    %Convert to Binary Image
    for i=1:rows
        for j=1:columns
            if Out(i,j) > BINARY_THRES
                BinaryImage(i,j)=1;
            else
                BinaryImage(i,j)=0;
            end
        end
    end
    %     subplot(2,2,3), imshow(BinaryImage), title('BinaryImage');
      
    %Apply Median filter to remove Noise
    %     FilteredImage=medfilt2(BinaryImage,[5 5]);
    s = [0 0 0; 0 1 0; 0 0 0];
    FilteredImage = conv2(BinaryImage, s);
    %     subplot(2,2,4), imshow(FilteredImage), title('FilteredImage');

    %    title('L         Original Image         R');
    ans = FilteredImage;
end
   
function img = Gaussianfilter(FILENAME)
    % I = gpuArray(imread(FILENAME));
    I = FILENAME;
    Iblur = imgaussfilt(I, 6);
    
    subplot(1,2,1), imshow(I), title('Original Image');

    subplot(1,2,2), imshow(Iblur)
    title('Gaussian filtered image, \sigma = 1')
    
  
    
    img = Iblur;
end

function matchedPoints = DetectFeaturedPoints_DummyVersion_Sum_X_Y(I1,THRESH,MIN_AREA,MAX_AREA)
  % Segmentation with a threshold by KEVIN
    % Convert to binary images depending on a threshold
    GI1 = rgb2gray(I1);
    % GI2 = rgb2gray(I2);
    thresh = THRESH;
    BW1_TH = GI1 > thresh ;
    % BW2_TH = GI2 > thresh ;

    % detect blobs from BW1_TH
    BW1 = bwconncomp(BW1_TH); 
    stats = regionprops(BW1, 'Area','Eccentricity'); 
    outOfPhase = find([stats.Area] > MIN_AREA & [stats.Area] < MAX_AREA); 
    BW1_BLOB = ismember(labelmatrix(BW1), outOfPhase);
    % detect blobs from BW2_TH
    % BW2 = bwconncomp(BW2_TH); 
    % stats = regionprops(BW2, 'Area','Eccentricity'); 
    % outOfPhase = find([stats.Area] > 100); 
    % BW2_BLOB = ismember(labelmatrix(BW2), outOfPhase);

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
    % imshow(I1);
    % hold on
    % plot(matchedPoints(:,1),matchedPoints(:,2),'b*');
    % hold off
    % 
    % imshow(I2);
    % hold on
    % plot(matchedPoints2(:,1),matchedPoints2(:,2),'b+');
    % hold off


    if length(matchedPoints) > 1
        % asendent sort by the sim of its X and Y values
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
end

function [conv2Img BinaryImage markerPoints] = ExtractPositions(I1,THRESH_MIN,THRESH_MAX,MIN_AREA,MAX_AREA)

    figure
    subplot(2,3,1), imshow(I1), title('color img');
    
    % Convert to binary images depending on a threshold
    GI1 = rgb2gray(I1);
    subplot(2,3,2), imshow(GI1), title('gray img');
    
    %Apply Median filter to remove Noise
    % GI1 = medfilt2(GI1,[3 3]);
    s = [-1.0  -1.0  -1.0; 
         -1.0  8.0  -1.0; 
         -1.0  -1.0  -1.0];
    conv2Img = conv2(GI1, s);
    
    subplot(2,3,3), imshow(conv2Img), title('conv2 img');
    GI1 = conv2Img;
    thresh_min = THRESH_MIN;
    thresh_max = THRESH_MAX;
    BW1_TH = thresh_min < GI1;
    subplot(2,3,4), imshow(BW1_TH), title('thresh_min img');

    %Apply Median filter to remove Noise
    %     MedFilt_BLOB = medfilt2(BW1_TH,[5 5]);
        
    % figure
    % imshowpair(GI1,BW1_TH,'montage');
    % title('L         Undistorted Images         R');
    % detect blobs from BW1_TH
    BW1 = bwconncomp(BW1_TH); 
    %     subplot(2,3,5), imshow(BW1), title('bwconncomp');
    
    stats = regionprops(BW1, 'Area','Eccentricity'); 
    outOfPhase = find([stats.Area] > MIN_AREA & [stats.Area] < MAX_AREA); 
    BW1_BLOB = ismember(labelmatrix(BW1), outOfPhase);
    subplot(2,3,6), imshow(BW1_BLOB), title('ismember img');

    % Calculate centroids
    CC1 = regionprops(BW1_BLOB,'centroid');

    % return binary image and list of marker positions
    BinaryImage = BW1_BLOB;
    markerPoints = single(cat(1,CC1.Centroid));
end

function [conv2Img markerImage markerPoints] = TestKernels(I1,THRESH_MIN,SURGE_MEAN_AREA, TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM)
    outline = [ -1.0  -1.0  -1.0; 
                -1.0   8.0  -1.0; 
                -1.0  -1.0  -1.0  ];

    blur    = [ 0.0625  0.125  0.0625; 
                0.125   0.25   0.125; 
                0.0625  0.125  0.0625  ];

    sharpen = [ 0.0  -1.0   0.0; 
               -1.0   7.0  -1.0; 
                0.0  -1.0   0.0  ];
     

    %     I1 = imgaussfilt(I1, 2);        
    %     figure
    %     subplot(2,3,1), imshow(I1), title('color img');
        
        % Convert to binary images depending on a threshold
    % GI1 = rgb2gray(I1); 
    %     subplot(2,3,2), imshow(GI1), title('gray img');
        
    %     thresh_min = THRESH_MIN;
    %     thresh_max = THRESH_MAX;
    r=I1(:,:,1);
    g=I1(:,:,2);
    b=I1(:,:,3);
    % img = THRESH_MIN < g;
    whiteImage = 255 * ones(720,1280, 'uint8');
    img = whiteImage - (r.*((4.*r)-(1.2.*b)));

    BW1_TH = THRESH_MIN < img ; % green channel
    %     subplot(2,3,3), imshow(BW1_TH), title('thresh_min img');
        
    %     BW1_TH = im2uint8(BW1_TH);
    %     %Applying kernels
    %     
    %     outline_img = conv2(BW1_TH, outline);
    %     subplot(2,3,4), imshow(outline_img), title('outline_img');
    %     
    %     blur_img = conv2(BW1_TH, blur);
    %     subplot(2,3,5), imshow(blur_img), title('blur_img');
    %     
    %     sharpen_img = conv2(BW1_TH, sharpen);
    %     subplot(2,3,6), imshow(sharpen_img), title('sharpen_img');
    %     
    %     GI1 = conv2Img;
    %     Rmin =5;
    %     Rmax = 300;
    %     [centersBright, radiiBright] = imfindcircles(BW1_TH,[Rmin Rmax],'ObjectPolarity','bright');

      %Apply Median filter to remove Noise
    %     MedFilt_BLOB = medfilt2(BW1_TH,[5 5]);

    %     subplot(2,3,4), imshow(MedFilt_BLOB), title('MedFilt_BLOB');
        
    % figure
    % imshowpair(GI1,BW1_TH,'montage');
    % title('L         Undistorted Images         R');

    % detect blobs from BW1_TH
    BW1 = bwconncomp(BW1_TH); 
    %     subplot(2,3,5), imshow(BW1), title('bwconncomp');
    
    stats = regionprops(BW1, 'Area','Eccentricity'); 
    mean_area =  mean([stats.Area]);
    ids = find([stats.Area] > mean_area * SURGE_MEAN_AREA); 
    marker_blobs = ismember(labelmatrix(BW1), ids);
    
   
    %     subplot(2,3,6), imshow(marker_blobs), title('makerblobs img');
    %    
    %     
        % Calculate centroids
    CC1 = regionprops(marker_blobs,'centroid');

    % return marker binary image and  the list of marker positions
    markerImage = marker_blobs;
    markerPoints = single(cat(1,CC1.Centroid));
   
    markerPointsIsideArea = [];
    
    % length = size(markerPoints,1);
    % id = 1;
    % if length > 0
    %     for i=1:length
    %         if markerPoints(i,2) >  TOP_LIM && ...
    %            markerPoints(i,2) <  BOT_LIM && ...
    %            markerPoints(i,1) >  LEF_LIM && ...
    %            markerPoints(i,1) <  RIG_LIM 
    %             markerPointsIsideArea(id,:) = markerPoints(i,:);
    %             id = id + 1;
    %         end
    %     end
    % end
    % markerPoints = markerPointsIsideArea;
    conv2Img=marker_blobs;
end

function trackedPoints = ExtractPositionFromFilteredImage(IMAGE,MIN_AREA,MAX_AREA)

    % figure
    % imshowpair(GI1,BW1_TH,'montage');
    % title('L         Undistorted Images         R');
    % detect blobs from BW1_TH
    BW1 = bwconncomp(IMAGE); 
    stats = regionprops(BW1, 'Area','Eccentricity'); 
    outOfPhase = find([stats.Area] > MIN_AREA & [stats.Area] < MAX_AREA); 
    BW1_BLOB = ismember(labelmatrix(BW1), outOfPhase);
    
    % Calculate centroids
    CC1 = regionprops(BW1_BLOB,'centroid');

    % return list of markers
    trackedPoints = single(cat(1,CC1.Centroid));
end

%% Marker sorts
function sortList = SortAscendentBySumOfAxisValues(MARKER_POINTS)
    if length(MARKER_POINTS) > 1
        % asendent sort by the sim of its X and Y values
        [rows, columns] = size(MARKER_POINTS);
        tempMat = [];
        for i = 1:rows
            newRow = [(MARKER_POINTS(i,1) + MARKER_POINTS(i,2)) MARKER_POINTS(i,1) MARKER_POINTS(i,2)];
            tempMat = cat(1, tempMat, newRow); 
        end

        out = sortrows(tempMat);
        out(:,1) =[];
        sortList = out;    
    end
end

function sortList = SortDescendByYAxisValues(MARKER_POINTS)
    if length(MARKER_POINTS) > 2
        % asendent by Y-axis  values
        sortList = sortrows(MARKER_POINTS,-2); 
        
    end
end

%% Labe markers
function labeledMarkers = labelMarkers2DImages_7(SORTED_LIST_Y)

    % Work with the last three markers of the list which are basically the
    % ankle, feet, and malleole extern (not necessarily sorted)   
    % calculate distance, in space, between the these markers
    a  = sqrt( ((SORTLIST(2,1) - SORTLIST(1,1)) .^ 2) + ((SORTLIST(2,2) - SORTLIST(1,2)) .^ 2) );
    b  = sqrt( ((SORTLIST(2,1) - SORTLIST(3,1)) .^ 2) + ((SORTLIST(2,2) - SORTLIST(3,2)) .^ 2) );
    c  = sqrt( ((SORTLIST(1,1) - SORTLIST(3,1)) .^ 2) + ((SORTLIST(1,2) - SORTLIST(3,2)) .^ 2) );
    % having the three distances, The algorithm must iterate each marker
    % and sum the two distance calculated from the marker itself to the 
    % other two markers left.
    
    matSum =[]; 
    
    for i = 1:3
        sumDist = 0;
        for j=1:3
            if i ~= j
            dist = sqrt( ((SORTED_LIST_Y(i,1) - SORTED_LIST_Y(j,1)) .^ 2) + ((SORTED_LIST_Y(i,2) - SORTED_LIST_Y(j,2)) .^ 2) );
            sumDist = sumDist + dist; 
            end
        end
        newRow = [i, sumDist];

        matSum = cat(1, matSum, newRow);
    end
    
    % descendet sort by its sum of dist
    matSumOrdered = sortrows(matSum,-2);

    % Copy sort list to a temporal list 
     temList = SORTED_LIST_Y;
     
    % Finally, put them back to the original list 
    % then, the HIGHEST SUM is the FEET MARKER
    SORTED_LIST_Y(1,:) = temList( matSumOrdered(1,1), :);
    
    % the last but no the least, 
    % the INTERMEDIUM sum is the CALCANIUM MARKER
    SORTED_LIST_Y(2,:) = temList( matSumOrdered(2,1), :);
    
    % likewise, the LOWEST sum is the MALLEOLE MARKER
    SORTED_LIST_Y(3,:) = temList( matSumOrdered(3,1), :);
    
  
    
    
    
    % To label the KNEE, the sorted list did that job for us as it is
    % ordered in terms of y-axis, likewise the TROCHANTER (placed down 
    % for visibility purposes). The next in the list, starting from the
    % end is the KNEE at 4th position, the TROCHANTER at 5th position.
    
    % To calcula the ANTERIOR ILIAC SPINE CREST and POSTERIOR SUPERIOR 
    % ILLIAC CREST, let's comapare it X values  
    if SORTED_LIST_Y(6,1) > SORTED_LIST_Y(7,1)
        SORTED_LIST_Y(7,1) = temList(6,1);
        SORTED_LIST_Y(6,1) = temList(7,1);
    end
    
    % and at 7nd postion will be located the POSTERIOR SUPERIOR ILLIAC
    % CREST
    labeledMarkers = SORTED_LIST_Y;
end

function labeledMarkers = labelMarkers2DImages_9(SORTED_LIST_Y)

    % Work with the last three markers of the list which are basically the
    % ankle, feet, and malleole extern (not necessarily sorted)   
    % calculate distance, in space, between the these markers

    % a  = sqrt( ((SORTLIST(2,1) - SORTLIST(1,1)) .^ 2) + ((SORTLIST(2,2) - SORTLIST(1,2)) .^ 2) );
    % b  = sqrt( ((SORTLIST(2,1) - SORTLIST(3,1)) .^ 2) + ((SORTLIST(2,2) - SORTLIST(3,2)) .^ 2) );
    % c  = sqrt( ((SORTLIST(1,1) - SORTLIST(3,1)) .^ 2) + ((SORTLIST(1,2) - SORTLIST(3,2)) .^ 2) );
     
    % having the three distances, The algorithm must iterate each marker
    % and sum the two distance calculated from the marker itself to the 
    % other two markers left.
    
    matSum =[]; 
    
    for i = 1:3
        sumDist = 0;
        for j=1:3
            if i ~= j
            dist = sqrt( ((SORTED_LIST_Y(i,1) - SORTED_LIST_Y(j,1)) .^ 2) + ((SORTED_LIST_Y(i,2) - SORTED_LIST_Y(j,2)) .^ 2) );
            sumDist = sumDist + dist; 
            end
        end
        newRow = [i, sumDist];

        matSum = cat(1, matSum, newRow);
    end
    
    % descendet sort by its sum of dist
    matSumOrdered = sortrows(matSum,-2);

    % Copy sort list to a temporal list 
     temList = SORTED_LIST_Y;
     
    % Finally, put them back to the original list 
    % then, the HIGHEST SUM is the FEET MARKER
    SORTED_LIST_Y(1,:) = temList( matSumOrdered(1,1), :);
    
    % the last but no the least, 
    % the INTERMEDIUM sum is the CALCANIUM MARKER
    SORTED_LIST_Y(2,:) = temList( matSumOrdered(2,1), :);
    
    % likewise, the LOWEST sum is the MALLEOLE MARKER
    SORTED_LIST_Y(3,:) = temList( matSumOrdered(3,1), :);
    

    % TODO : label extra markers in this position after foot and before
    % hip markers. Consider the knee postion in order to add Thigt and 
    % shank markers
    
    % To label the SHANK, KNEE, THIGHT, the sorted list did that job 
    % for us as it is
    % ordered in terms of y-axis, likewise the TROCHANTER (placed down 
    % for visibility purposes). The next in the list, starting from the
    % end is the KNEE at 4th position, the TROCHANTER at 5th position.
    
    % To calcula the ANTERIOR ILIAC SPINE CREST and POSTERIOR SUPERIOR 
    % ILLIAC CREST, let's comapare it X values  
    if SORTED_LIST_Y(8,1) > SORTED_LIST_Y(9,1)
        SORTED_LIST_Y(9,1) = temList(8,1);
        SORTED_LIST_Y(8,1) = temList(9,1);
    end
    
    % and at 7nd postion will be located the POSTERIOR SUPERIOR ILLIAC
    % CREST
    labeledMarkers = SORTED_LIST_Y;
end

%% Save 3D points to JSON FORMAT 
function SavePoints3DToJsonFormatSample (POINT3D , idx_raw_data,lkne_x)
  
    % x = [40;43;
    % y = [43;69];
    % z = [10;41];

    % jsonencode(table(x,y,z))
    %  =   '[{"x":"40","y":43, "z":10},{"x":"43","y":69, "z":41}]'

end

%% Methods that turn 3DPoints into dictionaries
function newMapDicPoint3D = GetDicPoint3D(POINT3D)

    % cm = CMarker3D;
    [m n] = size(POINT3D);
    keySet   = {'x','y','z'};

    for i = 1:n
        valueSet(i) = POINT3D(1,i)
    end
    newMapDicPoint3D = containers.Map(keySet,valueSet);

end

function [lbwt lfwt ltrc lkne lank lhee lteo] = GetArrayDicPoints3D(POINTS3D)
    
    lteo = GetDicPoint3D(POINTS3D(1,:));   
    lhee = GetDicPoint3D(POINTS3D(2,:));
    lank = GetDicPoint3D(POINTS3D(3,:));
    lkne = GetDicPoint3D(POINTS3D(4,:));
    ltrc = GetDicPoint3D(POINTS3D(5,:));
    lfwt = GetDicPoint3D(POINTS3D(6,:));
    lbwt = GetDicPoint3D(POINTS3D(7,:)); % lbwt( 1 , 1=x | 2=y | 3=z);

end

%% Methods that calculate angles
% function [ANG_HIP ANG_PEL ANG_KNE ANG_ANK] = CalculateAnglesPerImage(POINTS3D)

%    [lbwt lfwt ltrc lkne lank lhee lteo] = GetArrayDicPoints3D(POINTS3D);
   
%    ANG_HIP = CalculateHipAnglesSagittal(lbwt, lfwt, ltrc, lkne);
%    ANG_PEL = CalculatePelvisAnglesSagittal(lbwt, lfwt);
%    ANG_KNE = CalculateKneeAnglesSagittal(ltrc, lkne, lank);
%    ANG_ANK = CalculateAnkleAnglesSagittal(lkne, lank, lhee, lteo);
%    trop =32 ;
%    % ... 

% end
function listResAngles = CalculateAnglesPerImage(POINTS3D)

   [lbwt lfwt ltrc lkne lank lhee lteo] = GetArrayDicPoints3D(POINTS3D);
   
   S_ANG_HIP = CalculateHipAnglesSagittal(lbwt, lfwt, ltrc, lkne);
   S_ANG_PEL = CalculatePelvisAnglesSagittal(lbwt, lfwt);
   S_ANG_KNE = CalculateKneeAnglesSagittal(ltrc, lkne, lank);
   S_ANG_ANK = CalculateAnkleAnglesSagittal(lkne, lank, lhee, lteo);

   F_ANG_HIP = CalculateHipAnglesSagittal(lbwt, lfwt, ltrc, lkne);
   F_ANG_PEL = CalculatePelvisAnglesSagittal(lbwt, lfwt);
   F_ANG_KNE = CalculateKneeAnglesSagittal(ltrc, lkne, lank);
   F_ANG_ANK = CalculateAnkleAnglesSagittal(lkne, lank, lhee, lteo);

   T_ANG_HIP = CalculateHipAnglesSagittal(lbwt, lfwt, ltrc, lkne);
   T_ANG_PEL = CalculatePelvisAnglesSagittal(lbwt, lfwt);
   T_ANG_KNE = CalculateKneeAnglesSagittal(ltrc, lkne, lank);
   T_ANG_ANK = CalculateAnkleAnglesSagittal(lkne, lank, lhee, lteo);


   listResAngles  = ...
       jsonencode( ...
           containers.Map( ...
               {
                    'S_ANG_HIP', ...
                    'S_ANG_PEL', ...
                    'S_ANG_KNE', ...
                    'S_ANG_ANK', ...
                    'F_ANG_HIP', ...
                    'F_ANG_PEL', ...
                    'F_ANG_KNE', ...
                    'F_ANG_ANK', ...
                    'T_ANG_HIP', ...
                    'T_ANG_PEL', ...
                    'T_ANG_KNE', ...
                    'T_ANG_ANK', ...
               }, ...
               { ...
                   S_ANG_HIP, ...
                   S_ANG_PEL, ...
                   S_ANG_KNE, ...
                   S_ANG_ANK, ...
                   F_ANG_HIP, ...
                   F_ANG_PEL, ...
                   F_ANG_KNE, ...
                   F_ANG_ANK, ...
                   T_ANG_HIP, ...
                   T_ANG_PEL, ...
                   T_ANG_KNE, ...
                   T_ANG_ANK, ...
               } ...
           ) ...
       );
   trop =32 ;

end

function ang = HCCalculateHipAnglesSagittal()
    v_top = [10, 0] - [-10, 0];
    v_bott  = [0, -1] - [10, -1];
    
    v_topI = [-10, 0] - [10, 0];
    v_bottI  = [10, -1] - [0, -10];
    
    ang = CalculateAngles2Vectors(v_top, v_bott, 'Absolute');
    angI = CalculateAngles2Vectors(v_topI, v_bottI, 'Absolute')
    % check this FUNCTINNNNNNNNNNNNNNNNNNNN
    ang = 90 - ang; % sin perpendicular to Eilic spain
end

%% Sagital Angles calculation

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
    
    iliacSpainAxis = [LFWT('x'), LFWT('y')] - [LBWT('x'), LBWT('y')];
    thighAxis  = [LKNE('x'), LKNE('y')] - [LTRC('x'), LTRC('y')];  
    
    ang = CalculateAngles2Vectors(iliacSpainAxis, thighAxis, 'relative');
    
    % then the angle must be substracted 90 to get the rigth angle which is
    % formed beteen the thighAxis and the perpendicular axis to iliacSpainAxis 
    ang = 90 - ang;
end

function ang = CalculatePelvisAnglesSagittal(LBWT, LFWT)
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

    P_TOP  = [LBWT('x'), LBWT('y')];  
    P_CENT = [LFWT('x'), LFWT('y')];
    P_BOTT = [  -9999  , LFWT('y')];
    
    ang = CalculateAngle3Points(P_TOP, P_CENT, P_BOTT, 'relative');
    
    % iliacSpainAxis = [LFWT('x'), LFWT('y')] - [LBWT('x'), LBWT('y')];
    % hotizontalAxis  = [LFWT('x'), LFWT('y')] - [-999, LFWT('y')]; 
    % angVec = CalculateAngles2Vectors(iliacSpainAxis, hotizontalAxis, 'relative');
    
    % ang = angVec;
end

function ang = CalculateKneeAnglesSagittal(LTRC, LKNE, LANK)
    % Knee flexion/extension
    % Relative

    % The sagittal shank axis is projected into the plane perpendicular to 
    % the knee flexion axis. Knee flexion is the angle in that plane between 
    % this projection and the sagittal thigh axis. The sign is such that a 
    % positive angle corresponds to a flexed knee.
    
    P_TOP  = [LTRC('x'), LTRC('y')];  
    P_CENT = [LKNE('x'), LKNE('y')];
    P_BOTT = [LANK('x'), LANK('y')];
    
    ang = CalculateAngle3Points(P_TOP, P_CENT, P_BOTT, 'relative');
    
    % thighAxis  = [LKNE('x'), LKNE('y')] - [LTRC('x'), LTRC('y')];
    % shankAxis  = [LANK('x'), LANK('y')] - [LKNE('x'), LKNE('y')]; 
    % angVec = CalculateAngles2Vectors(thighAxis, shankAxis, 'relative');
    
    % ang = angVec;
end

function ang = CalculateAnkleAnglesSagittal(LKNE, LANK, LHEE, LTOE)
    % Ankle dorsi/plantar flexion
    % Relative
    
    % The foot vector is projected into the foot sagittal plane. The angle 
    % between the foot vector and the sagittal axis of the shank is the foot
    % dorsi/plantar flexion. A positive number corresponds to dorsiflexion.

    % the fact that is left kinematic analysis means that both vectors 
    % should point from left to rigth (->) to get the internal angle

    % perfTriX = LANK('x') - (LKNE('y') - LANK('y'));
    % perfTriy = LANK('y') - (LANK('x') - LKNE('x'));
    
    % perpenAxisToShankAxis = [LANK('x'), LANK('y')] - [LFWT('x'), LFWT('y')];

    shankAxis  = [LANK('x'), LANK('y')] - [LKNE('x'), LKNE('y')]; 
    retropieAxis  = [LTOE('x'), LTOE('y')] - [LHEE('x'), LHEE('y')]; 
    
    ang = CalculateAngles2Vectors(shankAxis, retropieAxis, 'Absolute');
    % check this FUNCTINNNNNNNNNNNNNNNNNNNN
    ang = 90 - ang; % sin perpendicular to Eilic spain
end

%% Frontal Angles Calculation
% theorically done - to be tested after resolving the multiple cameras
% visibilities toward the markers
function ang = CalculateHipAnglesFrontal(RFWT, LFWT,LTRC, LKNE)
    % Hip ab/adduction
    % Relative
    
    % Hip adduction is measured in the plane of the hip flexion axis and the 
    % knee joint centre. The angle is calculated between the long axis of the 
    % thigh and the frontal axis of the pelvis projected into this plane. 
    % A positive number corresponds to an adducted (inwardly moved) leg.   

    % the fact that is left kinematic analysis means that both vectors 
    % should point from left to right (->) to get the internal angle
    
    iliacSpainFrontalAxis = [LFWT('x'), LFWT('y')] - [RFWT('x'), RFWT('y')];
    thighFrontalAxis  = [LTRC('x'), LTRC('y')] - [LKNE('x'), LKNE('y')];  
    
    ang = CalculateAngles2Vectors(iliacSpainFrontalAxis, thighFrontalAxis, 'relative');
    
    % then the angle must be substracted 90 to get the rigth angle which is
    % formed beteen the thighAxis and the perpendicular axis to iliacSpainAxis 
    ang = 90 - ang;
end

% theorically done - to be tested after resolving the multiple cameras
% visibilities toward the markers
function ang = CalculatePelvisAnglesFrontal(RFWT, LFWT)
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
    
    iliacSpainFrontalAxis = [LFWT('z'), LFWT('y')] - [RFWT('z'), RFWT('y')];
    horizontalAxis  = [LFWT('z'), LFWT('y')] - [9999, LFWT('y')];  
    
    ang = CalculateAngles2Vectors(iliacSpainFrontalAxis, horizontalAxis, 'relative');
end

function ang = CalculateKneeAnglesFrontal(LTRC, LKNE, LANK)
    % Foot progression
    % Absolute
    % This is measured in the plane of the knee flexion axis and the ankle 
    % center, and is the angle between the long axis of the shank and the long 
    % axis of the thigh projected into this plane.
    % A positive number corresponds to varus (outward bend of the knee).

    P_TOP  = [LTRC('z'), LTRC('y')];  
    P_CENT = [LKNE('z'), LKNE('y')];
    P_BOTT = [LANK('z'), LANK('y')];
    
    ang = CalculateAngle3Points(P_TOP, P_CENT, P_BOTT, 'relative');
end

function ang = CalculateAnkleAnglesFrontal(LHEE, LTOE)
    % Assessing the foot progression angle (FPA) during gait is an important 
    % part of a clinician's examination. The FPA is defined as the angle made
    % by the long axis of the foot from the heel to 2nd metatarsal and the line 
    % of progression of gait. A negative FPA indicates in-toeing and a positive
    % FPA out-toeing.
    
    verticalAxis  = [LHEE('z'), LHEE('x')] - [LKNE('z'), -9999]; 
    retropieAxis  = [LHEE('z'), LHEE('x')] - [LTOE('z'), LTOE('x')]; 
    
    ang = CalculateAngles2Vectors(shankAxis, retropieAxis, 'Absolute');
end

%% Transversal Angles 
function ang = CalculateAnkleAnglesRotation(LHEE, LTOE)
    % Assessing the foot progression angle (FPA) during gait is an important 
    % part of a clinician's examination. The FPA is defined as the angle made
    % by the long axis of the foot from the heel to 2nd metatarsal and the line 
    % of progression of gait. A negative FPA indicates in-toeing and a positive
    % FPA out-toeing.
    
    verticalAxis  = [LHEE('z'), LHEE('x')] - [LKNE('z'), -9999]; 
    retropieAxis  = [LHEE('z'), LHEE('x')] - [LTOE('z'), LTOE('x')]; 
    
    ang = CalculateAngles2Vectors(shankAxis, retropieAxis, 'Absolute');
end


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
        
    % elseif TYPE == 'absolute' 
        angle = 180 - angle ;  % Stable
    end
end

function ang = CalculateAngles2Vectors(V_TOP, V_BOTT, TYPE)
    
    ang = (acos(sum(V_TOP.*V_BOTT)/(norm(V_TOP)*norm(V_BOTT))))  * 180/pi;
    
    if strcmp (TYPE,'relative') 

        ang = 180 - ang ; 
    end  
end
% Save as jsonstring  '   ' with          
% mlobj= jsondecode(hcd_to_string_raw)  =>  jsondecode(to acces the object in matlab)
% toPlaceDataAS MATRIX(:,2)= mlobj.series{3,1}.data  =>  
% create GaitCycleStruct = zeros(0-100 %,angles)
% pupulate the array
% set mlobj.series{2 1}.data = GaitCycleStruct <= right
% set mlobj.series{3, 1}.data = GaitCycleStruct <= left
% sen via web service from Matlab to server

% P0 = [x0,y0], P1 = [x1,y1], and P2 = [x2,y2

function [ ...
    isSync ...
    outOfPhase_l  ...
    outOfPhase_r  ...
    isSyncBlob_left_detectedd  ...
    isSyncBlob_right_detectedd  ...
    img_blob_l  ...
    img_blob_r ...
] = ...
    SyncBothFrames( ...
        FRAME_LEFT, ...
        FRAME_RIGHT, ...
        OUT_OF_PHASE_L, ...
        OUT_OF_PHASE_R,  ...
        IS_SYNC_BLOB_LEFT_DETECTED, ...
        IS_SYNC_BLOB_RIGHT_DETECTED, ...
        SYNC_UMBRAL, ...
        TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM ...
    )
    % 1 => L R 0 0 F F
    % 2 =>         F F
    % 3 =>         V F
    % 4 =>     11 0 V V
    % 5 =>             => isSync =  true
    isSync = false;

    if(IS_SYNC_BLOB_LEFT_DETECTED && IS_SYNC_BLOB_RIGHT_DETECTED)
        isSync= true;
    else

        if ~IS_SYNC_BLOB_LEFT_DETECTED
            [ isSyncBlob_left_detected img_blob_l ] = DetectSyncBlob(FRAME_LEFT, SYNC_UMBRAL, 20, TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM);
            IS_SYNC_BLOB_LEFT_DETECTED =  isSyncBlob_left_detected;
        else
           OUT_OF_PHASE_L = OUT_OF_PHASE_L + 1;
        end 

        if ~IS_SYNC_BLOB_RIGHT_DETECTED
            [ isSyncBlob_right_detected img_blob_r ] = DetectSyncBlob(FRAME_RIGHT, SYNC_UMBRAL, 20, TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM);
            IS_SYNC_BLOB_RIGHT_DETECTED = isSyncBlob_right_detected;
        else
            OUT_OF_PHASE_R = OUT_OF_PHASE_R + 1;
        end
    end

    % results
    outOfPhase_l = OUT_OF_PHASE_L;
    outOfPhase_r = OUT_OF_PHASE_R;
    isSyncBlob_left_detectedd = IS_SYNC_BLOB_LEFT_DETECTED;
    isSyncBlob_right_detectedd = IS_SYNC_BLOB_RIGHT_DETECTED;
    
end

function [ isSyncBlob  markerImage ] = DetectSyncBlob(I1, THRESH_MIN, MIN_AREA, TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM)
    
    isSyncBlob = false;

    GI1 = rgb2gray(I1);
    BW1_TH = THRESH_MIN < GI1;
    BW1 = bwconncomp(BW1_TH); 
    
    stats = regionprops(BW1, 'Area','Eccentricity'); 
    mean_area =  mean([stats.Area]); 
    ids = find([stats.Area] <= MIN_AREA); 
    marker_blobs = ismember(labelmatrix(BW1), ids); 
    
    % Calculate centroids
    CC1 = regionprops(marker_blobs,'centroid');
    markerPoints = single(cat(1,CC1.Centroid));
    
    length = size(markerPoints,1);
    cont = 0;
    for i=1:length
       if markerPoints(i,2) >  TOP_LIM && ...
          markerPoints(i,2) <  BOT_LIM && ...
          markerPoints(i,1) >  LEF_LIM && ...
          markerPoints(i,1) <  RIG_LIM 
           cont = cont + 1;
       end
    end
    
    
    % return marker binary image and isSyncBlob
    if(cont > 0)
        isSyncBlob = true;
    end
    markerImage = marker_blobs;
    
end

function [ Result ] = Factorial2( Value )
    %Factorial2 - Calculates the value of n!
    % Outputs the factorial value of the input number.
    if Value > 1
        Result = Factorial2(Value - 1) * Value;
    else
        Result = 1;
    end
end

function img = filterRGBChannels(IMG)
    r=IMG(:,:,1);
    g=IMG(:,:,2);
    b=IMG(:,:,3);

    whiteImage = 255 * ones(720,1280, 'uint8');

    % img = whiteImage - (r.*(r-b));
    img  = 100 < g


    GI1 = rgb2gray(IMG);
    BW1_TH = 160 < GI1

    k1=imfuse(BW1_TH,50 < g,'montage'); %composite of 2images
    imshow(k1); 

    INT =1;
end

function res = postTest_weird_error (DATA)

    method = matlab.net.http.RequestMethod.POST;

    contentType = matlab.net.http.HeaderField('ContentType','application/json');
    token       = matlab.net.http.HeaderField('x-access-token','eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyIkX18iOnsic3RyaWN0TW9kZSI6dHJ1ZSwiZ2V0dGVycyI6e30sIndhc1BvcHVsYXRlZCI6ZmFsc2UsImFjdGl2ZVBhdGhzIjp7InBhdGhzIjp7Im1lZGljYWxDZW50ZXJzLnJlcXVlc3RlZF9hdCI6ImRlZmF1bHQiLCJfX3YiOiJpbml0IiwiYWRkcmVzcy5jb3VudHJ5IjoiaW5pdCIsImFkZHJlc3MuemlwIjoiaW5pdCIsImFkZHJlc3Muc3RhdGUiOiJpbml0IiwiYWRkcmVzcy5jaXR5IjoiaW5pdCIsImFkZHJlc3Muc3RyZWV0IjoiaW5pdCIsIm1lZGljYWxDZW50ZXJzLmFjY2VwdGVkX2F0IjoiaW5pdCIsIm1lZGljYWxDZW50ZXJzLnN0YXR1c19yZXF1ZXN0IjoiaW5pdCIsIm1lZGljYWxDZW50ZXJzLm5hbWUiOiJpbml0IiwibWVkaWNhbENlbnRlcnMuX2lkIjoiaW5pdCIsIm5hbWVzIjoiaW5pdCIsImdlbmRlciI6ImluaXQiLCJpZF9Eb2N1bWVudF90eXBlIjoiaW5pdCIsImlkX0RvY3VtZW50X251bSI6ImluaXQiLCJiaXJ0aCI6ImluaXQiLCJlbWFpbCI6ImluaXQiLCJwaG9uZSI6ImluaXQiLCJjZWxscGhvbmUiOiJpbml0IiwibnVtX2N0bXAiOiJpbml0IiwibnVtX25kdGEiOiJpbml0IiwiaXNfYWN0aXZlIjoiaW5pdCIsInVzZXJuYW1lIjoiaW5pdCIsInBhc3N3b3JkIjoiaW5pdCIsIl9pZCI6ImluaXQifSwic3RhdGVzIjp7Imlnbm9yZSI6e30sImRlZmF1bHQiOnsibWVkaWNhbENlbnRlcnMucmVxdWVzdGVkX2F0Ijp0cnVlfSwiaW5pdCI6eyJfX3YiOnRydWUsImFkZHJlc3MuY291bnRyeSI6dHJ1ZSwiYWRkcmVzcy56aXAiOnRydWUsImFkZHJlc3Muc3RhdGUiOnRydWUsImFkZHJlc3MuY2l0eSI6dHJ1ZSwiYWRkcmVzcy5zdHJlZXQiOnRydWUsIm1lZGljYWxDZW50ZXJzLmFjY2VwdGVkX2F0Ijp0cnVlLCJtZWRpY2FsQ2VudGVycy5zdGF0dXNfcmVxdWVzdCI6dHJ1ZSwibWVkaWNhbENlbnRlcnMubmFtZSI6dHJ1ZSwibWVkaWNhbENlbnRlcnMuX2lkIjp0cnVlLCJuYW1lcyI6dHJ1ZSwiZ2VuZGVyIjp0cnVlLCJpZF9Eb2N1bWVudF90eXBlIjp0cnVlLCJpZF9Eb2N1bWVudF9udW0iOnRydWUsImJpcnRoIjp0cnVlLCJlbWFpbCI6dHJ1ZSwicGhvbmUiOnRydWUsImNlbGxwaG9uZSI6dHJ1ZSwibnVtX2N0bXAiOnRydWUsIm51bV9uZHRhIjp0cnVlLCJpc19hY3RpdmUiOnRydWUsInVzZXJuYW1lIjp0cnVlLCJwYXNzd29yZCI6dHJ1ZSwiX2lkIjp0cnVlfSwibW9kaWZ5Ijp7fSwicmVxdWlyZSI6e319LCJzdGF0ZU5hbWVzIjpbInJlcXVpcmUiLCJtb2RpZnkiLCJpbml0IiwiZGVmYXVsdCIsImlnbm9yZSJdfSwiZW1pdHRlciI6eyJkb21haW4iOm51bGwsIl9ldmVudHMiOnt9LCJfZXZlbnRzQ291bnQiOjAsIl9tYXhMaXN0ZW5lcnMiOjB9fSwiaXNOZXciOmZhbHNlLCJfZG9jIjp7ImFkZHJlc3MiOnsiY291bnRyeSI6IlBlcnUiLCJ6aXAiOjQ1NzY0NSwic3RhdGUiOiJMaW1hIiwiY2l0eSI6IkxpbWEiLCJzdHJlZXQiOiJDYWxsZSBBbGFtZWRhIFNhbnRvcyAzNDQgRHB0byAzMDQifSwibWVkaWNhbENlbnRlcnMiOnsicmVxdWVzdGVkX2F0IjoiMjAxNi0xMi0xNVQwMTo1Mzo0Mi4xMzJaIiwiYWNjZXB0ZWRfYXQiOiIyMDE2LTExLTIwVDA0OjE5OjEzLjAwMFoiLCJzdGF0dXNfcmVxdWVzdCI6IjEiLCJuYW1lIjoiTHVpcyBNYW51ZWwiLCJfaWQiOiIzNDU2Nzg5MDQ1NjU4NDhmcjVnciJ9LCJfX3YiOjAsIm5hbWVzIjoiSG9ydGVuY2lhIiwiZ2VuZGVyIjoiNCIsImlkX0RvY3VtZW50X3R5cGUiOiJETkkiLCJpZF9Eb2N1bWVudF9udW0iOjEyMzQ1Njc4LCJiaXJ0aCI6IjIwMTYtMTEtMjBUMDQ6MTk6MTMuMDAwWiIsImVtYWlsIjoibWFudWVsQGdtYWlsLmNvbSIsInBob25lIjoiMjM0IDU0IDEzIiwiY2VsbHBob25lIjoiOTk5IDk5OSA5OTkiLCJudW1fY3RtcCI6NjU0MiwibnVtX25kdGEiOjU0NTM0NTQzLCJpc19hY3RpdmUiOmZhbHNlLCJ1c2VybmFtZSI6InRlcmFwZXV0YSIsInBhc3N3b3JkIjoiYWRtaW4iLCJfaWQiOiI1ODUxZjYwMTczZGMxMTA3MmEwYTFhOTIifSwiX3ByZXMiOnsiJF9fb3JpZ2luYWxfc2F2ZSI6W251bGwsbnVsbF0sIiRfX29yaWdpbmFsX3ZhbGlkYXRlIjpbbnVsbF0sIiRfX29yaWdpbmFsX3JlbW92ZSI6W251bGxdfSwiX3Bvc3RzIjp7IiRfX29yaWdpbmFsX3NhdmUiOltdLCIkX19vcmlnaW5hbF92YWxpZGF0ZSI6W10sIiRfX29yaWdpbmFsX3JlbW92ZSI6W119LCJpYXQiOjE0ODE3NjY4MjJ9.xDNN-rILCYc5vqVJzpLn3DIqOqMMPTEBuYHgvISoHPw');
    header = [contentType token];

    input = struct('kinematics_analysis_id','58327f939d4fe93d29435260');
    paramsInput = struct('params', input);

    paramsInput = jsonencode(DATA);
 
    body = {json, paramsInput};
    body = matlab.net.http.MessageBody(body);
 
    disp(body)

    request = matlab.net.http.RequestMessage(method,header,body);

    uri = matlab.net.URI('http://52.89.123.49:8080/api/kinematics_analysis_matlab/58327f939d4fe93d29435260');
    response = send(request,uri);
    show(response)
    disp(response)

end

function call_get_method_test_OK()

    method = matlab.net.http.RequestMethod.GET;

    contentType = matlab.net.http.HeaderField('ContentType','application/json');
    token       = matlab.net.http.HeaderField('x-access-token','eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyIkX18iOnsic3RyaWN0TW9kZSI6dHJ1ZSwiZ2V0dGVycyI6e30sIndhc1BvcHVsYXRlZCI6ZmFsc2UsImFjdGl2ZVBhdGhzIjp7InBhdGhzIjp7Im1lZGljYWxDZW50ZXJzLnJlcXVlc3RlZF9hdCI6ImRlZmF1bHQiLCJfX3YiOiJpbml0IiwiYWRkcmVzcy5jb3VudHJ5IjoiaW5pdCIsImFkZHJlc3MuemlwIjoiaW5pdCIsImFkZHJlc3Muc3RhdGUiOiJpbml0IiwiYWRkcmVzcy5jaXR5IjoiaW5pdCIsImFkZHJlc3Muc3RyZWV0IjoiaW5pdCIsIm1lZGljYWxDZW50ZXJzLmFjY2VwdGVkX2F0IjoiaW5pdCIsIm1lZGljYWxDZW50ZXJzLnN0YXR1c19yZXF1ZXN0IjoiaW5pdCIsIm1lZGljYWxDZW50ZXJzLm5hbWUiOiJpbml0IiwibWVkaWNhbENlbnRlcnMuX2lkIjoiaW5pdCIsIm5hbWVzIjoiaW5pdCIsImdlbmRlciI6ImluaXQiLCJpZF9Eb2N1bWVudF90eXBlIjoiaW5pdCIsImlkX0RvY3VtZW50X251bSI6ImluaXQiLCJiaXJ0aCI6ImluaXQiLCJlbWFpbCI6ImluaXQiLCJwaG9uZSI6ImluaXQiLCJjZWxscGhvbmUiOiJpbml0IiwibnVtX2N0bXAiOiJpbml0IiwibnVtX25kdGEiOiJpbml0IiwiaXNfYWN0aXZlIjoiaW5pdCIsInVzZXJuYW1lIjoiaW5pdCIsInBhc3N3b3JkIjoiaW5pdCIsIl9pZCI6ImluaXQifSwic3RhdGVzIjp7Imlnbm9yZSI6e30sImRlZmF1bHQiOnsibWVkaWNhbENlbnRlcnMucmVxdWVzdGVkX2F0Ijp0cnVlfSwiaW5pdCI6eyJfX3YiOnRydWUsImFkZHJlc3MuY291bnRyeSI6dHJ1ZSwiYWRkcmVzcy56aXAiOnRydWUsImFkZHJlc3Muc3RhdGUiOnRydWUsImFkZHJlc3MuY2l0eSI6dHJ1ZSwiYWRkcmVzcy5zdHJlZXQiOnRydWUsIm1lZGljYWxDZW50ZXJzLmFjY2VwdGVkX2F0Ijp0cnVlLCJtZWRpY2FsQ2VudGVycy5zdGF0dXNfcmVxdWVzdCI6dHJ1ZSwibWVkaWNhbENlbnRlcnMubmFtZSI6dHJ1ZSwibWVkaWNhbENlbnRlcnMuX2lkIjp0cnVlLCJuYW1lcyI6dHJ1ZSwiZ2VuZGVyIjp0cnVlLCJpZF9Eb2N1bWVudF90eXBlIjp0cnVlLCJpZF9Eb2N1bWVudF9udW0iOnRydWUsImJpcnRoIjp0cnVlLCJlbWFpbCI6dHJ1ZSwicGhvbmUiOnRydWUsImNlbGxwaG9uZSI6dHJ1ZSwibnVtX2N0bXAiOnRydWUsIm51bV9uZHRhIjp0cnVlLCJpc19hY3RpdmUiOnRydWUsInVzZXJuYW1lIjp0cnVlLCJwYXNzd29yZCI6dHJ1ZSwiX2lkIjp0cnVlfSwibW9kaWZ5Ijp7fSwicmVxdWlyZSI6e319LCJzdGF0ZU5hbWVzIjpbInJlcXVpcmUiLCJtb2RpZnkiLCJpbml0IiwiZGVmYXVsdCIsImlnbm9yZSJdfSwiZW1pdHRlciI6eyJkb21haW4iOm51bGwsIl9ldmVudHMiOnt9LCJfZXZlbnRzQ291bnQiOjAsIl9tYXhMaXN0ZW5lcnMiOjB9fSwiaXNOZXciOmZhbHNlLCJfZG9jIjp7ImFkZHJlc3MiOnsiY291bnRyeSI6IlBlcnUiLCJ6aXAiOjQ1NzY0NSwic3RhdGUiOiJMaW1hIiwiY2l0eSI6IkxpbWEiLCJzdHJlZXQiOiJDYWxsZSBBbGFtZWRhIFNhbnRvcyAzNDQgRHB0byAzMDQifSwibWVkaWNhbENlbnRlcnMiOnsicmVxdWVzdGVkX2F0IjoiMjAxNi0xMi0xNVQwMTo1Mzo0Mi4xMzJaIiwiYWNjZXB0ZWRfYXQiOiIyMDE2LTExLTIwVDA0OjE5OjEzLjAwMFoiLCJzdGF0dXNfcmVxdWVzdCI6IjEiLCJuYW1lIjoiTHVpcyBNYW51ZWwiLCJfaWQiOiIzNDU2Nzg5MDQ1NjU4NDhmcjVnciJ9LCJfX3YiOjAsIm5hbWVzIjoiSG9ydGVuY2lhIiwiZ2VuZGVyIjoiNCIsImlkX0RvY3VtZW50X3R5cGUiOiJETkkiLCJpZF9Eb2N1bWVudF9udW0iOjEyMzQ1Njc4LCJiaXJ0aCI6IjIwMTYtMTEtMjBUMDQ6MTk6MTMuMDAwWiIsImVtYWlsIjoibWFudWVsQGdtYWlsLmNvbSIsInBob25lIjoiMjM0IDU0IDEzIiwiY2VsbHBob25lIjoiOTk5IDk5OSA5OTkiLCJudW1fY3RtcCI6NjU0MiwibnVtX25kdGEiOjU0NTM0NTQzLCJpc19hY3RpdmUiOmZhbHNlLCJ1c2VybmFtZSI6InRlcmFwZXV0YSIsInBhc3N3b3JkIjoiYWRtaW4iLCJfaWQiOiI1ODUxZjYwMTczZGMxMTA3MmEwYTFhOTIifSwiX3ByZXMiOnsiJF9fb3JpZ2luYWxfc2F2ZSI6W251bGwsbnVsbF0sIiRfX29yaWdpbmFsX3ZhbGlkYXRlIjpbbnVsbF0sIiRfX29yaWdpbmFsX3JlbW92ZSI6W251bGxdfSwiX3Bvc3RzIjp7IiRfX29yaWdpbmFsX3NhdmUiOltdLCIkX19vcmlnaW5hbF92YWxpZGF0ZSI6W10sIiRfX29yaWdpbmFsX3JlbW92ZSI6W119LCJpYXQiOjE0ODE3NjY4MjJ9.xDNN-rILCYc5vqVJzpLn3DIqOqMMPTEBuYHgvISoHPw');
    header = [contentType token];

    request = matlab.net.http.RequestMessage(method,header);

    uri = matlab.net.URI('http://52.89.123.49:8080/api/kinematics_analysis_matlab');
    response = send(request,uri);

end

function call_put_method_from_onlineMatlab_OK()

    uri = matlab.net.URI('http://52.89.123.49:8080/api/kinematics_analysis_matlab/58327fa19d4fe93d29435261');

    contentType = matlab.net.http.HeaderField('ContentType','application/json');
    token       = matlab.net.http.HeaderField('x-access-token','eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyIkX18iOnsic3RyaWN0TW9kZSI6dHJ1ZSwiZ2V0dGVycyI6e30sIndhc1BvcHVsYXRlZCI6ZmFsc2UsImFjdGl2ZVBhdGhzIjp7InBhdGhzIjp7Im1lZGljYWxDZW50ZXJzLnJlcXVlc3RlZF9hdCI6ImRlZmF1bHQiLCJfX3YiOiJpbml0IiwiYWRkcmVzcy5jb3VudHJ5IjoiaW5pdCIsImFkZHJlc3MuemlwIjoiaW5pdCIsImFkZHJlc3Muc3RhdGUiOiJpbml0IiwiYWRkcmVzcy5jaXR5IjoiaW5pdCIsImFkZHJlc3Muc3RyZWV0IjoiaW5pdCIsIm1lZGljYWxDZW50ZXJzLmFjY2VwdGVkX2F0IjoiaW5pdCIsIm1lZGljYWxDZW50ZXJzLnN0YXR1c19yZXF1ZXN0IjoiaW5pdCIsIm1lZGljYWxDZW50ZXJzLm5hbWUiOiJpbml0IiwibWVkaWNhbENlbnRlcnMuX2lkIjoiaW5pdCIsIm5hbWVzIjoiaW5pdCIsImdlbmRlciI6ImluaXQiLCJpZF9Eb2N1bWVudF90eXBlIjoiaW5pdCIsImlkX0RvY3VtZW50X251bSI6ImluaXQiLCJiaXJ0aCI6ImluaXQiLCJlbWFpbCI6ImluaXQiLCJwaG9uZSI6ImluaXQiLCJjZWxscGhvbmUiOiJpbml0IiwibnVtX2N0bXAiOiJpbml0IiwibnVtX25kdGEiOiJpbml0IiwiaXNfYWN0aXZlIjoiaW5pdCIsInVzZXJuYW1lIjoiaW5pdCIsInBhc3N3b3JkIjoiaW5pdCIsIl9pZCI6ImluaXQifSwic3RhdGVzIjp7Imlnbm9yZSI6e30sImRlZmF1bHQiOnsibWVkaWNhbENlbnRlcnMucmVxdWVzdGVkX2F0Ijp0cnVlfSwiaW5pdCI6eyJfX3YiOnRydWUsImFkZHJlc3MuY291bnRyeSI6dHJ1ZSwiYWRkcmVzcy56aXAiOnRydWUsImFkZHJlc3Muc3RhdGUiOnRydWUsImFkZHJlc3MuY2l0eSI6dHJ1ZSwiYWRkcmVzcy5zdHJlZXQiOnRydWUsIm1lZGljYWxDZW50ZXJzLmFjY2VwdGVkX2F0Ijp0cnVlLCJtZWRpY2FsQ2VudGVycy5zdGF0dXNfcmVxdWVzdCI6dHJ1ZSwibWVkaWNhbENlbnRlcnMubmFtZSI6dHJ1ZSwibWVkaWNhbENlbnRlcnMuX2lkIjp0cnVlLCJuYW1lcyI6dHJ1ZSwiZ2VuZGVyIjp0cnVlLCJpZF9Eb2N1bWVudF90eXBlIjp0cnVlLCJpZF9Eb2N1bWVudF9udW0iOnRydWUsImJpcnRoIjp0cnVlLCJlbWFpbCI6dHJ1ZSwicGhvbmUiOnRydWUsImNlbGxwaG9uZSI6dHJ1ZSwibnVtX2N0bXAiOnRydWUsIm51bV9uZHRhIjp0cnVlLCJpc19hY3RpdmUiOnRydWUsInVzZXJuYW1lIjp0cnVlLCJwYXNzd29yZCI6dHJ1ZSwiX2lkIjp0cnVlfSwibW9kaWZ5Ijp7fSwicmVxdWlyZSI6e319LCJzdGF0ZU5hbWVzIjpbInJlcXVpcmUiLCJtb2RpZnkiLCJpbml0IiwiZGVmYXVsdCIsImlnbm9yZSJdfSwiZW1pdHRlciI6eyJkb21haW4iOm51bGwsIl9ldmVudHMiOnt9LCJfZXZlbnRzQ291bnQiOjAsIl9tYXhMaXN0ZW5lcnMiOjB9fSwiaXNOZXciOmZhbHNlLCJfZG9jIjp7ImFkZHJlc3MiOnsiY291bnRyeSI6IlBlcnUiLCJ6aXAiOjQ1NzY0NSwic3RhdGUiOiJMaW1hIiwiY2l0eSI6IkxpbWEiLCJzdHJlZXQiOiJDYWxsZSBBbGFtZWRhIFNhbnRvcyAzNDQgRHB0byAzMDQifSwibWVkaWNhbENlbnRlcnMiOnsicmVxdWVzdGVkX2F0IjoiMjAxNi0xMi0xNVQwMTo1Mzo0Mi4xMzJaIiwiYWNjZXB0ZWRfYXQiOiIyMDE2LTExLTIwVDA0OjE5OjEzLjAwMFoiLCJzdGF0dXNfcmVxdWVzdCI6IjEiLCJuYW1lIjoiTHVpcyBNYW51ZWwiLCJfaWQiOiIzNDU2Nzg5MDQ1NjU4NDhmcjVnciJ9LCJfX3YiOjAsIm5hbWVzIjoiSG9ydGVuY2lhIiwiZ2VuZGVyIjoiNCIsImlkX0RvY3VtZW50X3R5cGUiOiJETkkiLCJpZF9Eb2N1bWVudF9udW0iOjEyMzQ1Njc4LCJiaXJ0aCI6IjIwMTYtMTEtMjBUMDQ6MTk6MTMuMDAwWiIsImVtYWlsIjoibWFudWVsQGdtYWlsLmNvbSIsInBob25lIjoiMjM0IDU0IDEzIiwiY2VsbHBob25lIjoiOTk5IDk5OSA5OTkiLCJudW1fY3RtcCI6NjU0MiwibnVtX25kdGEiOjU0NTM0NTQzLCJpc19hY3RpdmUiOmZhbHNlLCJ1c2VybmFtZSI6InRlcmFwZXV0YSIsInBhc3N3b3JkIjoiYWRtaW4iLCJfaWQiOiI1ODUxZjYwMTczZGMxMTA3MmEwYTFhOTIifSwiX3ByZXMiOnsiJF9fb3JpZ2luYWxfc2F2ZSI6W251bGwsbnVsbF0sIiRfX29yaWdpbmFsX3ZhbGlkYXRlIjpbbnVsbF0sIiRfX29yaWdpbmFsX3JlbW92ZSI6W251bGxdfSwiX3Bvc3RzIjp7IiRfX29yaWdpbmFsX3NhdmUiOltdLCIkX19vcmlnaW5hbF92YWxpZGF0ZSI6W10sIiRfX29yaWdpbmFsX3JlbW92ZSI6W119LCJpYXQiOjE0ODE3NjY4MjJ9.xDNN-rILCYc5vqVJzpLn3DIqOqMMPTEBuYHgvISoHPw');
    data = '{"frontal_ank_ang":[[0,36.6239548],[25,36.4931221],[50,37.341877],[75,36.720871],[100,37.5084839]],"frontal_hip_ang":[[0,-15.1074829],[25,-18.0453186],[50,-19.9585114],[75,-18.5435333],[100,-17.369957]],"frontal_kne_ang":[[0,11.1828613],[25,11.6198273],[50,11.7042236],[75,12.1821899],[100,12.5167847]],"frontal_pel_ang":[[0,32.3737335],[25,34.7032623],[50,35.633728],[75,33.6650696],[100,32.0141296]],"lank_x":[-0.196061328,-0.203023106,-0.218700916,-0.225095689,-0.231692597],"lank_y":[-0.420783848,-0.42057389,-0.421292871,-0.42000255,-0.419347942],"lank_z":[2.17449808,2.16681194,2.16470051,2.15764642,2.15254688],"lbwt_x":[-0.393105537,-0.396787971,-0.396567762,-0.396149278,-0.396363825],"lbwt_y":[0.5032323,0.505407453,0.504468918,0.503683329,0.503267586],"lbwt_z":[2.09581971,2.10831928,2.10714889,2.1042223,2.10406065],"lfwt_x":[-0.452916354,-0.453837514,-0.452574,-0.4539482,-0.455589712],"lfwt_y":[0.465313643,0.465899676,0.464322478,0.465187132,0.466238827],"lfwt_z":[2.06577969,2.06500983,2.05494,2.05894017,2.06281328],"lhee_x":[-0.309819102,-0.317615777,-0.332148224,-0.337794483,-0.346177489],"lhee_y":[-0.284540892,-0.285112321,-0.285361111,-0.284648478,-0.2865206],"lhee_z":[2.03655028,2.03527379,2.03279686,2.02381492,2.02748036],"lkne_x":[-0.449661583,-0.454836875,-0.461455703,-0.466332018,-0.471145421],"lkne_y":[0.0472802892,0.0475298762,0.0474403,0.0473098457,0.0473588556],"lkne_z":[2.07491016,2.07238793,2.05870271,2.059623,2.05925727],"lteo_x":[-0.363877326,-0.372853577,-0.38802281,-0.396190941,-0.401162326],"lteo_y":[-0.400856256,-0.402344018,-0.403678596,-0.404509485,-0.40268153],"lteo_z,":[2.09296727,2.09668612,2.09081817,2.08942771,2.07546449],"ltrc_x":[-0.587236404,-0.587881267,-0.586992264,-0.587410867,-0.58835113],"ltrc_y":[0.489899963,0.49217546,0.494793296,0.495378137,0.495900422],"ltrc_z":[2.08689213,2.08598948,2.08208537,2.08216238,2.08255839],"sagittal_ank_ang":[[0,36.6239548],[25,36.4931221],[50,37.341877],[75,36.720871],[100,37.5084839]],"sagittal_hip_ang":[[0,-15.1074829],[25,-18.0453186],[50,-19.9585114],[75,-18.5435333],[100,-17.369957]],"sagittal_kne_ang":[[0,11.1828613],[25,11.6198273],[50,11.7042236],[75,12.1821899],[100,12.5167847]],"sagittal_pel_ang":[[0,32.3737335],[25,34.7032623],[50,35.633728],[75,33.6650696],[100,32.0141296]],"transversal_ank_ang":[[0,36.6239548],[25,36.4931221],[50,37.341877],[75,36.720871],[100,37.5084839]],"transversal_hip_ang":[[0,-15.1074829],[25,-18.0453186],[50,-19.9585114],[75,-18.5435333],[100,-17.369957]],"transversal_kne_ang":[[0,11.1828613],[25,11.6198273],[50,11.7042236],[75,12.1821899],[100,12.5167847]],"transversal_pel_ang":[[0,32.3737335],[25,34.7032623],[50,35.633728],[75,33.6650696],[100,32.0141296]]}';
    data = matlab.net.http.HeaderField('data',data);
    header = [contentType token data];
    
    request=matlab.net.http.RequestMessage('put',header,matlab.net.http.MessageBody('useless'))
    response=request.send( uri)

end

%% Methods for cooking data to create a json which will be send to gaitcome.con server
function  res = cookKinematicData(LIST_ANGLES,LIST_RAW_POINTS)

    lstResAngles = createGaitCycleInPercentageObj(LIST_ANGLES);

    lstRawPoints = jsondecode(LIST_RAW_POINTS);
    obj = lstRawPoints(1,1);

    res  = ...
        jsonencode( ...
            containers.Map( ...
                {
                    'sagittal_hip_ang', ...
                    'sagittal_pel_ang', ...
                    'sagittal_kne_ang', ...
                    'sagittal_ank_ang' ...
                    'frontal_hip_ang', ...
                    'frontal_pel_ang', ...
                    'frontal_kne_ang', ...
                    'frontal_ank_ang' ...
                    'transversal_hip_ang', ...
                    'transversal_pel_ang', ...
                    'transversal_kne_ang', ...
                    'transversal_ank_ang' ...
                    'lbwt_x','lbwt_y','lbwt_z', ...
                    'lfwt_x','lfwt_y','lfwt_z', ...
                    'ltrc_x','ltrc_y','ltrc_z', ...
                    'lkne_x','lkne_y','lkne_z', ...
                    'lank_x','lank_y','lank_z', ...
                    'lhee_x','lhee_y','lhee_z', ...
                    'lteo_x','lteo_y','lteo_z,' ...
                }, ...
                {
                    lstResAngles{1}, ...
                    lstResAngles{2}, ...
                    lstResAngles{3}, ...
                    lstResAngles{4} ...
                    lstResAngles{1}, ...
                    lstResAngles{2}, ...
                    lstResAngles{3}, ...
                    lstResAngles{4} ...
                    lstResAngles{1}, ...
                    lstResAngles{2}, ...
                    lstResAngles{3}, ...
                    lstResAngles{4} ...
                    obj.lbwt_x,obj.lbwt_y,obj.lbwt_z, ...
                    obj.lfwt_x,obj.lfwt_y,obj.lfwt_z, ...
                    obj.ltrc_x,obj.ltrc_y,obj.ltrc_z, ...
                    obj.lkne_x,obj.lkne_y,obj.lkne_z, ...
                    obj.lank_x,obj.lank_y,obj.lank_z, ...
                    obj.lhee_x,obj.lhee_y,obj.lhee_z, ...
                    obj.lteo_x,obj.lteo_y,obj.lteo_z, ...
                } ...
            ) ...
        );

end
function res = cookRawData_MarkerPostions(DATA)

    data = jsondecode(DATA);
   
    obj = data(1,1);
    res  = ...
        jsonencode( ...
            containers.Map( ...
                { 
                    'lbwt_x','lbwt_y','lbwt_z', ...
                    'lfwt_x','lfwt_y','lfwt_z', ...
                    'ltrc_x','ltrc_y','ltrc_z', ...
                    'lkne_x','lkne_y','lkne_z', ...
                    'lank_x','lank_y','lank_z', ...
                    'lhee_x','lhee_y','lhee_z', ...
                    'lteo_x','lteo_y','lteo_z,' ...
                }, ...
                {
                    obj.lbwt_x,obj.lbwt_y,obj.lbwt_z, ...
                    obj.lfwt_x,obj.lfwt_y,obj.lfwt_z, ...
                    obj.ltrc_x,obj.ltrc_y,obj.ltrc_z, ...
                    obj.lkne_x,obj.lkne_y,obj.lkne_z, ...
                    obj.lank_x,obj.lank_y,obj.lank_z, ...
                    obj.lhee_x,obj.lhee_y,obj.lhee_z, ...
                    obj.lteo_x,obj.lteo_y,obj.lteo_z, ...
                } ...
            ) ...
        )

end

function  res = cookAnglesWithPercentageProgress(LIST_ANGLES)

    lstResAngles = createGaitCycleInPercentageObj(LIST_ANGLES);
    
    res  = ...
        jsonencode( ...
            containers.Map( ...
                {
                    'sagittal_hip_ang', ...
                    'sagittal_pel_ang', ...
                    'sagittal_kne_ang', ...
                    'sagittal_ank_ang' ...
                    'frontal_hip_ang', ...
                    'frontal_pel_ang', ...
                    'frontal_kne_ang', ...
                    'frontal_ank_ang' ...
                    'transversal_hip_ang', ...
                    'transversal_pel_ang', ...
                    'transversal_kne_ang', ...
                    'transversal_ank_ang' ...
                }, ...
                {
                    lstResAngles{1}, ...
                    lstResAngles{2}, ...
                    lstResAngles{3}, ...
                    lstResAngles{4} ...
                    lstResAngles{1}, ...
                    lstResAngles{2}, ...
                    lstResAngles{3}, ...
                    lstResAngles{4} ...
                    lstResAngles{1}, ...
                    lstResAngles{2}, ...
                    lstResAngles{3}, ...
                    lstResAngles{4} ...
                } ...
            ) ...
        );

end

function lstRes = createGaitCycleInPercentageObj(MULTI_LIST)

    mul_list = jsondecode(MULTI_LIST);
    mul_list = struct2cell(mul_list);

    lenLst =size(mul_list,1);

    for ii=1:lenLst
        lstRes{ii} = getHighChartsAngleObject_DATA(mul_list{ii});
    end

end

function res = getHighChartsAngleObject_DATA(LIST)

    lenLst =size(LIST,1);
    gaitIntervals = 100/(lenLst-1);
  
    for ii=1:lenLst
        % data for patient-angles filed
        percentualGaitProgress = (ii-1)*gaitIntervals;
        gaitAngle = LIST(ii,1);

        res{ii} = [ percentualGaitProgress, gaitAngle ];
        
        ii = ii+1;  
    end

end


function testPUTPUT(DATA) %OK It sends throug oniline matlab, since from here has a wierd problem

    uri = matlab.net.URI('http://52.89.123.49:8080/api/kinematics_analysis_matlab/58327fa19d4fe93d29435261');

    contentType = matlab.net.http.HeaderField('ContentType','application/json');
    token       = matlab.net.http.HeaderField('x-access-token','eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyIkX18iOnsic3RyaWN0TW9kZSI6dHJ1ZSwiZ2V0dGVycyI6e30sIndhc1BvcHVsYXRlZCI6ZmFsc2UsImFjdGl2ZVBhdGhzIjp7InBhdGhzIjp7Im1lZGljYWxDZW50ZXJzLnJlcXVlc3RlZF9hdCI6ImRlZmF1bHQiLCJfX3YiOiJpbml0IiwiYWRkcmVzcy5jb3VudHJ5IjoiaW5pdCIsImFkZHJlc3MuemlwIjoiaW5pdCIsImFkZHJlc3Muc3RhdGUiOiJpbml0IiwiYWRkcmVzcy5jaXR5IjoiaW5pdCIsImFkZHJlc3Muc3RyZWV0IjoiaW5pdCIsIm1lZGljYWxDZW50ZXJzLmFjY2VwdGVkX2F0IjoiaW5pdCIsIm1lZGljYWxDZW50ZXJzLnN0YXR1c19yZXF1ZXN0IjoiaW5pdCIsIm1lZGljYWxDZW50ZXJzLm5hbWUiOiJpbml0IiwibWVkaWNhbENlbnRlcnMuX2lkIjoiaW5pdCIsIm5hbWVzIjoiaW5pdCIsImdlbmRlciI6ImluaXQiLCJpZF9Eb2N1bWVudF90eXBlIjoiaW5pdCIsImlkX0RvY3VtZW50X251bSI6ImluaXQiLCJiaXJ0aCI6ImluaXQiLCJlbWFpbCI6ImluaXQiLCJwaG9uZSI6ImluaXQiLCJjZWxscGhvbmUiOiJpbml0IiwibnVtX2N0bXAiOiJpbml0IiwibnVtX25kdGEiOiJpbml0IiwiaXNfYWN0aXZlIjoiaW5pdCIsInVzZXJuYW1lIjoiaW5pdCIsInBhc3N3b3JkIjoiaW5pdCIsIl9pZCI6ImluaXQifSwic3RhdGVzIjp7Imlnbm9yZSI6e30sImRlZmF1bHQiOnsibWVkaWNhbENlbnRlcnMucmVxdWVzdGVkX2F0Ijp0cnVlfSwiaW5pdCI6eyJfX3YiOnRydWUsImFkZHJlc3MuY291bnRyeSI6dHJ1ZSwiYWRkcmVzcy56aXAiOnRydWUsImFkZHJlc3Muc3RhdGUiOnRydWUsImFkZHJlc3MuY2l0eSI6dHJ1ZSwiYWRkcmVzcy5zdHJlZXQiOnRydWUsIm1lZGljYWxDZW50ZXJzLmFjY2VwdGVkX2F0Ijp0cnVlLCJtZWRpY2FsQ2VudGVycy5zdGF0dXNfcmVxdWVzdCI6dHJ1ZSwibWVkaWNhbENlbnRlcnMubmFtZSI6dHJ1ZSwibWVkaWNhbENlbnRlcnMuX2lkIjp0cnVlLCJuYW1lcyI6dHJ1ZSwiZ2VuZGVyIjp0cnVlLCJpZF9Eb2N1bWVudF90eXBlIjp0cnVlLCJpZF9Eb2N1bWVudF9udW0iOnRydWUsImJpcnRoIjp0cnVlLCJlbWFpbCI6dHJ1ZSwicGhvbmUiOnRydWUsImNlbGxwaG9uZSI6dHJ1ZSwibnVtX2N0bXAiOnRydWUsIm51bV9uZHRhIjp0cnVlLCJpc19hY3RpdmUiOnRydWUsInVzZXJuYW1lIjp0cnVlLCJwYXNzd29yZCI6dHJ1ZSwiX2lkIjp0cnVlfSwibW9kaWZ5Ijp7fSwicmVxdWlyZSI6e319LCJzdGF0ZU5hbWVzIjpbInJlcXVpcmUiLCJtb2RpZnkiLCJpbml0IiwiZGVmYXVsdCIsImlnbm9yZSJdfSwiZW1pdHRlciI6eyJkb21haW4iOm51bGwsIl9ldmVudHMiOnt9LCJfZXZlbnRzQ291bnQiOjAsIl9tYXhMaXN0ZW5lcnMiOjB9fSwiaXNOZXciOmZhbHNlLCJfZG9jIjp7ImFkZHJlc3MiOnsiY291bnRyeSI6IlBlcnUiLCJ6aXAiOjQ1NzY0NSwic3RhdGUiOiJMaW1hIiwiY2l0eSI6IkxpbWEiLCJzdHJlZXQiOiJDYWxsZSBBbGFtZWRhIFNhbnRvcyAzNDQgRHB0byAzMDQifSwibWVkaWNhbENlbnRlcnMiOnsicmVxdWVzdGVkX2F0IjoiMjAxNi0xMi0xNVQwMTo1Mzo0Mi4xMzJaIiwiYWNjZXB0ZWRfYXQiOiIyMDE2LTExLTIwVDA0OjE5OjEzLjAwMFoiLCJzdGF0dXNfcmVxdWVzdCI6IjEiLCJuYW1lIjoiTHVpcyBNYW51ZWwiLCJfaWQiOiIzNDU2Nzg5MDQ1NjU4NDhmcjVnciJ9LCJfX3YiOjAsIm5hbWVzIjoiSG9ydGVuY2lhIiwiZ2VuZGVyIjoiNCIsImlkX0RvY3VtZW50X3R5cGUiOiJETkkiLCJpZF9Eb2N1bWVudF9udW0iOjEyMzQ1Njc4LCJiaXJ0aCI6IjIwMTYtMTEtMjBUMDQ6MTk6MTMuMDAwWiIsImVtYWlsIjoibWFudWVsQGdtYWlsLmNvbSIsInBob25lIjoiMjM0IDU0IDEzIiwiY2VsbHBob25lIjoiOTk5IDk5OSA5OTkiLCJudW1fY3RtcCI6NjU0MiwibnVtX25kdGEiOjU0NTM0NTQzLCJpc19hY3RpdmUiOmZhbHNlLCJ1c2VybmFtZSI6InRlcmFwZXV0YSIsInBhc3N3b3JkIjoiYWRtaW4iLCJfaWQiOiI1ODUxZjYwMTczZGMxMTA3MmEwYTFhOTIifSwiX3ByZXMiOnsiJF9fb3JpZ2luYWxfc2F2ZSI6W251bGwsbnVsbF0sIiRfX29yaWdpbmFsX3ZhbGlkYXRlIjpbbnVsbF0sIiRfX29yaWdpbmFsX3JlbW92ZSI6W251bGxdfSwiX3Bvc3RzIjp7IiRfX29yaWdpbmFsX3NhdmUiOltdLCIkX19vcmlnaW5hbF92YWxpZGF0ZSI6W10sIiRfX29yaWdpbmFsX3JlbW92ZSI6W119LCJpYXQiOjE0ODE3NjY4MjJ9.xDNN-rILCYc5vqVJzpLn3DIqOqMMPTEBuYHgvISoHPw');
    data = '{"frontal_ank_ang":[[0,36.6239548],[25,36.4931221],[50,37.341877],[75,36.720871],[100,37.5084839]],"frontal_hip_ang":[[0,-15.1074829],[25,-18.0453186],[50,-19.9585114],[75,-18.5435333],[100,-17.369957]],"frontal_kne_ang":[[0,11.1828613],[25,11.6198273],[50,11.7042236],[75,12.1821899],[100,12.5167847]],"frontal_pel_ang":[[0,32.3737335],[25,34.7032623],[50,35.633728],[75,33.6650696],[100,32.0141296]],"lank_x":[-0.196061328,-0.203023106,-0.218700916,-0.225095689,-0.231692597],"lank_y":[-0.420783848,-0.42057389,-0.421292871,-0.42000255,-0.419347942],"lank_z":[2.17449808,2.16681194,2.16470051,2.15764642,2.15254688],"lbwt_x":[-0.393105537,-0.396787971,-0.396567762,-0.396149278,-0.396363825],"lbwt_y":[0.5032323,0.505407453,0.504468918,0.503683329,0.503267586],"lbwt_z":[2.09581971,2.10831928,2.10714889,2.1042223,2.10406065],"lfwt_x":[-0.452916354,-0.453837514,-0.452574,-0.4539482,-0.455589712],"lfwt_y":[0.465313643,0.465899676,0.464322478,0.465187132,0.466238827],"lfwt_z":[2.06577969,2.06500983,2.05494,2.05894017,2.06281328],"lhee_x":[-0.309819102,-0.317615777,-0.332148224,-0.337794483,-0.346177489],"lhee_y":[-0.284540892,-0.285112321,-0.285361111,-0.284648478,-0.2865206],"lhee_z":[2.03655028,2.03527379,2.03279686,2.02381492,2.02748036],"lkne_x":[-0.449661583,-0.454836875,-0.461455703,-0.466332018,-0.471145421],"lkne_y":[0.0472802892,0.0475298762,0.0474403,0.0473098457,0.0473588556],"lkne_z":[2.07491016,2.07238793,2.05870271,2.059623,2.05925727],"lteo_x":[-0.363877326,-0.372853577,-0.38802281,-0.396190941,-0.401162326],"lteo_y":[-0.400856256,-0.402344018,-0.403678596,-0.404509485,-0.40268153],"lteo_z,":[2.09296727,2.09668612,2.09081817,2.08942771,2.07546449],"ltrc_x":[-0.587236404,-0.587881267,-0.586992264,-0.587410867,-0.58835113],"ltrc_y":[0.489899963,0.49217546,0.494793296,0.495378137,0.495900422],"ltrc_z":[2.08689213,2.08598948,2.08208537,2.08216238,2.08255839],"sagittal_ank_ang":[[0,36.6239548],[25,36.4931221],[50,37.341877],[75,36.720871],[100,37.5084839]],"sagittal_hip_ang":[[0,-15.1074829],[25,-18.0453186],[50,-19.9585114],[75,-18.5435333],[100,-17.369957]],"sagittal_kne_ang":[[0,11.1828613],[25,11.6198273],[50,11.7042236],[75,12.1821899],[100,12.5167847]],"sagittal_pel_ang":[[0,32.3737335],[25,34.7032623],[50,35.633728],[75,33.6650696],[100,32.0141296]],"transversal_ank_ang":[[0,36.6239548],[25,36.4931221],[50,37.341877],[75,36.720871],[100,37.5084839]],"transversal_hip_ang":[[0,-15.1074829],[25,-18.0453186],[50,-19.9585114],[75,-18.5435333],[100,-17.369957]],"transversal_kne_ang":[[0,11.1828613],[25,11.6198273],[50,11.7042236],[75,12.1821899],[100,12.5167847]],"transversal_pel_ang":[[0,32.3737335],[25,34.7032623],[50,35.633728],[75,33.6650696],[100,32.0141296]]}';
    data = matlab.net.http.HeaderField('data',data);
    header = [contentType token data];
    
    request=matlab.net.http.RequestMessage('put',header,matlab.net.http.MessageBody('useless'))
    response=request.send( uri)

end

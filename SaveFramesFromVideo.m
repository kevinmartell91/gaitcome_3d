% Extracting & Saving frames from a video file using Matlab

% Clearing workspace
clc;
close all;
clear all;

% Parameters for frame extraction
jumpBtwnFrames = 30;
fps = 120;
sec_ini = 9 / fps;
sec_end = 90;

% File paths for input and output
readfilename = {'../ekenRawFiles/camera_a/test_17_video/FHD0057.MOV', ... % camera A
                '../ekenRawFiles/camera_b/test_17_video/FHD0067.MOV'};    % camera B

savefilename = {'/visionData/videoCalibration/camera_a/snap_test_17_b', ... % camera A
                '/visionData/videoCalibration/camera_b/snap_test_17_b'};    % camera B

% Get the number of frames before the first blink for each camera
[outOfPhase_r, outOfPhase_l] = getNumFramesBeforeFirstBlink(readfilename);
outOfPhaseArray = {outOfPhase_l, ... % camera A
                    outOfPhase_r};    % camera B

% Extract frames from videos with synchronization process
for i = 1:2
    extractFramesFromVideoWithSyncProcess(readfilename{i}, ...
                                          savefilename{i}, ...
                                          outOfPhaseArray{i}, ...
                                          jumpBtwnFrames, ...
                                          sec_ini, ...
                                          sec_end, ...
                                          fps);
end


function [numFramesRight numFramesLeft] = getNumFramesBeforeFirstBlink (readfilename) 
    
    % Initialize variables
    isSync = false;
    outOfPhase_l = 0; % number of out of phase left frames
    outOfPhase_r = 0; % number of out of phase right frames
    isBlinking_l = false;
    isBlinking_r = false;
    SYNC_UMBRAL = 230;
    
    % Define limits
    MARGIN = 0;
    TOP_LIM = MARGIN;
    BOT_LIM = 720 - MARGIN;
    LEF_LIM = MARGIN;
    RIG_LIM = 1280 - MARGIN;

    iteFrames = 1;
    endFrames = 720; % first blink is no greater than 6 seconds

    % Initialize video players
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

    % Read video files
    movleft = VideoReader(readfilename{1});
    movRight = VideoReader(readfilename{2});

    % Iterate through video frames
    while ~isSync
        % Retrieve video frames before syncing process
        frameLeft = read(movleft, iteFrames);
        frameRight = read(movRight, iteFrames ); 

        step(player_left, frameLeft);
        step(player_right, frameRight);

        % Synchronization process
        [isSync, outOfPhase_l, outOfPhase_r, isBlinking_l, isBlinking_r] = ...
            DetectOutOfPhaseFrames(frameLeft, frameRight, outOfPhase_l, ...
                                   outOfPhase_r, isBlinking_l, isBlinking_r, ...
                                   SYNC_UMBRAL, TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM);

        iteFrames = iteFrames + 1;

        if isSync
            release(player_left);
            release(player_right);
        end
    end

    % Return the number of out of phase frames for each camera
    numFramesRight = outOfPhase_r;
    numFramesLeft = outOfPhase_l;
end

function data = extractFramesFromVideoWithSyncProcess(FILE_NAME, SAVE_FILENAME, ...
    OUT_OF_PHASE_FRAMES, FRAME_JUMPS, SEC_INI, SEC_END, FPS)
    
    % Read the video file
    mov = VideoReader(FILE_NAME);

    % Define the output folder for saving frames
    opFolder = fullfile(cd, SAVE_FILENAME);
    
    % Create the output folder if it doesn't exist
    if ~exist(opFolder, 'dir')
        mkdir(opFolder);
    end

    % Get the total number of frames in the video
    numFrames = mov.NumberOfFrames;

    % Initialize the counter for frames written
    numFramesWritten = 0;

    % Loop through frames from SEC_INI to SEC_END with a specified jump
    for t = SEC_INI * FPS : FRAME_JUMPS : numFrames
        if t < SEC_END * FPS
            % Read individual frames with an offset of OUT_OF_PHASE_FRAMES
            currFrame = read(mov, t + OUT_OF_PHASE_FRAMES);    
            opBaseFileName = sprintf('%3.3d.png', t);
            opFullFileName = fullfile(opFolder, opBaseFileName);
            imwrite(currFrame, opFullFileName, 'png');   % Save the frame as PNG
            % Display progress information
            progIndication = sprintf('Wrote frame %4d of %d.', t, numFrames);
            disp(progIndication);
            numFramesWritten = numFramesWritten + 1;
        end
    end

    % Display the total number of frames written
    progIndication = sprintf('Wrote %d frames to folder "%s"', numFramesWritten, opFolder);
    disp(progIndication);

    data = 0;
end
% DetectOutOfPhaseFrames
function [isSync, outOfPhase_l, outOfPhase_r, isBlinking_l, isBlinking_r] = ...
    DetectOutOfPhaseFrames(FRAME_LEFT, FRAME_RIGHT, OUT_OF_PHASE_L, OUT_OF_PHASE_R, ...
                            IS_BLINKING_LEFT, IS_BLINKING_RIGHT, SYNC_UMBRAL, ...
                            TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM)
    % Detects out-of-phase frames and blinking status for left and right frames

    isSync = false;

    if (IS_BLINKING_LEFT && IS_BLINKING_RIGHT)
        isSync = true;
    else
        if ~IS_BLINKING_LEFT
            [isBlinking_l, ~] = DetectFirstBlink(FRAME_LEFT, SYNC_UMBRAL, 50, ...
                                                 TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM);
            IS_BLINKING_LEFT = isBlinking_l;
            OUT_OF_PHASE_L = OUT_OF_PHASE_L + 1;
        end

        if ~IS_BLINKING_RIGHT
            [isBlinking_r, ~] = DetectFirstBlink(FRAME_RIGHT, SYNC_UMBRAL, 50, ...
                                                 TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM);
            IS_BLINKING_RIGHT = isBlinking_r;
            OUT_OF_PHASE_R = OUT_OF_PHASE_R + 1;
        end
    end

    outOfPhase_l = OUT_OF_PHASE_L;
    outOfPhase_r = OUT_OF_PHASE_R;
    isBlinking_l = IS_BLINKING_LEFT;
    isBlinking_r = IS_BLINKING_RIGHT;
end

% DetectFirstBlink
function [isBlinking, blobImage] = DetectFirstBlink(I1, THRESH_MIN, MIN_AREA, ...
                                                    TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM)
    % Detects the first blink in a frame

    isBlinking = false;

    GI1 = rgb2gray(I1);
    BW1_TH = THRESH_MIN < GI1;
    BW1 = bwconncomp(BW1_TH);

    stats = regionprops(BW1, 'Area', 'Eccentricity');
    mean_area = mean([stats.Area]);
    ids = find([stats.Area] >= MIN_AREA);
    blobImage = ismember(labelmatrix(BW1), ids);

    CC1 = regionprops(blobImage, 'centroid');
    markerPoints = single(cat(1, CC1.Centroid));

    length = size(markerPoints, 1);
    cont = 0;
    for i = 1:length
        if markerPoints(i, 2) > TOP_LIM && markerPoints(i, 2) < BOT_LIM && ...
           markerPoints(i, 1) > LEF_LIM && markerPoints(i, 1) < RIG_LIM
            cont = cont + 1;
        end
    end

    if cont > 0
        isBlinking = true;
    end
end

% extractFramesFromVideo
function data = extractFramesFromVideo(FILE_NAME, SAVE_FILENAME, FRAME_JUMPS, ...
                                       SEC_INI, SEC_END, FPS)
    % Extracts frames from a video file

    mov = VideoReader(FILE_NAME);
    opFolder = fullfile(cd, SAVE_FILENAME);

    if ~exist(opFolder, 'dir')
        mkdir(opFolder);
    end

    numFrames = mov.NumberOfFrames;
    numFramesWritten = 0;

    for t = SEC_INI * FPS : FRAME_JUMPS : numFrames
        if t < SEC_END * FPS
            currFrame = read(mov, t);
            opBaseFileName = sprintf('%3.3d.png', t);
            opFullFileName = fullfile(opFolder, opBaseFileName);
            imwrite(currFrame, opFullFileName, 'png');
            progIndication = sprintf('Wrote frame %4d of %d.', t, numFrames);
            disp(progIndication);
            numFramesWritten = numFramesWritten + 1;
        end
    end

    progIndication = sprintf('Wrote %d frames to folder "%s"', numFramesWritten, opFolder);
    disp(progIndication);

    data = 0;
end
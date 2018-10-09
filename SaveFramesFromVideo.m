%%Extracting & Saving of frames from a Video file through Matlab Code%%
clc;
close all;
clear all;

% assigning the name of sample avi file to a variable
% readfilename = {'./visionData/videoCalibration/camera_a/FHD0398.MOV', ... % camera A          
%             './visionData/videoCalibration/camera_b/FHD0384.MOV'};    % camera B
     


jumpBtwnFrames = 30;            
fps = 120; 
        
sec_ini = 9/fps;
sec_end= 90;

readfilename = {'../ekenRawFiles/camera_a/test_17_video/FHD0057.MOV', ... % camera A          
                '../ekenRawFiles/camera_b/test_17_video/FHD0067.MOV'};    % camera B
                

savefilename = {'/visionData/videoCalibration/camera_a/snap_test_17_b', ... % camera A          
                '/visionData/videoCalibration/camera_b/snap_test_17_b'};    % camera B


[outOfPhase_r, outOfPhase_l] = getNumFramesBeforeFirstBlink(readfilename);
outOfPhaseArray = { outOfPhase_l, ...   % camera A
                    outOfPhase_r }      % camera B


for i = 1:2 
    extractFramesFromVideoWithSyncProcess(  readfilename{i}, ...
                                            savefilename{i}, ...
                                            outOfPhaseArray{i}, ...
                                            jumpBtwnFrames, ...
                                            sec_ini, ...
                                            sec_end, ...
                                            fps);
end


function [numFramesRight numFramesLeft] = getNumFramesBeforeFirstBlink (readfilename) 
    
    % Out of Phase frames
    isSync = false;
    outOfPhase_l =0;             % number of out of phase left frames
    outOfPhase_r = 0;             % number of out of phase right frames
    isBlinking_l = false;
    isBlinking_r = false;
    SYNC_UMBRAL = 230;
    
    % Limits
    MARGIN = 0;
    TOP_LIM = MARGIN;
    BOT_LIM = 720 - MARGIN;
    LEF_LIM = MARGIN;
    RIG_LIM = 1280 - MARGIN;

    iteFrames = 1;
    endFrames = 720; % firts blink is no grater than 6 seconds

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


    % reading both video files
    movleft = VideoReader(readfilename{1});
    movRight = VideoReader(readfilename{2});

    %% iterating both video frames

        while ~isSync
            % retrieve video frames before syncing process
            frameLeft = read(movleft, iteFrames);
            frameRight = read(movRight, iteFrames ); 

            step(player_left, frameLeft);
            step(player_right, frameRight);

            % sync process
            [ isSync, outOfPhase_l, outOfPhase_r, isBlinking_l, isBlinking_r] = ...
                            DetectOutOfPhaseFrames(...
                                frameLeft, ...
                                frameRight, ...
                                outOfPhase_l, ...
                                outOfPhase_r, ...
                                isBlinking_l, ...
                                isBlinking_r, ...
                                SYNC_UMBRAL, ...
                                TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM ...
                            )

            iteFrames = iteFrames +1;

            if isSync
                release(player_left);
                release(player_right);
            end

        end
        numFramesRight = outOfPhase_r;
        numFramesLeft  = outOfPhase_l;
end

function data = extractFramesFromVideoWithSyncProcess (FILE_NAME,SAVE_FILENAME,OUT_OF_PHASE_FRAMES,FRAME_JUMPS, SEC_INI, SEC_END, FPS)

    %reading a video file
    mov = VideoReader(FILE_NAME);

    % Defining Output folder as 'snaps'
    opFolder = fullfile(cd, SAVE_FILENAME);
    % opFolder = fullfile(cd, '/visionData/videoCalibration/camera_b/snaps');
    %if  not existing 
    if ~exist(opFolder, 'dir')
    %make directory & execute as indicated in opfolder variable
    mkdir(opFolder);
    end

    %getting no of frames
    numFrames = mov.NumberOfFrames;

    %setting current status of number of frames written to zero
    numFramesWritten = 0;

    %for loop to traverse & process from frame '1' to 'last' frames 
    for t = SEC_INI * FPS :FRAME_JUMPS: numFrames
        if( t < SEC_END * FPS)
            %reading individual frames + OUT_OF_PHASE_FRAMES
            currFrame = read(mov, t + OUT_OF_PHASE_FRAMES);    
            opBaseFileName = sprintf('%3.3d.png', t);
            opFullFileName = fullfile(opFolder, opBaseFileName);
            imwrite(currFrame, opFullFileName, 'png');   %saving as 'png' file
            %indicating the current progress of the file/frame written
            progIndication = sprintf('Wrote frame %4d of %d.', t, numFrames);
            disp(progIndication);
            numFramesWritten = numFramesWritten + 1;
        end
            
    end      %end of 'for' loop
    progIndication = sprintf('Wrote %d frames to folder "%s"',numFramesWritten, opFolder);
    disp(progIndication);
    %End of the code


    data = 0;
end

% DetectOutOfPhaseFrames
function [ ...
    isSync ...
    outOfPhase_l  ...
    outOfPhase_r  ...
    isBlinking_l  ...
    isBlinking_r  ...
    ] = ...
    DetectOutOfPhaseFrames( ...
        FRAME_LEFT, ...
        FRAME_RIGHT, ...
        OUT_OF_PHASE_L, ...
        OUT_OF_PHASE_R,  ...
        IS_BLINKING_LEFT, ...
        IS_BLINKING_RIGHT, ...
        SYNC_UMBRAL, ...
        TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM ...
    )
    % 1 => L R 0 0 F F
    % 2 =>         F F
    % 3 =>         V F
    % 4 =>    11 0 V V
    % 5 =>             => isSync =  true
    isSync = false;

    if(IS_BLINKING_LEFT && IS_BLINKING_RIGHT)
        isSync= true;
    else

        if ~IS_BLINKING_LEFT
            [ isBlinking_l, blobImg_l ] = DetectFirstBlink(FRAME_LEFT, SYNC_UMBRAL, 50, TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM);
            IS_BLINKING_LEFT =  isBlinking_l;
            % counting the frames before the fist blink
            OUT_OF_PHASE_L = OUT_OF_PHASE_L + 1;
        end 

        if ~IS_BLINKING_RIGHT
            [ isBlinking_r, blobImg_r ] = DetectFirstBlink(FRAME_RIGHT, SYNC_UMBRAL, 50, TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM);
            IS_BLINKING_RIGHT = isBlinking_r;
            % counting the frames before the first blink
            OUT_OF_PHASE_R = OUT_OF_PHASE_R + 1;
        end
    end

    % results
    outOfPhase_l = OUT_OF_PHASE_L;
    outOfPhase_r = OUT_OF_PHASE_R;
    isBlinking_l = IS_BLINKING_LEFT;
    isBlinking_r = IS_BLINKING_RIGHT;
end

function [ isBlinking  blobImage ] = DetectFirstBlink(I1, THRESH_MIN, MIN_AREA, TOP_LIM, BOT_LIM, LEF_LIM, RIG_LIM)
    
    isBlinking = false;

    GI1 = rgb2gray(I1);
    BW1_TH = THRESH_MIN < GI1;
    BW1 = bwconncomp(BW1_TH); 
    
    stats = regionprops(BW1, 'Area','Eccentricity'); 
    mean_area =  mean([stats.Area]); 
    ids = find([stats.Area] >= MIN_AREA); 
    blobImage = ismember(labelmatrix(BW1), ids); 
    
    % Calculate centroids
    CC1 = regionprops(blobImage,'centroid');
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
 %     if (length > 0)
        isBlinking = true;
    end
end

function data = extractFramesFromVideo (FILE_NAME,SAVE_FILENAME,FRAME_JUMPS, SEC_INI, SEC_END, FPS)

    %reading a video file
    mov = VideoReader(FILE_NAME);

    % Defining Output folder as 'snaps'
    opFolder = fullfile(cd, SAVE_FILENAME);
    % opFolder = fullfile(cd, '/visionData/videoCalibration/camera_b/snaps');
    %if  not existing 
    if ~exist(opFolder, 'dir')
        %make directory & execute as indicated in opfolder variable
        mkdir(opFolder);
    end

    %getting no of frames
    numFrames = mov.NumberOfFrames;

    %setting current status of number of frames written to zero
    numFramesWritten = 0;

    %for loop to traverse & process from frame '1' to 'last' frames 
    for t = SEC_INI * FPS :FRAME_JUMPS: numFrames
        if( t < SEC_END * FPS)
            currFrame = read(mov, t);    %reading individual frames
            opBaseFileName = sprintf('%3.3d.png', t);
            opFullFileName = fullfile(opFolder, opBaseFileName);
            imwrite(currFrame, opFullFileName, 'png');   %saving as 'png' file
            %indicating the current progress of the file/frame written
            progIndication = sprintf('Wrote frame %4d of %d.', t, numFrames);
            disp(progIndication);
            numFramesWritten = numFramesWritten + 1;
        end
            
    end      %end of 'for' loop
    progIndication = sprintf('Wrote %d frames to folder "%s"',numFramesWritten, opFolder);
    disp(progIndication);
    %End of the code


    data = 0;
    
    % Read more: http://www.divilabs.com/2013/11/extracting-saving-of-frames-from-video.html#ixzz4tdheYcy4
end


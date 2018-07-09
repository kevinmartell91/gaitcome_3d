%%Extracting & Saving of frames from a Video file through Matlab Code%%
clc;
close all;
clear all;

% assigning the name of sample avi file to a variable
% readfilename = {'./visionData/videoCalibration/camera_a/FHD0398.MOV', ... % camera A          
%             './visionData/videoCalibration/camera_b/FHD0384.MOV'};    % camera B
        
readfilename = {'../ekenRawFiles/camera_a/test_12_video/FHD0657.MOV', ... % camera A          
                '../ekenRawFiles/camera_b/test_12_video/FHD0650.MOV'};    % camera B
                

savefilename = {'/visionData/videoCalibration/camera_a/snap_test_12_backgroundImg', ... % camera A          
                '/visionData/videoCalibration/camera_b/snap_test_12_backgroundImg'};    % camera B

sec_ini = 10;
sec_end= 14;

jumpBtwnFrames = 1;            
fps = 120; 

for i = 1:2 
    extractFramesFromVideo( readfilename{i}, ...
                            savefilename{i}, ...
                            jumpBtwnFrames, ...
                            sec_ini, ...
                            sec_end, ...
                            fps);
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
end

% Read more: http://www.divilabs.com/2013/11/extracting-saving-of-frames-from-video.html#ixzz4tdheYcy4
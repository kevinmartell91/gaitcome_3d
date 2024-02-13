% Initialize video file reader with a specified video file
videoFReader = vision.VideoFileReader('../ekenRawFiles/camera_b/test_07_video/FHD0432.MOV');
% Initialize video player for displaying video frames
videoPlayer = vision.VideoPlayer;

% Set the frame rate for video playback
framerate = 1; % in frames/second

% Create a timer object to control the playback rate
t = timer('ExecutionMode', 'fixedRate', ...
          'Period', 1 / framerate, ...
          'TimerFcn', @(tmr,evnt)timerfcn(tmr, videoFReader, videoPlayer), ...
          'ErrorFcn', @(tmr,evnt)cleanup(tmr, videoFReader, videoPlayer));

% Display the video player window
show(videoPlayer);
% Start the timer to begin video playback
start(t)

% Define the timer callback function for video playback
function timerfcn(tmr, reader, player)
    % Check if more frames are available and the player window is open
    if ~isDone(reader) && isOpen(player)
        % Read and display the next frame
        step(player, step(reader));
    else
        % Cleanup resources if playback is done or window is closed
        cleanup(tmr, reader, player)
    end
end

% Define the cleanup function to release resources
function cleanup(tmr, reader, player)
    % Stop and delete the timer if it's still running
    if isvalid(tmr) && strcmpi(tmr.Running, 'on')
        stop(tmr);
        delete(tmr)
    end
    % Release the video player and reader resources
    release(player);
    release(reader);
end
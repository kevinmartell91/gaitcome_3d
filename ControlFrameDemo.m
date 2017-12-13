videoFReader = vision.VideoFileReader('../ekenRawFiles/camera_b/test_07_video/FHD0432.MOV');
videoPlayer = vision.VideoPlayer;

framerate = 1; % in frames/second

t = timer('ExecutionMode', 'fixedRate', ...
          'Period', 1 / framerate, ...
          'TimerFcn', @(tmr,evnt)timerfcn(tmr, videoFReader, videoPlayer), ...
          'ErrorFcn', @(tmr,evnt)cleanup(tmr, videoFReader, videoPlayer));

show(videoPlayer);
start(t)

function timerfcn(tmr, reader, player)
    % While we have more to read, read and display it.
    if ~isDone(reader) && isOpen(player)
        step(player, step(reader));
    else
        cleanup(tmr, reader, player)
    end
end

function cleanup(tmr, reader, player)
    % Callback to ensure proper cleanup of objects
    if isvalid(tmr) && strcmpi(tmr.Running, 'on')
        stop(tmr);
        delete(tmr)
    end

    release(player);
    release(reader);
end

% afr = dsp.AudioFileReader('../ekenRawFiles/camera_b/test_07_video/FHD0432.MOV');
% adw = audioDeviceWriter('SampleRate', afr.SampleRate);
% 
% while ~isDone(afr)
%     audio = afr();
%     adw(audio);
% end
% release(afr); 
% release(adw);
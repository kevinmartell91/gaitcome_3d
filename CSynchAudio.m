classdef CSynchAudio
    %CSYNCHAUDIO Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        soundFile
    end
    
    methods
%         Do its job, but it crush before closing the function
        function r =  aviToWavFile(val, source, name)
          
            file = source;
            file1= strcat('audio_test_',name,'.WAV');
            
            [input_file, Fs] = audioread(file);
            audiowrite(file1, input_file, Fs);


            hmfr= video.MultimediaFileReader(file,'AudioOutputPort',true,'VideoOutputPort',false);
            hmfw = video.MultimediaFileWriter(file1,'AudioInputPort',true,'FileFormat','WAV');

            while ~isDone(hmfr)
                audioFrame = step(hmfr);
                step(hmfw,audioFrame);  
            end
            


            close(hmfw);
            close(hmfr);
            r = 99;
        end
    end
    
end


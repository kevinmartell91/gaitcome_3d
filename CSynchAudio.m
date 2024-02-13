classdef CSynchAudio
    % CSYNCHAUDIO provides functionality to convert AVI files to WAV format.
    
    properties
        soundFile % Placeholder for sound file path
    end
    
    methods
        function r = aviToWavFile(~, source, name)
            % Converts an AVI file to a WAV file.
            % Parameters:
            %   source: Path to the source AVI file.
            %   name: Base name for the output WAV file.
            
            % Define input and output file paths
            file = source;
            file1 = strcat('audio_test_', name, '.WAV');
            
            % Read the audio from the source file
            [input_file, Fs] = audioread(file);
            % Write the read audio to the output file
            audiowrite(file1, input_file, Fs);
            
            % Initialize multimedia file reader for audio extraction
            hmfr = video.MultimediaFileReader(file, 'AudioOutputPort', true, ...
                                              'VideoOutputPort', false);
            % Initialize multimedia file writer for saving audio in WAV format
            hmfw = video.MultimediaFileWriter(file1, 'AudioInputPort', true, ...
                                              'FileFormat', 'WAV');
            
            % Read and write audio frames until done
            while ~isDone(hmfr)
                audioFrame = step(hmfr);
                step(hmfw, audioFrame);  
            end
            
            % Close the file writer and reader objects
            close(hmfw);
            close(hmfr);
            
            % Return a dummy success value
            r = 99;
        end
    end
end
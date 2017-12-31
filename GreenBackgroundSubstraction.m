% https://www.mathworks.com/matlabcentral/answers/235307-removing-non-green-background-from-image
% https://www.mathworks.com/help/matlab/examples/convert-between-image-sequences-and-video.html

TEMP_NAME = './visionData/removeBackground/';
BINARY_COLOR = 255; %   0 = black
				  % 255 = white
INI_SEC = 7;
END_SEC = 9;

PROC_VIDEO = false;

VIDEO_INPUT_PATH = '../colorCameraRawFiles/ls3/test_11_video/VID_20171228_173722.mp4';
VIDEO_OUT_PATH = './visionData/removeBackground/video_edited';
FORMAT_VIDEO = 'MPEG-4';

if ~PROC_VIDEO
	% IMAGE = imread('./visionData/removeBackground/images/029.jpg');
	% img = removeBackground(IMAGE,BINARY_COLOR);
	% k1=imfuse(IMAGE, img,'montage'); %composite of 2images
	% imshow(k1); 

	removeBackgroundVideo(VIDEO_INPUT_PATH,VIDEO_OUT_PATH,INI_SEC,END_SEC,FORMAT_VIDEO);
	
else
	%% setup
	workingDir = TEMP_NAME;
	mkdir(workingDir)
	mkdir(workingDir,'images')

	%% create a video reader
	shuttleVideo = VideoReader(VIDEO_INPUT_PATH);
	iteFrames = INI_SEC * shuttleVideo.FrameRate;
	endFrames = END_SEC * shuttleVideo.FrameRate;

	%% create the image secuence
	ii = 1;
	while iteFrames < endFrames
	   img = read(shuttleVideo, iteFrames);
	   % img = removeBackground(img,BINARY_COLOR);

	   filename = [sprintf('%03d',ii) '.jpg'];
	   fullname = fullfile(workingDir,'images',filename);
	   imwrite(img,fullname)    % Write out to a JPEG file (img1.jpg, img2.jpg, etc.)
	   ii = ii+1;
	   iteFrames = iteFrames+1;

	end

	%% find image file names
	imageNames = dir(fullfile(workingDir,'images','*.jpg'));
	imageNames = {imageNames.name}';

	%% create new video with image sequence
	outputVideo = VideoWriter(fullfile(workingDir,'video_edited.avi'));
	outputVideo.FrameRate = shuttleVideo.FrameRate;
	open(outputVideo)

	%  write it to a video
	for ii = 1:length(imageNames)
	   img = imread(fullfile(workingDir,'images',imageNames{ii}));
	   writeVideo(outputVideo,img)
	end

	close(outputVideo)

	%% view the final video
	% construct a reader object.
	shuttleAvi = VideoReader(fullfile(workingDir,'video_edited.avi'));
	% Create a MATLAB movie struct from the video frames.

	ii = 1;
	while hasFrame(shuttleAvi)
	   mov(ii) = im2frame(readFrame(shuttleAvi));
	   ii = ii+1;
	end
	% Resize the current figure and axes based on the video's width and height, and view the first frame of the movie.
	figure
	imshow(mov(1).cdata, 'Border', 'tight')


	%Play back the movie once at the video's frame rate.

	movie(mov,1,shuttleAvi.FrameRate)

end


function img = removeBackground(IMAGE,BINARY_COLOR)

	r=single(IMAGE(:,:,1));
	g=single(IMAGE(:,:,2));
	b=single(IMAGE(:,:,3));

	ExGreen=2*g-r-b;
	ExRed=1.4*r-g-b;
	ExBlue=1.2*b-r-g;

	% Remove green
	dev=imsubtract(ExGreen,ExRed); 
	
	thres_level = multithresh(dev,1); % automatic thresholding
	seg_I = imquantize(dev,thres_level);
	RGB = label2rgb(seg_I,'gray');
	RGB2 = single(bwareaopen(RGB,1)); %clean the areas smaller than 1million pixels

	BW = RGB2(:,:,1);
	BW = uint8(imfill(BW,'holes'));
	    
	[row col] = size(BW); 

	for i = 1:row
		for j = 1:col
			if BW(i,j) == 0
			    BW(i,j) = 1 ;
			else
			    BW(i,j) = BINARY_COLOR ;
			end
		end
	end

	%% Remove Black   
	dev=imsubtract(ExGreen,ExBlue); %black
	thres_level = multithresh(dev,1); % automatic thresholding
	seg_I = imquantize(dev,thres_level);
	RGB = label2rgb(seg_I,'gray');
	RGB2 = single(bwareaopen(RGB,1)); %clean the areas smaller than 1million pixels

	BW2 = RGB2(:,:,1);
	BW2 = uint8(imfill(BW2,'holes'));

	for i = 1:row
		for j = 1:col
			if BW2(i,j) == 0
			    BW(i,j) = BINARY_COLOR ;
			end
		end
	end

	body_only = IMAGE;
	for a = 1:3
	    body_only(:,:,a) = body_only(:,:,a).*BW;
	end

	img = body_only;
end


function removeBackgroundVideo(VIDEO_INPUT_PATH,VIDEO_OUT_PATH,INI_SEC,END_SEC,FORMAT_VIDEO)

	% https://blogs.mathworks.com/steve/2014/08/12/it-aint-easy-seeing-green-unless-you-have-matlab/

	% Open the input video files
	% v1 = VideoReader('background.mp4');
	v2 = VideoReader(VIDEO_INPUT_PATH);
	% Determing the size of the white image as a background
	whiteImage = 255 * ones(v2.Height,v2.Width, 3, 'uint8');
	blackImage = 0 * ones(v2.Height,v2.Width, 3, 'uint8');
	% Determine the number of frames for the final video
	% nFrames = min(v1.NumberOfFrames,v2.NumberOfFrames);
	% nFrames = v2.NumberOfFrames;
	% Set the output dimensions
	% outDims = [400 640];
	% Open a video file to write the result to
	vOut = VideoWriter(VIDEO_OUT_PATH,FORMAT_VIDEO);
	vOut.FrameRate = v2.FrameRate;
	open(vOut);
	% Loop over all the frames
	for k = INI_SEC*v2.FrameRate:END_SEC*v2.FrameRate
	    % Get the kth frame of both inputs
	    % x = imresize(read(v1,k),outDims);
	    % y = imresize(read(v2,k),outDims);
	    y = read(v2,k);
	    % Mix them together
	    z = y;  % Preallocate space for the result
	    % Find the green pixels in the foreground (y)
	    yd = double(y)/255; 
	    % Greenness = G*(G-R)*(G-B)
	    greenness = yd(:,:,2).*(yd(:,:,2)-yd(:,:,1)).*(yd(:,:,2)-yd(:,:,3));
	    % Threshold the greenness value
	    thresh = 0.02*mean(greenness(greenness>0));
	    isgreen = greenness > thresh;
	    % Thicken the outline to expand the greenscreen mask a little
	    outline = edge(isgreen,'roberts');
	    se = strel('disk',1);
	    outline = imdilate(outline,se);
	    isgreen = isgreen | outline;
	    % Blend the images
	    % Loop over the 3 color planes (RGB)
	    for j = 1:3
	        rgb1 = blackImage(:,:,j);  % Extract the jth plane of the background
	        rgb2 = y(:,:,j);  % Extract the jth plane of the foreground
	        % Replace the green pixels of the foreground with the background
	        rgb2(isgreen) = rgb1(isgreen);
	        % Put the combined image into the output
	        z(:,:,j) = rgb2;
	    end
	    k1=imfuse(y, z,'montage'); %composite of 2images
	    imshow(k1); 
	    % Write the result to file
	    writeVideo(vOut,z)
	end
	% And we're done!
	close(vOut);

end


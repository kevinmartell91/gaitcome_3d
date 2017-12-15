% https://www.mathworks.com/matlabcentral/answers/235307-removing-non-green-background-from-image


BINARY_COLOR = 0;

al1 = imread('./visionData/greenBackground/green_01.png');        

r=single(al1(:,:,1));
g=single(al1(:,:,2));
b=single(al1(:,:,3));

ExGreen=2*g-r-b;
ExRed=1.4*r-g-b;
ExBlue=1.2*b-r-g;

dev=imsubtract(ExGreen,ExRed); %gree
% Remove green
thres_level = multithresh(dev,1); % automatic thresholding
seg_I = imquantize(dev,thres_level);
RGB = label2rgb(seg_I,'gray');
RGB2 = single(bwareaopen(RGB,1)); %clean the areas smaller than 1million pixels

BW = RGB2(:,:,1);
BW = uint8(imfill(BW,'holes'));
    
[row col] = size(BW); 

%%% Remove Greem   
for i = 1:row
	for j = 1:col
		if BW(i,j) == 0
		    BW(i,j) = 1 ;
		else
		    BW(i,j) = BINARY_COLOR ;
		end
	end
end

leaves_only = al1;
for a = 1:3
    leaves_only(:,:,a) = leaves_only(:,:,a).*BW;
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

leaves_only = al1;
for a = 1:3
    leaves_only(:,:,a) = leaves_only(:,:,a).*BW;
end




% subplot(2,2,1);
% k1=imfuse(al1, ExBlue,'montage'); %composite of 2images
% imshow(k1); 

% subplot(2,2,2);
% k1=imfuse(al1, ExGreen,'montage'); %composite of 2images
% imshow(k1); 

% subplot(2,2,3);
% k1=imfuse(al1, ExRed,'montage'); %composite of 2images
% imshow(k1); 

% subplot(2,2,4);
k1=imfuse(al1, leaves_only,'montage'); %composite of 2images
imshow(k1); 



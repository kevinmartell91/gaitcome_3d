% cb = checkerboard(4,2);
% cb_ref = imref2d(size(cb));
% 
% background = zeros(150);
% imshowpair(cb,cb_ref,background,imref2d(size(background)))
% 
% 
% T = [1 0 0;0 1 0;100 0 1];
% tform_t = affine2d(T);
% 
% R = [cosd(30) sind(30) 0;-sind(30) cosd(30) 0;0 0 1];
% tform_r = affine2d(R);
% 
% TR = T*R;
% tform_tr = affine2d(TR);
% [out,out_ref] = imwarp(cb,cb_ref,tform_tr);
% imshowpair(out,out_ref,background,imref2d(size(background)))
% 
% RT = R*T;
% tform_rt = affine2d(RT);
% [out,out_ref] = imwarp(cb,cb_ref,tform_rt);
% imshowpair(out,out_ref,background,imref2d(size(background)))

% I=imread('./visionData/removeBackground/images_black/039.jpg');
I=imread('./visionData/removeBackground/780.png');

% projective 2d
% theta = 1;
% tm = [cosd(theta) sind(theta) 0.0001; ...
%     sind(theta) cosd(theta) 0.00001; ...
%     0 0 1];
theta = 10.001;
tm = [4 0 0.0001; ...
    0 4 0.00001; ...
    0 0 4];
tform = projective2d(tm);

outputImage = imwarp(I,tform);
figure
imshow(outputImage);

% I=double(I);
% 
% s=[2,3];
% tform1 = maketform('affine',[s(1) 0 0; 0 s(2) 0; 0 0 1]);       % scaling
% I1 = imtransform(I,tform1);
% 
% sh=[0.5 0];
% tform2 = maketform('affine',[1 sh(1) 0; sh(2) 1 0; 0 0 1]);     % shear
% I2 = imtransform(I,tform2);
% 
% % theta=3*pi/4;                                                   % rotation
% theta=pi/2;
% A=[cos(theta) sin(theta) 0; -sin(theta) cos(theta) 0; 0 0 1];
% tform3 = maketform('affine',A);
% I3 = imtransform(I,tform3);
% 
% figure
% subplot(2,2,1),imagesc(I),axis image
% title('Original','FontSize',18)
% subplot(2,2,2),imagesc(I1),axis image
% title('Scaled','FontSize',18)
% subplot(2,2,3),imagesc(I2),axis image
% title('Shear','FontSize',18)
% subplot(2,2,4),imagesc(I3),axis image
% title('Rotation','FontSize',18)
% colormap(gray)
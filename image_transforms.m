%% image_transforms.m
%  based on Matlab 2014a image processing toolbox code

clear all; close all;

%% Non-reflective Similarity
%  [u v] = [x y 1] T, where T is 3x3 matrix with 4 parameters
I=checkerboard(10,2);
%I=imread('cameraman.tif');
figure; imshow(I); title('original');

scale = 1.2;        %scale factor
angle = 40*pi/180;  %radians
tx = 10;  %x translation
ty = -10; %y translation

sc = scale*cos(angle);
ss = scale*sin(angle);

T = [sc -ss 0;
     ss  sc 0;
     tx  ty 1;];

%create a 2D affine object since T is a subset of affine transforms
t_nonsim = affine2d(T); %RI is spatial reference of an object
[I_nonreflective_similarity,RI] = imwarp(I,t_nonsim,'FillValues',.3);
figure; imshow(I_nonreflective_similarity,RI); title('nonreflective similarity'); axis on;


%% Reflective Similarity
%  [u v] = [x y 1] T, where T is 3x3 matrix with 5 parameters
I=checkerboard(10,2);
%I=imread('cameraman.tif');
figure; imshow(I); title('original');

scale = 1.5;        %scale factor
angle = 10*pi/180;  %radians
tx = 0; %x translation
ty = 0; %y translation
a = -1; %-1 reflection / +1 no reflection

sc = scale*cos(angle);
ss = scale*sin(angle);

T = [  sc   -ss   0;
     a*ss  a*sc   0;
       tx    ty   1;];

%create a 2D affine object since T is a subset of affine transforms
t_sim = affine2d(T); %RI is spatial reference of an object
[I_similarity,RI] = imwarp(I,t_sim,'FillValues',.3);
figure; imshowpair(I,I_similarity,'montage'); title('similarity'); axis on;


%% Affine Transformation
%  [u v] = [x y 1] T, where T is 3x3 matrix with 6 parameters
%  parallel lines remain parallel, straight lines remain straight
I=checkerboard(10,2);
%I=imread('cameraman.tif');
figure; imshow(I); title('original');

T = [1 0.3  0;
     1   1  0;
     0   0  1;];

%create a 2D affine object since T is a subset of affine transforms
t_aff = affine2d(T); %RI is spatial reference of an object
[I_affine,RI] = imwarp(I,t_aff,'FillValues',.3);
figure; imshowpair(I,I_affine,'montage'); title('affine'); axis on;


%% Projective Transformation (Homography)
%  [up vp wp] = [x y w] T, where T is 3x3 with 9 parameters
%  Let T = [ [A B C]', [D E F]', [G H I]' ], then 
%  u = (Ax + By + C) / (Gx + Hy + I) and v = (Dx + Ey + F) / (Gx + Hy + I)
%  quadrilaterals map to quadrilaterals (parallel lines are no longer
%  parallel), straight lines remain straight
%I=imread('cameraman.tif');

T = [1   0  0.008;
     1   1   0.01;
     0   0    1;];

t_proj = projective2d(T); %RI is spatial reference of an object
I_projective = imwarp(I,t_proj,'FillValues',.3);
figure; imshow(I_projective); title('projective'); axis on;

%% Polynomial Transformation
% [u,v] = [1 x y xy x^2 y^2]
%I=imread('cameraman.tif');
fixedPoints = reshape(randn(12,1),6,2);
movingPoints = fixedPoints;
%fit a geometric transformation given type: polynomial (picks best model)
t_poly = fitgeotrans(movingPoints,fixedPoints,'polynomial',2);
I_polynomial = imwarp(I,t_poly,'FillValues',.3);
figure; imshow(I_polynomial); title('polynomial');

%% Piece-wise Linear Transformation
%I=imread('cameraman.tif');
movingPoints = [10 10; 10 30; 30 30; 30 10];
fixedPoints  = [10 10; 10 30; 40 35; 30 10];
t_piecewise_linear = fitgeotrans(movingPoints,fixedPoints,'pwl');
I_piecewise_linear = imwarp(I,t_piecewise_linear);
figure; imshow(I_piecewise_linear); title('piecewise linear'); axis on;

%% Sinusoidal Transformation
% explicity mapping of each point (xi,yi) to (u,v)
%I=imread('cameraman.tif');
[nrows, ncols] = size(I);
[xi,yi] = meshgrid(1:ncols, 1:nrows);
a1=5; a2=3; %sinusoid amplitude
imid = round(size(I,2)/2);

u = xi + a1*sin(pi*xi/imid);
v = yi - a2*sin(pi*yi/imid);

tmap_B = cat(3,u,v); %concatenate u and v
resamp = makeresampler('linear','fill');

I_sinusoid = tformarray(I,[],resamp,[2 1],[1 2],[],tmap_B,.3);
figure; imshow(I_sinusoid); title('sinusoid'); axis on;

%% Barrel Transformation
% perturbs an image radially outward from its center resulting in convex
% sides

%I=imread('cameraman.tif');
[nrows, ncols] = size(I);
[xi,yi] = meshgrid(1:ncols, 1:nrows);
imid = round(size(I,2)/2);

xt = xi(:) - imid;
yt = yi(:) - imid;

[theta,r] = cart2pol(xt,yt);
a = .001; % Try varying the amplitude of the cubic term.
s = r + a*r.^3; %barrel transform in the radial domain
[ut,vt] = pol2cart(theta,s);

u = reshape(ut,size(xi)) + imid;
v = reshape(vt,size(yi)) + imid;

tmap_B = cat(3,u,v);
resamp = makeresampler('linear','fill');

I_barrel = tformarray(I,[],resamp,[2 1],[1 2],[],tmap_B,.3);
figure; imshow(I_barrel); title('barrel'); axis on;

%% Pin Cushion Transformation
%  inverse of a barrel transformation
%I=imread('cameraman.tif');
[nrows, ncols] = size(I);
[xi,yi] = meshgrid(1:ncols, 1:nrows);
imid = round(size(I,2)/2);

xt = xi(:) - imid;
yt = yi(:) - imid;

[theta,r] = cart2pol(xt,yt);
a = -0.0005; % Try varying the amplitude of the cubic term.
s = r + a*r.^3; %barrel transform in the radial domain
[ut,vt] = pol2cart(theta,s);

u = reshape(ut,size(xi)) + imid;
v = reshape(vt,size(yi)) + imid;

tmap_B = cat(3,u,v);
resamp = makeresampler('linear','fill');

I_pin = tformarray(I,[],resamp,[2 1],[1 2],[],tmap_B,.3);
figure; imshow(I_pin); title('pin cushion'); axis on;










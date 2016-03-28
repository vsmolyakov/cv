%% Color based segmentation using K-means clustering

clear all; close all;

%%

he = imread('hestain.png'); 
figure; imshow(he); title('H&E image');

cform = makecform('srgb2lab');
lab_he = applycform(he,cform);

%use k-means to cluster objects (colors in ab)
ab = double(he(:,:,2:3));
nrows = size(ab,1); ncols = size(ab,2);
ab = reshape(ab, nrows*ncols, 2);

nColors = 3; %number of clusters
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);

pixel_labels = reshape(cluster_idx,nrows,ncols);
figure; imshow(pixel_labels,[]); title('image segmented by k-means');

%original colorspace images segmented by k-means
segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels, [1 1 3]);

for k = 1:nColors
   color = he; %original color space
   color(rgb_label ~= k) = 0; %zero all but color k components
   segmented_images{k} = color;
end

figure; 
subplot(131); imshow(segmented_images{1}); title('cluster 1');
subplot(132); imshow(segmented_images{2}); title('cluster 2');
subplot(133); imshow(segmented_images{3}); title('cluster 3');


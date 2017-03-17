

train_dir_ap = './Caltect4/ImageData/airplanes_train/';
test_dir_ap = './Caltech4/ImageData/ap_test/';

contents = dir(strcat(test_dir_ap, '*.jpg'));

image = imread(strcat(test_dir_ap, contents(1).name));
image = single(rgb2gray(image));


% find a subset of images from each class, 250 from each class
% concatnate their descriptors
% cluster those descriptors


% for each image, find descriptor
% Assign each descriptor to a cluster
% find the frequency of descriptors in each cluster



[f, d] = vl_sift(image);
size(d)
d = double(d);

MAXIT = 10;
MAXK = 400;

[idx,C, ~, D] = kmeans(d', MAXK);
size(idx)
size(C)
size(D)         % 619 x 400, each descriptor to cluster center distance
                % 619 descriptors and 400 clusters

% finding the closest cluster, min_index: index of closest cluster
[~, min_index] = min(D, [], 2);

% count the frequency of each cluster
[freq, cluster_index] = hist(min_index, unique(min_index));
size(freq)
size(cluster_index)

% probably store those according to the image.

% iterate through every image in the folder
% for i=1:size(contents, 1)   
%     image = imread(strcat(test_dir_ap, contents(i).name));
%     image = single(rgb2gray(image));
%     
%     [f, d] = vl_sift(image);    
%     
% end

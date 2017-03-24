totalImages = 250; % number of images to load per class
kmeans_iterations = 100;

kmeans_clusters_path = char(strcat(data_folder, 'preprocessing/kmeans_clusters', '_', string(clusters_number), '.mat'));
kmeans_descriptors_path = char(strcat(data_folder, 'preprocessing/kmeans_descriptors', '_', string(totalImages), '.mat'));

% create a matrix with all the descriptors of the different images
if exist(kmeans_descriptors_path, 'file')
    load(kmeans_descriptors_path, 'descriptors');
else
    descriptors = loadDescriptors(totalImages, sift_type);
    save(kmeans_descriptors_path, 'descriptors');
end

% do k-means clustering of the descriptors, C is the centroids matrix
if exist(kmeans_clusters_path, 'file')
    load(kmeans_clusters_path, 'C');
else
    [idx,C] = kmeans(descriptors', clusters_number, 'Display', 'iter', 'MaxIter', kmeans_iterations);
    save(kmeans_clusters_path, 'C');
end


%%
function [descriptors] = loadDescriptors(numImages, sift_type)
    descriptors = [];
    dataset_dir = '../Caltech4/ImageData/';

    contents = dir(dataset_dir); % all the image folders
    % loop over all the folders
    for i = 1:numel(contents)
        foldername = contents(i).name;
        if contains(foldername, 'train')
            folder_contents = dir(strcat(dataset_dir, foldername, '/*.jpg'));
            disp(foldername)
             % loop over all the files in each folder
            for j=1:numImages
                filename = folder_contents(j).name;

                image = imread(strcat(dataset_dir, foldername, '/', filename));

                [~, d] = sift(sift_type, image);
                descriptors = [descriptors, d];
            end
        end
    end
    descriptors = double(descriptors);
end
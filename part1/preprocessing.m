% create a matrix with all the descriptors of the different images
descriptors = loadDescriptors(80);
size(descriptors);

% do k-means clustering of the descriptors, C is the centroids matrix
% size(C) = 400x128
clusters_number = 400; % clusters number
[idx,C] = kmeans(descriptors', clusters_number);

%%
function [descriptors] = loadDescriptors(imagesPerFolder)
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
            for j=1:imagesPerFolder
                filename = folder_contents(j).name;

                image = imread(strcat(dataset_dir, foldername, '/', filename));
                if size(image, 3) > 1
                    image = single(rgb2gray(image));
                else
                    image = single(image);
                end

                [~, d] = sift('grayscale', image);
                descriptors = [descriptors, d];
            end
        end
    end
    descriptors = double(descriptors);
end
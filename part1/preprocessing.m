%% Visual vocabulary building
% in order to create the visual vocabulary, first the descriptors of the
% first preprocessing_images are extracted, followed by the clusters
% calculation using vl_kmeans.

% various folder paths used
kmeans_clusters_path = [data_folder, 'preprocessing/kmeans_clusters', '_', num2str(clusters_number), '_', num2str(preprocessing_images), '.mat'];
kmeans_descriptors_path = [data_folder, 'preprocessing/kmeans_descriptors', '_', num2str(preprocessing_images), '.mat'];

% check if the descriptors have already been calculated
if exist(kmeans_descriptors_path, 'file')
    % load the descriptors
    load(kmeans_descriptors_path, 'descriptors');
else
    % create a matrix with all the descriptors of the different images
    disp('---descriptors calculation');
    descriptors = loadDescriptors(preprocessing_images, descriptor_type, preprocessing_descriptors);
    % save the descriptors
    save(kmeans_descriptors_path, 'descriptors');
end

% check if the visual vocabulary has already been calculated
if exist(kmeans_clusters_path, 'file')
    % load the visual vocabulary
    load(kmeans_clusters_path, 'C');
else
    % run k-means clustering of the descriptors, C contains the centroids
    % coordinates
    disp('---kmeans clustering');
    [C, ~] = vl_kmeans(descriptors, clusters_number, 'verbose', 'algorithm', 'elkan');
    C = double(C');
    
    % save the visual vocabulary
    save(kmeans_clusters_path, 'C');
end


%% Support functions

% calculation of the descriptors of the first preprocessing_images
%
% --inputs
% preprocessing_images: number of images to load per class
% descriptor_type: type of descriptor used
% preprocessing_descriptors: number of descriptors to return per image, if
% -1 all descriptors are returned
%
% --outputs
% descriptors: matrix containing the descriptors of all the loaded images

function [descriptors] = loadDescriptors(preprocessing_images, descriptor_type, preprocessing_descriptors)
    descriptors = [];
    dataset_dir = '../Caltech4/ImageData/';

    contents = dir(dataset_dir); % all the image folders
    % loop over all the folders
    for i = 1:numel(contents)
        foldername = contents(i).name;
        
        % enter only in the train folders
        if contains(foldername, 'train')
            folder_contents = dir(strcat(dataset_dir, foldername, '/*.jpg'));
            disp(foldername)
            
             % loop over all the files in each folder
            for j=1:preprocessing_images
                filename = folder_contents(j).name;
                
                % read the image
                image = imread(strcat(dataset_dir, foldername, '/', filename));

                % calculate the descriptors
                [~, d] = getDescriptors(descriptor_type, image, preprocessing_descriptors);

                % add the descriptors to the final matrix
                descriptors = [descriptors, d]; %#ok<AGROW>
            end
        end
    end
    
    % convert the matrix to single in order to save space in memory
    % (serious problem when using the matlab built in kmeans function)
    descriptors = single(descriptors);
end
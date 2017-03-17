%now i try to get all the airplanes and check the accuracy =):D
cluster_number = 400;
imagesPerFolder = 50;
dataset_dir = '../Caltech4/ImageData/';
contents = dir(dataset_dir); % all the image folders

test_positives = [];
test_negatives = [];

% loop over all the folders
for i = 1:numel(contents)
    foldername = contents(i).name;
    if contains(foldername, 'test')
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
            h = descriptorToHistogram(cluster_number, C, d);
            
            if contains(foldername, 'airplanes')
                test_positives = [test_positives; h];
            else
                test_negatives = [test_negatives; h];
            end
        end
    end
end
test_labels = [repmat(1,[imagesPerFolder 1]); repmat(0,[imagesPerFolder * 3 1])];
test_data_matrix = sparse([test_positives; test_negatives]);
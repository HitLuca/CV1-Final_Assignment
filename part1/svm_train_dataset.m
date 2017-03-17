% i try to create the positive/negative examples for svm training
% #justForAirplanes

cluster_number = 400;
imagesPerFolder = 100;
dataset_dir = '../Caltech4/ImageData/';
contents = dir(dataset_dir); % all the image folders

train_positives = [];
train_negatives = [];

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
            h = descriptorToHistogram(cluster_number, C, d);
            
            if contains(foldername, 'airplanes')
                train_positives = [train_positives; h];
            else
                train_negatives = [train_negatives; h];
            end
        end
    end
end
train_positives = double(train_positives);
train_negatives = double(train_negatives);

train_labels = [repmat(1,[imagesPerFolder 1]); repmat(0,[imagesPerFolder * 3 1])];
train_data_matrix = sparse([train_positives;train_negatives]);
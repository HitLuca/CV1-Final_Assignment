% i try to create the positive/negative examples for svm training
cluster_number = 400;
dataset_dir = '../Caltech4/ImageData/';
contents = dir(dataset_dir); % all the image folders

training_airplanes = [];
training_cars = [];
training_faces = [];
training_motorbikes = [];

% loop over all the folders
for i = 1:numel(contents)
    foldername = contents(i).name;
    if contains(foldername, 'train')
        folder_contents = dir(strcat(dataset_dir, foldername, '/*.jpg'));
        disp(foldername)
        
        % loop over all the files in each folder
        for j=1:numel(folder_contents)
            filename = folder_contents(j).name;

            image = imread(strcat(dataset_dir, foldername, '/', filename));
            if size(image, 3) > 1
                image = single(rgb2gray(image));
            else
                image = single(image);
            end
            
            % generate the histogram for the specific image and put it in
            % the right matrix
            [~, d] = sift('grayscale', image);
            h = descriptorToHistogram(cluster_number, C, d);
            if contains(foldername, 'airplanes')
                training_airplanes = [training_airplanes; h];
            elseif contains(foldername, 'cars')
                training_cars = [training_cars; h];
            elseif contains(foldername, 'faces')
                training_faces = [training_faces; h];
            elseif contains(foldername, 'motorbikes')
                training_motorbikes = [training_motorbikes; h];
            end
        end
    end
end

save('data/training_airplanes.mat', 'training_airplanes');
save('data/training_cars.mat', 'training_cars');
save('data/training_faces.mat', 'training_faces');
save('data/training_motorbikes.mat', 'training_motorbikes');
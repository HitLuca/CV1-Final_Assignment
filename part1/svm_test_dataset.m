cluster_number = 400;
dataset_dir = '../Caltech4/ImageData/';
contents = dir(dataset_dir); % all the image folders

testing_airplanes = [];
testing_cars = [];
testing_faces = [];
testing_motorbikes = [];

% loop over all the folders
for i = 1:numel(contents)
    foldername = contents(i).name;
    if contains(foldername, 'test')
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

            [~, d] = sift('grayscale', image);
            h = descriptorToHistogram(cluster_number, C, d);
            if contains(foldername, 'airplanes')
                testing_airplanes = [testing_airplanes; h];
            elseif contains(foldername, 'cars')
                testing_cars = [testing_cars; h];
            elseif contains(foldername, 'faces')
                testing_faces = [testing_faces; h];
            elseif contains(foldername, 'motorbikes')
                testing_motorbikes = [testing_motorbikes; h];
            end
        end
    end
end

save('data/testing_airplanes.mat', 'testing_airplanes');
save('data/testing_cars.mat', 'testing_cars');
save('data/testing_faces.mat', 'testing_faces');
save('data/testing_motorbikes.mat', 'testing_motorbikes');
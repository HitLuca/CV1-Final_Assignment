testing_airplanes_path = strcat(data_folder, 'testing_data/testing_airplanes.mat');
testing_cars_path = strcat(data_folder, 'testing_data/testing_cars.mat');
testing_faces_path = strcat(data_folder, 'testing_data/testing_faces.mat');
testing_motorbikes_path = strcat(data_folder, 'testing_data/testing_motorbikes.mat');

if exist(testing_airplanes_path, 'file')
    load(testing_airplanes_path, 'testing_airplanes');
    load(testing_cars_path, 'testing_cars');
    load(testing_faces_path, 'testing_faces');
    load(testing_motorbikes_path, 'testing_motorbikes');
else
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

                [~, d] = sift(sift_type, image);
                h = descriptorToHistogram(clusters_number, C, d);
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

    save(testing_airplanes_path, 'testing_airplanes');
    save(testing_cars_path, 'testing_cars');
    save(testing_faces_path, 'testing_faces');
    save(testing_motorbikes_path, 'testing_motorbikes');
end
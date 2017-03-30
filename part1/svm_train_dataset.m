training_airplanes_path = strcat(data_folder, 'training_data/training_airplanes.mat');
training_cars_path = strcat(data_folder, 'training_data/training_cars.mat');
training_faces_path = strcat(data_folder, 'training_data/training_faces.mat');
training_motorbikes_path = strcat(data_folder, 'training_data/training_motorbikes.mat');

if exist(training_airplanes_path, 'file')
    load(training_airplanes_path, 'training_airplanes');
    load(training_cars_path, 'training_cars');
    load(training_faces_path, 'training_faces');
    load(training_motorbikes_path, 'training_motorbikes');
else
    % i try to create the positive/negative examples for svm training
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
            for j=preprocessing_images+1:numel(folder_contents)
                filename = folder_contents(j).name;
                
                image = imread(strcat(dataset_dir, foldername, '/', filename));

                % generate the histogram for the specific image and put it in
                % the right matrix
                [~, d] = sift(sift_type, image, -1);
                
                h = descriptorToHistogram(clusters_number, C, d);
                
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
    
    save(training_airplanes_path, 'training_airplanes');
    save(training_cars_path, 'training_cars');
    save(training_faces_path, 'training_faces');
    save(training_motorbikes_path, 'training_motorbikes');
end
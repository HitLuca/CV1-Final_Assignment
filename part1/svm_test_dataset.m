%% Creation of the testing dataset
% the testing dataset is created by loading every test image, creating a
% histogram representation of the extracted descriptors and storing
% it along the histograms for every other image of the same class

%#ok<*AGROW>
 
% various folder paths used
testing_airplanes_path = [data_folder, 'testing_data/testing_airplanes.mat'];
testing_cars_path = [data_folder, 'testing_data/testing_cars.mat'];
testing_faces_path = [data_folder, 'testing_data/testing_faces.mat'];
testing_motorbikes_path = [data_folder, 'testing_data/testing_motorbikes.mat'];

% check if the dataset has already been computed
if exist(testing_airplanes_path, 'file')
    % load the dataset
    load(testing_airplanes_path, 'testing_airplanes');
    load(testing_cars_path, 'testing_cars');
    load(testing_faces_path, 'testing_faces');
    load(testing_motorbikes_path, 'testing_motorbikes');
else
    % create the dataset
    dataset_dir = '../Caltech4/ImageData/';
    contents = dir(dataset_dir); % all the image folders

    testing_airplanes = [];
    testing_cars = [];
    testing_faces = [];
    testing_motorbikes = [];

    % loop over all the folders
    for i = 1:numel(contents)
        foldername = contents(i).name;
        
        % enter only in the test folders
        if contains(foldername, 'test')
            folder_contents = dir(strcat(dataset_dir, foldername, '/*.jpg'));
            disp(foldername)

             % loop over all the files in each folder
            for j=1:numel(folder_contents)
                filename = folder_contents(j).name;
                
                % read the image
                image = imread(strcat(dataset_dir, foldername, '/', filename));

                % calculate the descriptors
                [~, d] = getDescriptors(descriptor_type, image, -1);
                
                % convert the descriptors to an histogram
                h = descriptorToProbabilisticHistogram(clusters_number, C, d);
                
                % concatenate the histogram with the correct matrix
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

    % save the dataset
    save(testing_airplanes_path, 'testing_airplanes');
    save(testing_cars_path, 'testing_cars');
    save(testing_faces_path, 'testing_faces');
    save(testing_motorbikes_path, 'testing_motorbikes');
end

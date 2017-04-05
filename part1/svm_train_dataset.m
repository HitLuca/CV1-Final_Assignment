%% Creation of the training dataset
% the training dataset is created by loading every train image (excluding 
% the ones already used for the visual vocabulary creation), creating a
% histogram representation of the extracted descriptors and storing
% it along the histograms for every other image of the same class

 %#ok<*AGROW>
 
% various folder paths used
training_airplanes_path = [data_folder, 'training_data/training_airplanes.mat'];
training_cars_path = [data_folder, 'training_data/training_cars.mat'];
training_faces_path = [data_folder, 'training_data/training_faces.mat'];
training_motorbikes_path = [data_folder, 'training_data/training_motorbikes.mat'];

% check if the dataset has already been computed
if exist(training_airplanes_path, 'file')
    % load the dataset
    load(training_airplanes_path, 'training_airplanes');
    load(training_cars_path, 'training_cars');
    load(training_faces_path, 'training_faces');
    load(training_motorbikes_path, 'training_motorbikes');
else
    % create the dataset
    dataset_dir = '../Caltech4/ImageData/';
    contents = dir(dataset_dir); % all the image folders

    training_airplanes = [];
    training_cars = [];
    training_faces = [];
    training_motorbikes = [];

    % loop over all the folders
    for i = 1:numel(contents)
        foldername = contents(i).name;
        
        % enter only in the train folders
        if contains(foldername, 'train')
            folder_contents = dir(strcat(dataset_dir, foldername, '/*.jpg'));
            disp(foldername)

            % loop over all the files in each folder, excluding the ones
            % used in the visual vocabulary creation
            for j=preprocessing_images+1:numel(folder_contents)
                filename = folder_contents(j).name;
                
                % read the image
                image = imread(strcat(dataset_dir, foldername, '/', filename));

                % calculate the descriptors
                [~, d] = getDescriptors(descriptor_type, image, -1);
                
                % convert the descriptors to an histogram
                h = descriptorToProbabilisticHistogram(clusters_number, C, d);
                
                % concatenate the histogram with the correct matrix
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
    
    % save the dataset
    save(training_airplanes_path, 'training_airplanes');
    save(training_cars_path, 'training_cars');
    save(training_faces_path, 'training_faces');
    save(training_motorbikes_path, 'training_motorbikes');
end
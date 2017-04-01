%% Training of the svm classifiers
% this script trains the four svm classifiers using the training dataset
% and the chosen kernel type

% various folder paths used
airplanes_model_filepath = strcat(data_folder, 'models/', kernel_type, '/airplanes_model_', kernel_type, '.mat');
cars_model_filepath = strcat(data_folder, 'models/', kernel_type, '/cars_model_', kernel_type, '.mat');
faces_model_filepath = strcat(data_folder, 'models/', kernel_type, '/faces_model_', kernel_type, '.mat');
motorbikes_model_filepath = strcat(data_folder, 'models/', kernel_type, '/motorbikes_model_', kernel_type, '.mat');

% check if the models have already been trained
if exist(airplanes_model_filepath, 'file')
    % load the models
    load(airplanes_model_filepath, 'airplanes_model');
    load(cars_model_filepath, 'cars_model');
    load(faces_model_filepath, 'faces_model');
    load(motorbikes_model_filepath, 'motorbikes_model');
else
    % train the airplanes svm
    % create the training matrix
    train_images = double([training_airplanes; training_cars; training_faces; training_motorbikes]);
    
    % use a sparse representation
    train_data_matrix = sparse(train_images);

    % create the training labels
    train_labels = [ones(size(training_airplanes, 1), 1); 
        zeros(size(training_cars, 1), 1);
        zeros(size(training_faces, 1), 1);
        zeros(size(training_motorbikes, 1), 1)];

    % train the model
    airplanes_model = svmtrain(train_labels, train_data_matrix, strcat('-q', kernel_param));
    
    % save the model
    save(airplanes_model_filepath, 'airplanes_model');

    
    % train the cars svm
    % create the training matrix
    train_images = double([training_cars; training_airplanes; training_faces; training_motorbikes]);
    
    % use a sparse representation
    train_data_matrix = sparse(train_images);

    % create the training labels
    train_labels = [ones(size(training_cars, 1), 1); 
        zeros(size(training_airplanes, 1), 1);
        zeros(size(training_faces, 1), 1);
        zeros(size(training_motorbikes, 1), 1)];

    % train the model
    cars_model = svmtrain(train_labels, train_data_matrix, strcat('-q', kernel_param));
    
    % save the model
    save(cars_model_filepath, 'cars_model');

   
    % train the faces svm
    % create the training matrix
    train_images = double([training_faces; training_cars; training_airplanes; training_motorbikes]);
   
    % use a sparse representation
    train_data_matrix = sparse(train_images);

    % create the training labels
    train_labels = [ones(size(training_faces, 1), 1); 
        zeros(size(training_cars, 1), 1);
        zeros(size(training_airplanes, 1), 1);
        zeros(size(training_motorbikes, 1), 1)];

    % train the model
    faces_model = svmtrain(train_labels, train_data_matrix, strcat('-q', kernel_param));
   
    % save the model
    save(faces_model_filepath, 'faces_model');

    
    % train the motorbikes svm
    % create the training matrix
    train_images = double([training_motorbikes; training_cars; training_faces; training_airplanes]);
    
    % use a sparse representation
    train_data_matrix = sparse(train_images);

    % create the training labels
    train_labels = [ones(size(training_motorbikes, 1), 1); 
        zeros(size(training_cars, 1), 1);
        zeros(size(training_faces, 1), 1);
        zeros(size(training_airplanes, 1), 1)];

    % train the model
    motorbikes_model = svmtrain(train_labels, train_data_matrix, strcat('-q', kernel_param));
    
    % save the model
    save(motorbikes_model_filepath, 'motorbikes_model');
end
airplanes_model_filepath = strcat(data_folder, 'models/', kernel_type, '/airplanes_model_', kernel_type, '.mat');
cars_model_filepath = strcat(data_folder, 'models/', kernel_type, '/cars_model_', kernel_type, '.mat');
faces_model_filepath = strcat(data_folder, 'models/', kernel_type, '/faces_model_', kernel_type, '.mat');
motorbikes_model_filepath = strcat(data_folder, 'models/', kernel_type, '/motorbikes_model_', kernel_type, '.mat');

if exist(airplanes_model_filepath, 'file')
    load(airplanes_model_filepath, 'airplanes_model');
    load(cars_model_filepath, 'cars_model');
    load(faces_model_filepath, 'faces_model');
    load(motorbikes_model_filepath, 'motorbikes_model');
else
    % airplanes svm
    train_images = double([training_airplanes; training_cars; training_faces; training_motorbikes]);
    train_data_matrix = sparse(train_images);

    train_labels = [ones(size(training_airplanes, 1), 1); 
        zeros(size(training_cars, 1), 1);
        zeros(size(training_faces, 1), 1);
        zeros(size(training_motorbikes, 1), 1)];

    airplanes_model = svmtrain(train_labels, train_data_matrix, strcat('-q', kernel_param));
    save(airplanes_model_filepath, 'airplanes_model');

    % cars svm
    train_images = double([training_cars; training_airplanes; training_faces; training_motorbikes]);
    train_data_matrix = sparse(train_images);

    train_labels = [ones(size(training_cars, 1), 1); 
        zeros(size(training_airplanes, 1), 1);
        zeros(size(training_faces, 1), 1);
        zeros(size(training_motorbikes, 1), 1)];

    cars_model = svmtrain(train_labels, train_data_matrix, strcat('-q', kernel_param));
    save(cars_model_filepath, 'cars_model');

    % faces svm
    train_images = double([training_faces; training_cars; training_airplanes; training_motorbikes]);
    train_data_matrix = sparse(train_images);

    train_labels = [ones(size(training_faces, 1), 1); 
        zeros(size(training_cars, 1), 1);
        zeros(size(training_airplanes, 1), 1);
        zeros(size(training_motorbikes, 1), 1)];

    faces_model = svmtrain(train_labels, train_data_matrix, strcat('-q', kernel_param));
    save(faces_model_filepath, 'faces_model');

    % motorbikes svm
    train_images = double([training_motorbikes; training_cars; training_faces; training_airplanes]);
    train_data_matrix = sparse(train_images);

    train_labels = [ones(size(training_motorbikes, 1), 1); 
        zeros(size(training_cars, 1), 1);
        zeros(size(training_faces, 1), 1);
        zeros(size(training_airplanes, 1), 1)];

    motorbikes_model = svmtrain(train_labels, train_data_matrix, strcat('-q', kernel_param));
    save(motorbikes_model_filepath, 'motorbikes_model');
end
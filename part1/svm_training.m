

%% airplanes svm
train_images = double([training_airplanes; training_cars; training_faces; training_motorbikes]);
train_data_matrix = sparse(train_images);

train_labels = [ones(size(training_airplanes, 1), 1); 
    zeros(size(training_cars, 1), 1);
    zeros(size(training_faces, 1), 1);
    zeros(size(training_motorbikes, 1), 1)];

airplanes_model = train(train_labels, train_data_matrix);
save(strcat(data_folder, 'models/airplanes_model.mat'), 'airplanes_model');

%% cars svm
train_images = double([training_cars; training_airplanes; training_faces; training_motorbikes]);
train_data_matrix = sparse(train_images);

train_labels = [ones(size(training_cars, 1), 1); 
    zeros(size(training_airplanes, 1), 1);
    zeros(size(training_faces, 1), 1);
    zeros(size(training_motorbikes, 1), 1)];

cars_model = train(train_labels, train_data_matrix);
save(strcat(data_folder, 'models/cars_model.mat'), 'cars_model');

%% faces svm
train_images = double([training_faces; training_cars; training_airplanes; training_motorbikes]);
train_data_matrix = sparse(train_images);

train_labels = [ones(size(training_faces, 1), 1); 
    zeros(size(training_cars, 1), 1);
    zeros(size(training_airplanes, 1), 1);
    zeros(size(training_motorbikes, 1), 1)];

faces_model = train(train_labels, train_data_matrix);
save(strcat(data_folder, 'models/faces_model.mat'), 'faces_model');

%% motorbikes svm
train_images = double([training_motorbikes; training_cars; training_faces; training_airplanes]);
train_data_matrix = sparse(train_images);

train_labels = [ones(size(training_motorbikes, 1), 1); 
    zeros(size(training_cars, 1), 1);
    zeros(size(training_faces, 1), 1);
    zeros(size(training_airplanes, 1), 1)];

motorbikes_model = train(train_labels, train_data_matrix);
save(strcat(data_folder, 'models/motorbikes_model.mat'), 'motorbikes_model');
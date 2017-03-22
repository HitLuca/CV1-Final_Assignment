train_images = double([training_airplanes; training_cars; training_faces; training_motorbikes]);
train_data_matrix = sparse(train_images);

%% airplanes svm
train_labels = [repmat(1,[size(training_airplanes, 1), 1]); 
    repmat(-1,[size(training_cars, 1) 1]);
    repmat(-1,[size(training_faces, 1) 1]);
    repmat(-1,[size(training_motorbikes, 1) 1])];

airplanes_model = train(train_labels, train_data_matrix);
save('data/models/airplanes_model.mat', 'airplanes_model');

%% cars svm
train_labels = [repmat(-1,[size(training_airplanes, 1), 1]); 
    repmat(1,[size(training_cars, 1) 1]);
    repmat(-1,[size(training_faces, 1) 1]);
    repmat(-1,[size(training_motorbikes, 1) 1])];

cars_model = train(train_labels, train_data_matrix);
save('data/models/cars_model.mat', 'cars_model');

%% faces svm
train_labels = [repmat(-1,[size(training_airplanes, 1), 1]); 
    repmat(-1,[size(training_cars, 1) 1]);
    repmat(1,[size(training_faces, 1) 1]);
    repmat(-1,[size(training_motorbikes, 1) 1])];

faces_model = train(train_labels, train_data_matrix);
save('data/models/faces_model.mat', 'faces_model');

%% motorbikes svm
train_labels = [repmat(-1,[size(training_airplanes, 1), 1]); 
    repmat(-1,[size(training_cars, 1) 1]);
    repmat(-1,[size(training_faces, 1) 1]);
    repmat(1,[size(training_motorbikes, 1) 1])];

motorbikes_model = train(train_labels, train_data_matrix);
save('data/models/motorbikes_model.mat', 'motorbikes_model');
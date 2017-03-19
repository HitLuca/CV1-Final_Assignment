% in order to get libwathever working, first go into the matlab folder and
% run make, then go back one folder and run addpath matlab

%% airplanes svm
train_positives = double(training_airplanes);
train_negatives = double([training_cars;training_faces;training_motorbikes]);

train_labels = [repmat(1,[size(train_positives, 1), 1]); repmat(0,[size(train_negatives, 1) 1])];

train_data_matrix = sparse([train_positives;train_negatives]);

airplanes_model = train(train_labels, train_data_matrix);
save('data/airplanes_model.mat', 'airplanes_model');

%% cars svm
train_positives = double(training_cars);
train_negatives = double([training_airplanes;training_faces;training_motorbikes]);

train_labels = [repmat(1,[size(train_positives, 1), 1]); repmat(0,[size(train_negatives, 1) 1])];

train_data_matrix = sparse([train_positives;train_negatives]);

cars_model = train(train_labels, train_data_matrix);
save('data/cars_model.mat', 'cars_model');

%% faces svm
train_positives = double(training_faces);
train_negatives = double([training_airplanes;training_cars;training_motorbikes]);

train_labels = [repmat(1,[size(train_positives, 1), 1]); repmat(0,[size(train_negatives, 1) 1])];

train_data_matrix = sparse([train_positives;train_negatives]);

faces_model = train(train_labels, train_data_matrix);
save('data/faces_model.mat', 'faces_model');

%% motorbikes svm
train_positives = double(training_motorbikes);
train_negatives = double([training_airplanes;training_cars;training_faces]);

train_labels = [repmat(1,[size(train_positives, 1), 1]); repmat(0,[size(train_negatives, 1) 1])];

train_data_matrix = sparse([train_positives;train_negatives]);

motorbikes_model = train(train_labels, train_data_matrix);
save('data/motorbikes_model.mat', 'motorbikes_model');

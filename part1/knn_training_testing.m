% create the training matrix
train_images = double([training_airplanes; training_cars; training_faces; training_motorbikes]);

% create the training labels
train_labels = [repmat(1, size(training_airplanes, 1), 1);
    repmat(2, size(training_cars, 1), 1);
    repmat(3, size(training_faces, 1), 1);
    repmat(4, size(training_motorbikes, 1), 1)];

% fit the knn model
model = fitcknn(train_images,train_labels);

% create the testing matrix
test_images = double([testing_airplanes; testing_cars; testing_faces; testing_motorbikes]);

% create the testing labels
test_labels = [repmat(1, size(testing_airplanes, 1), 1);
    repmat(2, size(testing_cars, 1), 1);
    repmat(3, size(testing_faces, 1), 1);
    repmat(4, size(testing_motorbikes, 1), 1)];

% get predictions
[output_labels, probabilities, costs] = predict(model,test_images);

% calculate model accuracy
accuracy = sum(test_labels == output_labels) / numel(test_labels);


%% testing the svm models
% this routine takes the correct svm model and proceeds to test it with the
% loaded testing dataset

% convert the dataset to double
test_data_matrix = double([testing_airplanes; testing_cars; testing_faces; testing_motorbikes]);

% airplanes test
% load the model
load([data_folder, 'models/', kernel_type, '/airplanes_model_', kernel_type, '.mat'], 'airplanes_model');

% create test labels
test_labels = [ones(size(testing_airplanes, 1), 1); 
    zeros(size(testing_cars, 1), 1);
    zeros(size(testing_faces, 1), 1);
    zeros(size(testing_motorbikes, 1), 1)];

% get predictions
[airplanes_labels, airplanes_prob] = predict(airplanes_model, test_data_matrix);

% calculate ap
airplanes_ap = average_precision(airplanes_prob(:,2), test_labels, size(testing_airplanes, 1));

% calculate acc
airplanes_acc = accuracy(airplanes_labels, test_labels);

% cars test
% load the model
load([data_folder, 'models/', kernel_type, '/cars_model_', kernel_type, '.mat'], 'cars_model');

% create test labels
test_labels = [zeros(size(testing_airplanes, 1), 1); 
    ones(size(testing_cars, 1), 1);
    zeros(size(testing_faces, 1), 1);
    zeros(size(testing_motorbikes, 1), 1)];

% get predictions
[cars_labels, cars_prob] = predict(cars_model, test_data_matrix);

% calculate ap
cars_ap = average_precision(cars_prob(:,2), test_labels, size(testing_cars, 1));

% calculate acc
cars_acc = accuracy(cars_labels, test_labels);

% faces test
% load the model
load([data_folder, 'models/', kernel_type, '/faces_model_', kernel_type, '.mat'], 'faces_model');

% create test labels
test_labels = [zeros(size(testing_airplanes, 1), 1); 
    zeros(size(testing_cars, 1), 1);
    ones(size(testing_faces, 1), 1);
    zeros(size(testing_motorbikes, 1), 1)];

% get predictions
[faces_labels, faces_prob] = predict(faces_model, test_data_matrix);

% calculate ap
faces_ap = average_precision(faces_prob(:,2), test_labels, size(testing_faces, 1));

% calculate acc
faces_acc = accuracy(faces_labels, test_labels);

% motorbikes test
% load the model
load([data_folder, 'models/', kernel_type, '/motorbikes_model_', kernel_type, '.mat'], 'motorbikes_model');

% create test labels
test_labels = [zeros(size(testing_airplanes, 1), 1); 
    zeros(size(testing_cars, 1), 1);
    zeros(size(testing_faces, 1), 1);
    ones(size(testing_motorbikes, 1), 1)];

% get predictions
[motorbikes_labels, motorbikes_prob] = predict(motorbikes_model, test_data_matrix);

% calculate ap
motorbikes_ap = average_precision(motorbikes_prob(:,2), test_labels, size(testing_motorbikes, 1));

% calculate acc
motorbikes_acc = accuracy(motorbikes_labels, test_labels);


% calculation of the MAP
mean_average_precision = (airplanes_ap + cars_ap + faces_ap + motorbikes_ap) / 4;

% calculation of the MACC
mean_accuracy = (airplanes_acc + cars_acc + faces_acc + motorbikes_acc) / 4;


%% Support functions

% Class assignments calculations
% determines the classifier assignments to the various classes
% 
% --inputs
% probabilities: classifier probabilities
% labels: testing labels
%
% --outputs
% assignments: classifier assignments

function [assignments] = calculate_class_assignments(probabilities, labels)
    % sort the probabilities
    [~, indexes] = sort(probabilities, 'descend');
    
    % get the assignments
    assignments = labels(indexes);
end


% Accuracy calculation
% calculation of the accuracy for a model
% 
% --inputs
% labels: classifier labels
% test_labels: testing labels
%
% --outputs
% acc: calculated accuracy

function [acc] = accuracy(labels, test_labels)
    acc = sum(labels == test_labels & test_labels == 1) / sum(test_labels);
end


% Average precision calculation
% calculation of the average precision for a model
% 
% --inputs
% probabilities: classifier èrobabilities
% labels: testing labels
% m: number of true positives
%
% --outputs
% result: calculated average precision

function [result] = average_precision(probabilities, labels, m)
    % calculate the assignments
    assignments = calculate_class_assignments(probabilities, labels);
    
    % calculate the average precision
    total = 0;
    correct_so_far = 0;
    for i=1:numel(assignments)
        if assignments(i) == 1
            correct_so_far = correct_so_far + 1;
            total = total + correct_so_far / i;
        end
    end
    
    result = total / m;
end

%% testing the svm models
test_data_matrix = double([testing_airplanes; testing_cars; testing_faces; testing_motorbikes]);

% airplanes test
load([data_folder, 'models/', kernel_type, '/airplanes_model_', kernel_type, '.mat'], 'airplanes_model');
test_labels = [ones(size(testing_airplanes, 1), 1); 
    zeros(size(testing_cars, 1), 1);
    zeros(size(testing_faces, 1), 1);
    zeros(size(testing_motorbikes, 1), 1)];

[airplanes_labels, airplanes_prob] = predict(airplanes_model, test_data_matrix);
airplanes_ap = average_precision(airplanes_prob(:,2), test_labels, size(testing_airplanes, 1));
airplanes_acc = accuracy(airplanes_labels, test_labels);

% cars test
load([data_folder, 'models/', kernel_type, '/cars_model_', kernel_type, '.mat'], 'cars_model');
test_labels = [zeros(size(testing_airplanes, 1), 1); 
    ones(size(testing_cars, 1), 1);
    zeros(size(testing_faces, 1), 1);
    zeros(size(testing_motorbikes, 1), 1)];

[cars_labels, cars_prob] = predict(cars_model, test_data_matrix);
cars_ap = average_precision(cars_prob(:,2), test_labels, size(testing_cars, 1));
cars_acc = accuracy(cars_labels, test_labels);

% faces test
load([data_folder, 'models/', kernel_type, '/faces_model_', kernel_type, '.mat'], 'faces_model');
test_labels = [zeros(size(testing_airplanes, 1), 1); 
    zeros(size(testing_cars, 1), 1);
    ones(size(testing_faces, 1), 1);
    zeros(size(testing_motorbikes, 1), 1)];

[faces_labels, faces_prob] = predict(faces_model, test_data_matrix);
faces_ap = average_precision(faces_prob(:,2), test_labels, size(testing_faces, 1));
faces_acc = accuracy(faces_labels, test_labels);

% motorbikes test
load([data_folder, 'models/', kernel_type, '/motorbikes_model_', kernel_type, '.mat'], 'motorbikes_model');
test_labels = [zeros(size(testing_airplanes, 1), 1); 
    zeros(size(testing_cars, 1), 1);
    zeros(size(testing_faces, 1), 1);
    ones(size(testing_motorbikes, 1), 1)];

[motorbikes_labels, motorbikes_prob] = predict(motorbikes_model, test_data_matrix);
motorbikes_ap = average_precision(motorbikes_prob(:,2), test_labels, size(testing_motorbikes, 1));
motorbikes_acc = accuracy(motorbikes_labels, test_labels);

%% calculating MAP
mean_average_precision = (airplanes_ap + cars_ap + faces_ap + motorbikes_ap) / 4;
mean_accuracy = (airplanes_acc + cars_acc + faces_acc + motorbikes_acc) / 4;

%% Support functions

function [assignments] = calculate_class_assignments(probabilities, labels)
    [~, indexes] = sort(probabilities, 'descend');
    assignments = labels(indexes);
end

function [acc] = accuracy(labels, test_labels)
    acc = sum(labels == test_labels & test_labels == 1) / sum(test_labels);
end

function [result] = average_precision(probabilities, labels, m)
    assignments = calculate_class_assignments(probabilities, labels);
    
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
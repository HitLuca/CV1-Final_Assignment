%% testing the svm models
test_images = double([testing_airplanes; testing_cars; testing_faces; testing_motorbikes]);
test_data_matrix = sparse(test_images);

% airplanes test
test_labels = [ones([size(testing_airplanes, 1), 1]); 
    zeros([size(testing_cars, 1) 1]);
    zeros([size(testing_faces, 1) 1]);
    zeros([size(testing_motorbikes, 1) 1])];

[~, ~, airplanes_p] = predict(test_labels, test_data_matrix, airplanes_model);
airplanes_ap = average_precision(airplanes_p, test_labels, size(testing_airplanes, 1));

% cars test
test_labels = [zeros([size(testing_airplanes, 1), 1]); 
    ones([size(testing_cars, 1) 1]);
    zeros([size(testing_faces, 1) 1]);
    zeros([size(testing_motorbikes, 1) 1])];

[~, ~, cars_p] = predict(test_labels, test_data_matrix, cars_model);
cars_ap = average_precision(cars_p, test_labels, size(testing_cars, 1));

% faces test
test_labels = [zeros([size(testing_airplanes, 1), 1]); 
    zeros([size(testing_cars, 1) 1]);
    ones([size(testing_faces, 1) 1]);
    zeros([size(testing_motorbikes, 1) 1])];

[~, ~, faces_p] = predict(test_labels, test_data_matrix, faces_model);
faces_ap = average_precision(faces_p, test_labels, size(testing_faces, 1));

% motorbikes test
test_labels = [zeros([size(testing_airplanes, 1), 1]); 
    zeros([size(testing_cars, 1) 1]);
    zeros([size(testing_faces, 1) 1]);
    ones([size(testing_motorbikes, 1) 1])];

[~, ~, motorbikes_p] = predict(test_labels, test_data_matrix, motorbikes_model);
motorbikes_ap = average_precision(motorbikes_p, test_labels, size(testing_motorbikes, 1));


%% calculating MAP
mean_average_precision = (airplanes_ap + cars_ap + faces_ap + motorbikes_ap) / 4;

mean_average_precision


%%
function [assignments] = calculate_class_assignments(probabilities, labels)
      [~, indexes] = sort(probabilities, 'descend');
      assignments = labels(indexes);
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
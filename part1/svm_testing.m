%% testing the svm models
test_images = double([testing_airplanes; testing_cars; testing_faces; testing_motorbikes]);
test_data_matrix = sparse(test_images);

% airplanes test
airplanes_test_labels = [repmat(1,[size(testing_airplanes, 1), 1]); 
    repmat(0,[size(testing_cars, 1) 1]);
    repmat(0,[size(testing_faces, 1) 1]);
    repmat(0,[size(testing_motorbikes, 1) 1])];

[~, ~, airplanes_p] = predict(airplanes_test_labels, test_data_matrix, airplanes_model);

% cars test
cars_test_labels = [repmat(0,[size(testing_airplanes, 1), 1]); 
    repmat(1,[size(testing_cars, 1) 1]);
    repmat(0,[size(testing_faces, 1) 1]);
    repmat(0,[size(testing_motorbikes, 1) 1])];

[~, ~, cars_p] = predict(cars_test_labels, test_data_matrix, cars_model);

% faces test
faces_test_labels = [repmat(0,[size(testing_airplanes, 1), 1]); 
    repmat(0,[size(testing_cars, 1) 1]);
    repmat(1,[size(testing_faces, 1) 1]);
    repmat(0,[size(testing_motorbikes, 1) 1])];

[~, ~, faces_p] = predict(faces_test_labels, test_data_matrix, faces_model);

% motorbikes test
motorbikes_test_labels = [repmat(0,[size(testing_airplanes, 1), 1]); 
    repmat(0,[size(testing_cars, 1) 1]);
    repmat(0,[size(testing_faces, 1) 1]);
    repmat(1,[size(testing_motorbikes, 1) 1])];

[~, ~, motorbikes_p] = predict(motorbikes_test_labels, test_data_matrix, motorbikes_model);


%% calculating the mean average precision
airplanes_ap = average_precision(airplanes_p, airplanes_test_labels, size(testing_airplanes, 1));
cars_ap = average_precision(cars_p, cars_test_labels, size(testing_cars, 1));
faces_ap = average_precision(faces_p, faces_test_labels, size(testing_faces, 1));
motorbikes_ap = average_precision(motorbikes_p, motorbikes_test_labels, size(testing_motorbikes, 1));

mean_average_precision = (airplanes_ap + cars_ap + faces_ap + motorbikes_ap) / 4


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
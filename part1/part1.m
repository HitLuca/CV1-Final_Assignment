% sift_type = 'color';
sift_type = 'grayscale';
clusters_number = 400;

svm_param = ['-t -v 3 -c ', num2str(c), ' -g ', num2str(g)];

data_folder = char(strcat('data/', string(sift_type), '/', string(clusters_number), '/'));

mkdir(strcat(data_folder, 'models'));
mkdir(strcat(data_folder, 'preprocessing'));
mkdir(strcat(data_folder, 'testing_data'));
mkdir(strcat(data_folder, 'training_data'));

disp('PREPROCESSING');
run('preprocessing.m');

disp('TRAINING DATASET CREATION');
run('svm_train_dataset.m');

disp('TRAINING');
run('svm_training.m');

disp('TEST DATASET CREATION');
run('svm_test_dataset.m');

disp('TESTING');
run('svm_testing.m');

% store the MAP, and AP for each image category
ap_data = strcat(num2str(mean_average_precision), ',', num2str(airplanes_ap) ...
    , ',', num2str(cars_ap), ',', num2str(faces_ap), ',', num2str(motorbikes_ap));

log_data = strcat(num2str(clusters_number), ',', sift_type, ',', ap_data);

write_log(data_folder, log_data);

clear;

sift_type = 'RGB';
clusters_number = 400;

preprocessing_images = 100; % number of images to load per class for preprocessing
preprocessing_descriptors = -1; % number of descriptor to extract per image

data_folder = createFolders(sift_type, clusters_number);

disp('PREPROCESSING');
preprocessing;

disp('TRAINING DATASET CREATION');
svm_train_dataset;

disp('TEST DATASET CREATION');
svm_test_dataset;

disp('SVM TRAINING AND TESTING');
for i = {'linear', 'polynomial', 'rbf', 'sigmoid'}
    kernel_type = i{1};
    kernel_param = get_kernel_param(kernel_type);

    svm_training;
    svm_testing;
    disp(strcat(kernel_type, ': ', string(mean_average_precision)));
end


%%
function [data_folder] = createFolders(sift_type, clusters_number)
    warning('off', 'all');
    data_folder = char(strcat('data/', string(sift_type), '/', string(clusters_number), '/'));

    mkdir(strcat(data_folder, 'models'));
    mkdir(strcat(data_folder, 'preprocessing'));
    mkdir(strcat(data_folder, 'testing_data'));
    mkdir(strcat(data_folder, 'training_data'));
    
    mkdir(strcat(data_folder, 'models/linear'));
    mkdir(strcat(data_folder, 'models/polynomial'));
    mkdir(strcat(data_folder, 'models/rbf'));
    mkdir(strcat(data_folder, 'models/sigmoid'));
    warning('on', 'all');
end
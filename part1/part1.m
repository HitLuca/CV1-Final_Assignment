% clear;
sift_type = 'color';
clusters_number = 400;
kernel_type = 'linear';
kernel_param = get_kernel_param(kernel_type);
    
runAlgorithm(sift_type, clusters_number, kernel_param);

function [] = runAlgorithm(sift_type, clusters_number, kernel_param)
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
end

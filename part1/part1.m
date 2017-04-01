%% Main routine
% routine where all the subscripts are called
% the loop tests every combination of parameters and displays the relative
% MAP of the final classifier

% cycle through every sift type

for s = {'grayscale', 'RGB', 'rgb', 'opponent', 'grayscaleDense',  'RGBDense',  'rgbDense',  'opponentDense'}
    sift_type = s{1};
    disp(strcat('--Sift type: ' , string(sift_type)));
    
    % cycle through every vocabulary size
    for c = {400, 800, 1600}%, 2000, 4000}
        clusters_number = c{1};
        disp(strcat('--Vocabulary size: ' , string(clusters_number)));
        
        preprocessing_images = 100; % number of images to load per class for preprocessing
        preprocessing_descriptors = -1; % number of descriptor to extract per image, -1 for all
        
        data_folder = createFolders(sift_type, clusters_number);

        % descriptors extraction and creation of the visual vocabulary
        disp('--Preprocessing');
        preprocessing;

        % creation of the training dataset, without using the first
        % preprocessing_images images
        disp('--Training dataset creation');
        svm_train_dataset;

        %creation of the test dataset, using all the test images
        disp('--Test dataset creation');
        svm_test_dataset;

        % training and testing of the classifiers, cycling over every
        % kernel type
        disp('--SVM training and testing');
        for i = {'linear', 'polynomial', 'rbf', 'sigmoid'}
            kernel_type = i{1};
            kernel_param = get_kernel_param(kernel_type);

            svm_training;
            svm_testing;
            
            %displaying the results
            disp(strcat(kernel_type, ': ', string(mean_average_precision)));
        end
        
        knn_training_testing;
        
        disp(strcat('knn accuracy: ', string(accuracy)));
        fprintf('\n');
    end
    fprintf('\n');
end


%% Support functions

% creation of the folder hierarchy for a specific parameters combination
% under the data/ directory
%
% --inputs
% sift_type: type of sift used
% clusters_number: size of the visual vocabulary
%
% --outputs
% data_folder the path of the root folder for the current run

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
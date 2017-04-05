%% Main routine
% routine where all the subscripts are called
% the loop tests every combination of parameters and displays the relative
% MAP of the final classifier

% cycle through every descriptor type
for d = {'grayscaleLiop', 'RGBLiop', 'rgbLiop', 'opponentLiop', 'grayscaleSift', 'RGBSift', 'rgbSift', 'opponentSift', 'grayscaleDenseSift',  'RGBDenseSift',  'rgbDenseSift',  'opponentDenseSift'}
    descriptor_type = d{1};

    % cycle through every vocabulary size
    for c = {400, 1600} % {400, 800, 1600, 2000}
        clusters_number = c{1};

        if size(strfind(descriptor_type, 'Liop'), 2) > 0
            if clusters_number ~= 400
                break
            else
                clusters_number = 200;
            end
        end
        
        disp(['--Desscriptor type: ' , descriptor_type]);
        disp(['--Vocabulary size: ' , num2str(clusters_number)]);
        
        preprocessing_images = 50; % number of images to load per class for preprocessing
        preprocessing_descriptors = -1; % number of descriptor to extract per image, -1 for all
        
        data_folder = createFolders(descriptor_type, clusters_number);

        % descriptors extraction and creation of the visual vocabulary
%         disp('--Preprocessing');
        preprocessing;

        % creation of the training dataset, without using the first
        % preprocessing_images images
%         disp('--Training dataset creation');
%         svm_train_dataset;

        %creation of the test dataset, using all the test images
%         disp('--Test dataset creation');
        svm_test_dataset;

        % training and testing of the classifiers, cycling over every
        % kernel type
%         disp('--SVM training and testing');
        for i = {'rbf'} %{'linear', 'polynomial', 'rbf'}
            kernel_type = i{1};
            
%             svm_training;
            svm_testing;
            
            %displaying the results
%             disp([kernel_type, ' MAP: ', num2str(mean_average_precision)]);
            disp([kernel_type, ' accuracy: ', num2str(mean_accuracy)]);
        end
        
%         knn_training_testing;
        
%         disp(['knn accuracy: ', num2str(knn_accuracy)]);
        fprintf('\n');
    end
    fprintf('\n');
end


%% Support functions

% creation of the folder hierarchy for a specific parameters combination
% under the data/ directory
%
% --inputs
% descriptor_type: type of descriptor used
% clusters_number: size of the visual vocabulary
%
% --outputs
% data_folder the path of the root folder for the current run

function [data_folder] = createFolders(descriptor_type, clusters_number)
    warning('off', 'all');
    data_folder = char(['data/', descriptor_type, '/', num2str(clusters_number), '/']);

    mkdir([data_folder, 'models']);
    mkdir([data_folder, 'preprocessing']);
    mkdir([data_folder, 'testing_data']);
    mkdir([data_folder, 'training_data']);
    
    mkdir([data_folder, 'models/linear']);
    mkdir([data_folder, 'models/polynomial']);
    mkdir([data_folder, 'models/rbf']);
    mkdir([data_folder, 'models/sigmoid']);
    warning('on', 'all');
end
%% HTML files creation
% routine that created an html file for every combination of parameters and
% stores it into the html folder

% load all the images filenames
filenames = loadFilenames();

for d = {'grayscaleLiop', 'RGBLiop', 'rgbLiop', 'opponentLiop', 'grayscaleSift', 'RGBSift', 'rgbSift', 'opponentSift', 'grayscaleDenseSift',  'RGBDenseSift',  'rgbDenseSift',  'opponentDenseSift'}
    descriptor_type = d{1};
    disp(descriptor_type);
    
    % cycle through every vocabulary size
    for c = {400, 800, 1600, 2000}
        clusters_number = c{1};
        
        if size(strfind(descriptor_type, 'Liop'), 2) > 0
            if clusters_number ~= 400
                break
            else
                clusters_number = 200;
            end
        end
        
        disp(num2str(clusters_number));

        % create the correct html folder
        html_folder = createHtmlFolders(descriptor_type, clusters_number);

        % get the folder to load the models from
        data_folder = getDataFolder(descriptor_type, clusters_number);

        % load the testing dataset
        [testing_airplanes, testing_cars, testing_faces, testing_motorbikes] = loadTestingDataset(data_folder);

        %loop through all the kernel types
        for i = {'linear', 'polynomial', 'rbf'}
            kernel_type = i{1};
                  
            % create the html file
            createHtml(html_folder, data_folder, descriptor_type, clusters_number, kernel_type, filenames, testing_airplanes, testing_cars, testing_faces, testing_motorbikes);
        end
    end
end


%% Support functions

% HTML creation main routine
% this function creates the html file given the parameters value and stores
% it in the html folder
% 
% --inputs
% html_folder: list of filenames
% data_folder: classifier probabilities
% descriptor_type: descriptor type
% clusters_number: vocabulary size
% kernel_type: kernel type
% filenames: list of images filenames
% testing_airplanes: airplanes dataset 
% testing_cars: cars dataset 
% testing_faces: faces dataset 
% testing_motorbikes: motorbikes dataset 
%
% --outputs

function [] = createHtml(html_folder, data_folder, descriptor_type, clusters_number, kernel_type, filenames, testing_airplanes, testing_cars, testing_faces, testing_motorbikes)
    % test the models
    svm_testing();
    
    % create the html filename
    filename = [descriptor_type, '_', num2str(clusters_number), '_', kernel_type, '.html'];
    
    % open the file
    fid = fopen([html_folder, '/', filename],'wt');
    
    % metadata
    html = strcat('<!DOCTYPE html> <html lang="en">',...
        '<head> <meta charset="utf-8">',...
        '<title>Image list prediction</title>',...
        '<style type="text/css">',...
        'img { width:200px; }',...
        '</style> </head> <body>',...
        '<h2>Luca Simonetto, Heng Lin</h2>',...
        '<h1>Settings</h1> <table>');
    
    % put dense information if needed
    if strfind(descriptor_type, 'Dense')
        html = strcat(html, ...
            '<tr><th>SIFT step size</th><td>', num2str(11), ' px</td></tr>',...
            '<tr><th>SIFT block sizes</th><td>', num2str(10),' pixels</td></tr>');
    end
    
    % get the string to display for the descriptor type
    descriptor_name = getDescriptorName(descriptor_type);
    
    % put the training informations
    if size(strfind(descriptor_type, 'Liop'), 2) > 0
        html = strcat(html, '<tr><th>LIOP method</th><td>', descriptor_name, '</td></tr>');
        html = strcat(html,...
            '<tr><th>Vocabulary size</th><td>200 words</td></tr>',...
            '<tr><th>Vocabulary fraction</th><td> 10.72 %% </td></tr>',...
            '<tr><th>SVM training data</th><td>450 positive, 1350 negative per class</td></tr>');
    else
        html = strcat(html, '<tr><th>SIFT method</th><td>', descriptor_name, '</td></tr>');
        html = strcat(html,...
            '<tr><th>Vocabulary size</th><td>400 words</td></tr>',...
            '<tr><th>Vocabulary fraction</th><td> 21.45 %% </td></tr>',...
            '<tr><th>SVM training data</th><td>400 positive, 1200 negative per class</td></tr>');
    end
    
    % scores
    html = strcat(html,...
        '<tr><th>SVM kernel type</th><td>', kernel_type, '</td></tr>',...
        '</table>',...
        '<h1>Prediction lists (MAP: ', num2str(mean_average_precision), ')</h1>',...
        '<table> <thead> <tr>',...
        '<th>Airplanes (AP: ', num2str(airplanes_ap), ')</th>',...
        '<th>Cars (AP: ', num2str(cars_ap), ')</th>',...
        '<th>Faces (AP: ', num2str(faces_ap), ')</th>',...
        '<th>Motorbikes (AP: ', num2str(motorbikes_ap), ')</th>',...
        '</tr> </thead> <tbody>');
        
    image_root = '../../../../Caltech4/ImageData/';
    
    % order the filenames based on the SVM scores
    airplanes_ordered_filenames = order_filenames(filenames, airplanes_prob(:,2));
    cars_ordered_filenames = order_filenames(filenames, cars_prob(:,2));
   	faces_ordered_filenames = order_filenames(filenames, faces_prob(:,2));
    motorbikes_ordered_filenames = order_filenames(filenames, motorbikes_prob(:,2));
    
    % add the images list
    for i=1:200
        html = strcat(html,...
        '<tr><td><img src="', image_root, airplanes_ordered_filenames{i}, '" /></td>',...
        '<td><img src="', image_root, cars_ordered_filenames{i}, '" /></td>',...
        '<td><img src="', image_root, faces_ordered_filenames{i}, '" /></td>',...
        '<td><img src="', image_root, motorbikes_ordered_filenames{i}, '" /></td></tr>');
    end
    
    % write to file
    fprintf(fid, html);
    
    % close the file
    fclose(fid);
end


function [data_folder] = createHtmlFolders(descriptor_type, clusters_number)
    warning('off', 'all');
    data_folder = char(strcat('html/', string(descriptor_type), '/', string(clusters_number), '/'));

    mkdir(data_folder);
    warning('on', 'all');
end

function [data_folder] = getDataFolder(descriptor_type, clusters_number)
    data_folder = char(['data/', descriptor_type, '/', num2str(clusters_number), '/']);
end

function [descriptor_name] = getDescriptorName(descriptor_type)
    if strcmp(descriptor_type, 'grayscaleSift')
        descriptor_name = 'intensity-SIFT';
    elseif strcmp(descriptor_type, 'grayscaleDenseSift')
    	descriptor_name = 'intensity-SIFT dense';
    elseif strcmp(descriptor_type, 'RGBSift')
    	descriptor_name = 'RGB-SIFT';
    elseif strcmp(descriptor_type, 'RGBDenseSift')
    	descriptor_name = 'RGB-SIFT dense';
    elseif strcmp(descriptor_type, 'rgbSift')
    	descriptor_name = 'rgb-SIFT';
    elseif strcmp(descriptor_type, 'rgbDenseSift')
    	descriptor_name = 'rgb-SIFT dense';
    elseif strcmp(descriptor_type, 'opponentSift')
    	descriptor_name = 'opponent-SIFT';
    elseif strcmp(descriptor_type, 'opponentDenseSift')
    	descriptor_name = 'opponent-SIFT dense';
    elseif strcmp(descriptor_type, 'grayscaleLiop')
        descriptor_name = 'intensity-LIOP';
    elseif strcmp(descriptor_type, 'RGBLiop')
    	descriptor_name = 'RGB-LIOP';
    elseif strcmp(descriptor_type, 'rgbLiop')
    	descriptor_name = 'rgb-LIOP';
    elseif strcmp(descriptor_type, 'opponentLiop')
    	descriptor_name = 'opponent-LIOP';
    end
end

function [testing_airplanes, testing_cars, testing_faces, testing_motorbikes] = loadTestingDataset(data_folder)
    testing_airplanes_path = [data_folder, 'testing_data/testing_airplanes.mat'];
    testing_cars_path = [data_folder, 'testing_data/testing_cars.mat'];
    testing_faces_path = [data_folder, 'testing_data/testing_faces.mat'];
    testing_motorbikes_path = [data_folder, 'testing_data/testing_motorbikes.mat'];
    load(testing_airplanes_path, 'testing_airplanes');
    load(testing_cars_path, 'testing_cars');
    load(testing_faces_path, 'testing_faces');
    load(testing_motorbikes_path, 'testing_motorbikes');
end


% Images filenames extraction
% this function extracts and returns a list containing the filenames for
% every image in the testing set
% 
% --inputs
%
% --outputs
% filenames: vector with all the filenames

function [filenames] = loadFilenames()  
    airplanes_filenames = {};
    cars_filenames = {};
    faces_filenames = {};
    motorbikes_filenames = {};
    
    dataset_dir = '../Caltech4/ImageData/';
    contents = dir(dataset_dir); % all the image folders

    % loop over all the folders
    for i = 1:numel(contents)
        foldername = contents(i).name;
        
        % enter only in the test folders
        if contains(foldername, 'test')
            folder_contents = dir(strcat(dataset_dir, foldername, '/*.jpg'));

             % loop over all the files in each folder
            for j=1:numel(folder_contents)
                filename = folder_contents(j).name;
                
                % concatenate the full path to the filename
                filename = [foldername, '/', filename];
                
                % put it into the correct matrix
                if contains(foldername, 'airplanes')
                    airplanes_filenames = [airplanes_filenames; filename];
                elseif contains(foldername, 'cars')
                    cars_filenames = [cars_filenames; filename];
                elseif contains(foldername, 'faces')
                    faces_filenames = [faces_filenames; filename];
                elseif contains(foldername, 'motorbikes')
                    motorbikes_filenames = [motorbikes_filenames; filename];
                end
            end
        end
    end
    
    % concatenate the results
    filenames = [airplanes_filenames; cars_filenames; faces_filenames; motorbikes_filenames];
end


% Filenames ordering
% this function reorders the input filenames based on the output
% probabilities of the classifier
% 
% --inputs
% filenames: list of filenames
% probabilities: classifier probabilities
%
% --outputs
% ordered_filenames: vector with all the filenames reordered

function [ordered_filenames] = order_filenames(filenames, probabilities)
    % sort and return the filenames
    [~, indexes] = sort(probabilities, 'descend');
    ordered_filenames = filenames(indexes);
end
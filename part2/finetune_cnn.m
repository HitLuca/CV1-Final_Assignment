function [net, info, expdir] = finetune_cnn(varargin)

%% Define options
run('/home/henglin/matlab_R2016b_glnxa64/matconvnet-1.0-beta23/matlab/vl_setupnn.m');

opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin);     % update parameter-value pairs

opts.expDir = fullfile('data', ...
  sprintf('cnn_assignment-%s', opts.modelType));
[opts, varargin] = vl_argparse(opts, varargin);

opts.dataDir = './data/' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb-caltech.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

%gpu training is not installed
%opts.train.gpus = [1];

%% update model

net = update_model();

%% TODO: Implement getCaltechIMDB function below

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getCaltechIMDB() ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end


%%
net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

trainfn = @cnn_train ;
[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 2)) ;

expdir = opts.expDir;
end

%% GetBatch

function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

end

%% GetSimpleNNBatch

function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% call the data augmentation function
images = data_augmentation(images);

end

%% Data Augmentation Function

function [images] = data_augmentation(images)
    if rand > 0.5
%         angle = randi(10);             % random degree of rotation
%         images = imrotate(images, angle, 'nearest', 'crop');   % rotate the images
%         images = fliplr(images);                  % flip the image
        images = imnoise(images, 'gaussian', 0, 0.1);      % add gaussian noise to images
    end
end


%% GetCaltechIMDB

function imdb = getCaltechIMDB()

% Prepare the imdb structure, returns image data with mean image subtracted
classes = {'airplanes', 'cars', 'faces', 'motorbikes'};
splits = {'train', 'test'};

%% TODO: Implement your loop here, to create the data structure described in the assignment
% Initialize the placeholders for the data
data = zeros(32, 32, 3, 2079);
sets = [];
labels = [];
k=1;

% relevant file paths
dataset_dir = './../Caltech4/ImageData/';
train_filename = './../Caltech4/ImageSets/train.txt';
test_filename = './../Caltech4/ImageSets/test.txt';

contents = dir(dataset_dir); % all the image folders
train_filenames = textscan(fopen(train_filename), '%s'); % training file contents
train_filenames = train_filenames{1};
% test_filenames = textscan(fopen(test_filename), '%s'); % test file contents
% test_filenames = test_filenames{1};

% loop over all the folders
for i = 1:numel(contents)
    foldername = contents(i).name;
    folder_contents = dir(strcat(dataset_dir, foldername, '/*.jpg'));
    
    % Assign labels based on folder name
    if strmatch('airplanes', foldername)
        label = 1;
    elseif strmatch('cars', foldername)
        label = 2;
    elseif strmatch('faces', foldername)
        label = 3;
    elseif strmatch('motorbikes', foldername)
        label = 4;
    end
    
    % loop over all the files in each folder
    for j=1:numel(folder_contents)
        filename = folder_contents(j).name;
        image = imread(strcat(dataset_dir, foldername, '/', filename));
        
        % checking for grayscale images, ignore if identified
        if size(image, 3) == 1
            continue
        end
        
        % resize the images to satisfy the network input size
        image = imresize(image, [32, 32]);
        
        % match the foldername+filename with the train list
        string_to_match = strcat(foldername, '/', strtok(filename, '.'));
        if strmatch(string_to_match, train_filenames)
            set = 1;
        else
            set = 2;
        end
        data(:,:,:,k) = image;
        sets(k) = set;
        labels(k) = label;
        k = k + 1;
    end
end

sets = single(sets);

%%
% subtract mean
dataMean = mean(data(:, :, :, sets == 1), 4);
data = bsxfun(@minus, data, dataMean);
data = single(data);
imdb.images.data = data;
imdb.images.labels = single(labels) ;
imdb.images.set = sets;
imdb.meta.sets = {'train', 'val'} ;
imdb.meta.classes = classes;

perm = randperm(numel(imdb.images.labels));
imdb.images.data = imdb.images.data(:,:,:, perm);
imdb.images.labels = imdb.images.labels(perm);
imdb.images.set = imdb.images.set(perm);


end

function train_svm(nets, data)

%% replace loss with the classification as we will extract features
nets.pre_trained.layers{end}.type = 'softmax';
nets.fine_tuned.layers{end}.type = 'softmax';

%% extract features and train SVM classifiers, by validating their hyperparameters
[svm.pre_trained.trainset, svm.pre_trained.testset] = get_svm_data(data, nets.pre_trained);
[svm.fine_tuned.trainset,  svm.fine_tuned.testset] = get_svm_data(data, nets.fine_tuned);


%% measure the accuracy of different settings
[nn.accuracy] = get_nn_accuracy(nets.fine_tuned, data);
[svm.pre_trained.predictions, svm.pre_trained.accuracy] = get_predictions(svm.pre_trained);
[svm.fine_tuned.predictions, svm.fine_tuned.accuracy] = get_predictions(svm.fine_tuned);

fprintf('\n\n\n\n\n\n\n\n');

fprintf('CNN: fine_tuned_accuracy: %0.2f, SVM: pre_trained_accuracy: %0.2f, fine_tuned_accuracy: %0.2f\n', nn.accuracy, svm.pre_trained.accuracy(1), svm.fine_tuned.accuracy(1));

% Plot the feature spaces
plotFeatureSpace(svm);

end


%% Function to plot the feature space

function plotFeatureSpace(svm)

    % plot feature space for training set
    figure(2);
    title('Training set features')
    
    % use T-SNE to extract features
    pre_trained = tsne(svm.pre_trained.trainset.features, svm.pre_trained.trainset.labels, 2, 3, 30);
    fine_tuned = tsne(svm.fine_tuned.trainset.features, svm.fine_tuned.trainset.labels, 2, 3, 30);

    % plot the feature space for trainint set
    subplot(1, 2, 1);
    gscatter(pre_trained(:,1), pre_trained(:,2), svm.pre_trained.trainset.labels);
    subplot(1, 2, 2);
    gscatter(fine_tuned(:,1), fine_tuned(:,2), svm.fine_tuned.trainset.labels);
    
    % plot feature space for testing set
    figure(3);
    title('Testing set features')
    
    % use T-SNE to extract features
    pre_trained = tsne(svm.pre_trained.testset.features, svm.pre_trained.testset.labels, 2, 3, 30);
    fine_tuned = tsne(svm.fine_tuned.testset.features, svm.fine_tuned.testset.labels, 2, 3, 30);

    % plot the feature space for testing set
    subplot(1, 2, 1);
    gscatter(pre_trained(:,1), pre_trained(:,2), svm.pre_trained.testset.labels);
    subplot(1, 2, 2);
    gscatter(fine_tuned(:,1), fine_tuned(:,2), svm.fine_tuned.testset.labels);
    
end


%% GetNNAccuracy

function [accuracy] = get_nn_accuracy(net, data)

counter = 0;
for i = 1:size(data.images.data, 4)    
    if(data.images.set(i)==2)    
        res = vl_simplenn(net, data.images.data(:, :,:, i));
        [~, estimclass] = max(res(end).x);
        if(estimclass == data.images.labels(i))
            counter = counter+1;
        end
    end
end

accuracy = counter / nnz(data.images.set==2);
end

%% GetPredictions

function [predictions, accuracy] = get_predictions(data)

best = train(data.trainset.labels, data.trainset.features, '-C -s 0');
model = train(data.trainset.labels, data.trainset.features, sprintf('-c %f -s 0', best(1))); % use the same solver: -s 0
[predictions, accuracy, ~] = predict(data.testset.labels, data.testset.features, model);


end

%% GetSVMData

function [trainset, testset] = get_svm_data(data, net)

trainset.labels = [];
trainset.features = [];

testset.labels = [];
testset.features = [];
for i = 1:size(data.images.data, 4)
    
    res = vl_simplenn(net, data.images.data(:, :,:, i));
    feat = res(end-3).x; feat = squeeze(feat);
    
    if(data.images.set(i) == 1)
        
        trainset.features = [trainset.features feat];
        trainset.labels   = [trainset.labels;  data.images.labels(i)];
        
    else
        
        testset.features = [testset.features feat];
        testset.labels   = [testset.labels;  data.images.labels(i)];
        
        
    end
    
end

trainset.labels = double(trainset.labels);
trainset.features = sparse(double(trainset.features'));

testset.labels = double(testset.labels);
testset.features = sparse(double(testset.features'));

end

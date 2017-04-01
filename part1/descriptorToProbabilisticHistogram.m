function [hist_bins] = descriptorToProbabilisticHistogram(clusters_number, clusters, descriptors)
    k = 5;

    descriptors = double(descriptors');
    
    d_number = size(descriptors, 1); % number of descriptors
    hist_bins = zeros(1, clusters_number); % empty histogram
    
    % calculating the index of the closest cluster for every descriptor
    [indexes, distances] = knnsearch(clusters, descriptors, 'K', k);
    
    for i=1:size(indexes, 1)
        hist_bins(indexes(i, :)) = hist_bins(indexes(i,:)) + distances(i,:)./sum(distances(i,:));
    end
    
    % histogram normalization using the number of image descriptors
    hist_bins = hist_bins ./ d_number;
end
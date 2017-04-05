function [hist_bins] = descriptorToProbabilisticHistogram(clusters_number, clusters, descriptors)
    % number of nearest clusters to consider
    k = 5;

    descriptors = double(descriptors');
    
    % empty histogram
    hist_bins = zeros(1, clusters_number);
    
    % calculating the index of the closest cluster for every descriptor
    [indexes, distances] = knnsearch(clusters, descriptors, 'K', k);

    % calculating the inverse of the distances
    distances = 1 ./ distances;
    
    % normalizing them
    distances = distances ./ sum(distances, 2);
        
    % summing all the contributions
    for i=1:size(indexes, 1)
        hist_bins(indexes(i, :)) = hist_bins(indexes(i,:)) + distances(i, :);
    end
    
    % histogram normalization
    hist_bins = hist_bins ./ sum(hist_bins);
end
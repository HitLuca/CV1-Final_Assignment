% Descriptors to probabilistic histogram conversion
% this function converts the descriptors of an image to a normalized probabilistic histogram of
% clusters_number size
% 
% --inputs
% clusters_number: visual vocabulary size
% clusters: matrix containing the clusters coordinates
% descriptors: descriptors of the current image
%
% --outputs
% histogram: normalized histogram

function [histogram] = descriptorToProbabilisticHistogram(clusters_number, clusters, descriptors)
    % number of nearest clusters to consider
    k = 5;

    descriptors = double(descriptors');
    
    % empty histogram
    histogram = zeros(1, clusters_number);
    
    % calculating the index of the closest cluster for every descriptor
    [indexes, distances] = knnsearch(clusters, descriptors, 'K', k);

    % calculating the inverse of the distances
    distances = 1 ./ distances;
    
    % normalizing them
    distances = distances ./ sum(distances, 2);
        
    % summing all the contributions
    for i=1:size(indexes, 1)
        histogram(indexes(i, :)) = histogram(indexes(i,:)) + distances(i, :);
    end
    
    % histogram normalization
    histogram = histogram ./ sum(histogram);
end
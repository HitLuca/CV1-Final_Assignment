% Descriptors to histogram conversion
% this function converts the descriptors of an image to a normalized histogram of
% clusters_number size
% 
% --inputs
% clusters_number: visual vocabulary size
% clusters: matrix containing the clusters coordinates
% descriptors: descriptors of the current image
%
% --outputs
% histogram: normalized histogram

function [histogram] = descriptorToHistogram(clusters_number, clusters, descriptors)
    descriptors = double(descriptors');
    
    % number of descriptors
    d_number = size(descriptors, 1);
    
    % calculating the index of the closest cluster for every descriptor
    [indexes, ~] = knnsearch(clusters, descriptors);
    
    % histogram creation and normalization using the number of image descriptors
    histogram = hist(indexes, clusters_number) ./ d_number;
end
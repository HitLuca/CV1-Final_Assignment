% this function converts the descriptors of an image to a normalized histogram of
% clusters_number size
% 
% --inputs
% clusters_number: visual vocabulary size
% clusters: matrix containing the clusters coordinates
% descriptors: descriptors of the current image
%
% --outputs
% hist_bins: calculated histogram

function [hist_bins] = descriptorToHistogram(clusters_number, clusters, descriptors)
    descriptors = double(descriptors');
    
    d_number = size(descriptors, 1); % number of descriptors
    hist_bins = zeros(1, clusters_number); % empty histogram
    
    % calculating the index of the closest cluster for every descriptor
    [indexes, ~] = knnsearch(clusters, descriptors);
    
    % histogram creation and normalization using the number of image descriptors
    hist_bins = hist(indexes, clusters_number) ./ d_number;
end
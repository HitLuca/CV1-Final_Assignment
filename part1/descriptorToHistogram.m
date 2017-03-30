function [hist_bins] = descriptorToHistogram(cluster_number, clusters, descriptors)
    descriptors = double(descriptors');
    
    d_number = size(descriptors, 1); 
    hist_bins = zeros(1, cluster_number);
    
    [indexes, ~] = knnsearch(clusters, descriptors);
    
    hist_bins(indexes) = hist_bins(indexes) + 1;
    hist_bins = hist_bins ./ d_number;
end
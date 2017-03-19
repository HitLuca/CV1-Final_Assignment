function [hist_bins] = descriptorToHistogram(cluster_number, clusters, descriptors)
    descriptors = double(descriptors');
    d_number = size(descriptors, 1); 
    
    hist_bins = zeros(1, cluster_number);
    for i=1:d_number
        descriptor = descriptors(i, :);
        out = pdist2(clusters,descriptor);
        [~, index] = min(out);
        
        hist_bins(index) = hist_bins(index) + 1;
    end
    hist_bins = hist_bins / d_number;
end
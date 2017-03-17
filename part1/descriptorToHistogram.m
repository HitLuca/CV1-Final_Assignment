function [hist_bins] = descriptorToHistogram(clusters_number, clusters, descriptors)
    descriptors = double(descriptors');
    d_number = size(descriptors, 1); 
    
    hist_bins = zeros(1, clusters_number);
    for i=1:d_number
        descriptor = descriptors(i, :);
        out = pdist2(clusters,descriptor);
        [~, index] = min(out);
        
        hist_bins(index) = hist_bins(index) + 1;
    end
end
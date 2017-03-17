function [ f, d ] = sift( type, image )
    if strmatch(type, 'grayscale')
        [f, d] = grayscaleSift(image);
    elseif strmatch(type, 'grayscaleDense')
    	[f, d] = grayscaleDenseSift(image);
    end
end

function [f, d] = grayscaleSift(image)
    [f, d] = vl_sift(image);
end

function [f, d] = grayscaleDenseSift(image)
    step = 1;
    size = 8;
    magnif = 3 ;

    % pre-smoothing image (dsift doesn't do that)
    smoothed = vl_imsmooth(image, sqrt((size/magnif)^2 - .25));

    [f, d] = vl_dsift(smoothed, 'size', size, 'step', step);
    f(3,:) = size/magnif ;
    f(4,:) = 0 ;
end
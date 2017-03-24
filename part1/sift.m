function [ f, d ] = sift( type, image )
    if strmatch(type, 'grayscale')
        [f, d] = grayscaleSift(image);
    elseif strmatch(type, 'grayscaleDense')
    	[f, d] = grayscaleDenseSift(image);
    elseif strmatch(type, 'color')
    	[f, d] = colorSift(image);
    end
end

function [f, d] = grayscaleSift(image)
    if size(image, 3) > 1
        image = single(rgb2gray(image));
    else
        image = single(image);
    end
    [f, d] = vl_sift(image);
end

function [f, d] = grayscaleDenseSift(image)
    if size(image, 3) > 1
        image = single(rgb2gray(image));
    else
        image = single(image);
    end
    step = 1;
    s = 8;
    magnif = 3 ;

    % pre-smoothing image (dsift doesn't do that)
    smoothed = vl_imsmooth(image, sqrt((s/magnif)^2 - .25));

    [f, d] = vl_dsift(smoothed, 'size', s, 'step', step);
    f(3,:) = s/magnif ;
    f(4,:) = 0 ;
end

function [f, d] = colorSift(image)
    image = single(image);
    [f1, d1] = vl_sift(image(:,:,1));
    
    if size(image, 3) > 1
        [f2, d2] = vl_sift(image(:,:,2));
        [f3, d3] = vl_sift(image(:,:,3));
        f = [f1, f2, f3];
        d = [d1, d2, d3];
    else
        f = f1;
        d = d1;
    end
end
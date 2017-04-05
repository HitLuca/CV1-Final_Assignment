%% Sift/Liop descriptors calculation
% wrapper used to choose the correct sift/liop descriptors to compute
% 
% --inputs
% type: sift/liop type
% image: input image
% descriptors_per_image: number of descriptors to output per image. If -1
% or the sift type is dense, no changes are made
% 
% --outputs
% f: sift frames
% d: sift/liop descriptors

function [ f, d ] = getDescriptors(descriptor_type, image, descriptors_per_image)    
    % create a grayscale version of the image
    % workaround as some images are not rgb
    if size(image, 3) > 1
        grayscale_image = single(rgb2gray(image));
    else
        grayscale_image = single(image);
    end
    
    % convert the image to single precision
    image = single(image);
    
    % determine the sift version to use
    if strcmp(descriptor_type, 'grayscaleSift')
        [f, d] = grayscaleSift(grayscale_image);
    elseif strcmp(descriptor_type, 'grayscaleDenseSift')
    	[f, d] = grayscaleDenseSift(grayscale_image);  
    elseif strcmp(descriptor_type, 'RGBSift')
    	[f, d] = RGBSift(image, grayscale_image);
    elseif strcmp(descriptor_type, 'RGBDenseSift')
    	[f, d] = RGBDenseSift(image, grayscale_image);
    elseif strcmp(descriptor_type, 'rgbSift')
    	[f, d] = rgbSift(image, grayscale_image);
    elseif strcmp(descriptor_type, 'rgbDenseSift')
    	[f, d] = rgbDenseSift(image, grayscale_image);
    elseif strcmp(descriptor_type, 'opponentSift')
    	[f, d] = opponentSift(image, grayscale_image);
    elseif strcmp(descriptor_type, 'opponentDenseSift')
    	[f, d] = opponentDenseSift(image, grayscale_image);
    elseif strcmp(descriptor_type, 'grayscaleLiop')
        [f, d] = grayscaleLiop(grayscale_image);
    elseif strcmp(descriptor_type, 'RGBLiop')
    	[f, d] = RGBLiop(image);
    elseif strcmp(descriptor_type, 'rgbLiop')
    	[f, d] = rgbLiop(image);
    elseif strcmp(descriptor_type, 'opponentLiop')
    	[f, d] = opponentLiop(image);
    end
    
    % trim the outputs in order to get descriptors_per_image number of
    % descriptors
    if descriptors_per_image ~= -1 && strfind(type, 'Dense') == 0
        [f, d] = trimElements(f, d, descriptors_per_image);
    end
end


%% Support functions

% reduce the number of total descriptors in order to save memory if needed
%
% --inputs
% f: sift frames
% d: sift descriptors
% descriptors_per_image: number of descriptors to keep
%
% --outputs
% f: trimmed sift frames
% d: trimmed sift descriptors

function [f, d] = trimElements(f, d, descriptors_per_image)
    if size(d, 2) > descriptors_per_image
        elements_number = size(d, 2);
        permutation = randperm(elements_number);

        f = f(:, permutation);
        d = d(:, permutation);

        if size(d, 1) == 128
            f = f(:, 1:descriptors_per_image);
            d = d(:, 1:descriptors_per_image);
        elseif size(d, 1) == 384
            f = f(:, 1:int8(descriptors_per_image/3));
            d = d(:, 1:int8(descriptors_per_image/3));
        end
    end
end


% convert the input image to normalized rgb color space
%
% --inputs:
% image: input image
%
% --outputs: 
% r: normalized r 
% g: normalized g
% b: normalized b

function [r, g, b] = convertTorgb(image)
    %extract each channel
    R  = image(:,:,1);
    G  = image(:,:,2);
    B  = image(:,:,3);
    
    den = (R+G+B);
    
    %convert to rgb space
    r = R ./den;
    g = G ./den;
    b = B ./den;
    
    r(isnan(r)) = 0;
    g(isnan(g)) = 0;
    b(isnan(b)) = 0;
end


% convert the input image to opponent color space
%
% --inputs:
% image: input image
%
% --outputs: 
% o1: first opponent channel
% o2: second opponent channel
% o3: third opponent channel

function [o1, o2, o3] = convertToOpponent(image)
    %extract each channel
    R  = image(:,:,1);
    G  = image(:,:,2);
    B  = image(:,:,3);

    %convert to opponent space
    o1 = (R-G)./sqrt(2);
    o2 = (R+G-2*B)./sqrt(6);
    o3 = (R+G+B)./sqrt(3);
end


%% Sift descriptors extraction

% each function extracts different types of descriptors from the image
%
% --inputs
% image: input image
% grayscale_image: grayscale version of the input image
%
% --outputs
% f: sift frames
% d: sift/liop descriptors

function [f, d] = grayscaleSift(grayscale_image)
    [f, d] = vl_sift(grayscale_image);
end

function [f, d] = grayscaleDenseSift(grayscale_image)
    sift_step = 11; % step size
    sift_size = 10; % window size around descriptor
    
    [f, d] = vl_dsift(grayscale_image, 'step', sift_step, 'size', sift_size, 'fast');
end

function [f, d] = RGBSift(image, grayscale_image)
    % calculate the sift descriptors using the grayscale image in order to
    % get the frames locations
    [f_gray, d_gray] = vl_sift(grayscale_image);
    
    if size(image, 3) > 1
        % calculate sift descriptors at f_gray coordinates
        [f1, d1] = vl_sift(image(:,:,1), 'frames', f_gray);
        [f2, d2] = vl_sift(image(:,:,2), 'frames', f_gray);
        [f3, d3] = vl_sift(image(:,:,3), 'frames', f_gray);
        
        % concatenate the results
        f = [f1;f2;f3];
        d = [d1;d2;d3];
    else
        % if the image is grayscale concatenate the same descriptors
        f = [f_gray; f_gray; f_gray];
        d = [d_gray; d_gray; d_gray];
    end
end

function [f, d] = RGBDenseSift(image, grayscale_image)
    sift_step = 11; % step size
    sift_size = 10; % window size around descriptor
    
    if size(image, 3) > 1
        % calculate the dense sift descriptors for each channel
        [f1, d1] = vl_dsift(image(:,:,1), 'step', sift_step, 'size', sift_size, 'fast');
        [f2, d2] = vl_dsift(image(:,:,2), 'step', sift_step, 'size', sift_size, 'fast');
        [f3, d3] = vl_dsift(image(:,:,3), 'step', sift_step, 'size', sift_size, 'fast');
        
        % concatenate the results
        f = [f1;f2;f3];
        d = [d1;d2;d3];
    else
        % if the image is grayscale
        % calculate the dense sift descriptors
        [f_gray, d_gray] = vl_dsift(grayscale_image, 'step', sift_step, 'size', sift_size, 'fast');
        
        % concatenate the same descriptors
        f = [f_gray; f_gray; f_gray];
        d = [d_gray; d_gray; d_gray]; 
    end
end

function [f, d] = rgbSift(image, grayscale_image)
    % calculate the sift descriptors using the grayscale image in order to
    % get the frames locations
    [f_gray, d_gray] = vl_sift(grayscale_image);
    
    if size(image, 3) > 1
        % convert the image to rgb color space
        [r, g, b] = convertTorgb(image);

        % calculate the dense sift descriptors for each channel
        [f1, d1] = vl_sift(r, 'frames', f_gray);
        [f2, d2] = vl_sift(g, 'frames', f_gray);
        [f3, d3] = vl_sift(b, 'frames', f_gray);
        
        % concatenate the results
        f = [f1;f2;f3];
        d = [d1;d2;d3];
    else
        % if the image is grayscale concatenate the same descriptors
        f = [f_gray; f_gray; f_gray];
        d = [d_gray; d_gray; d_gray];
    end
end

function [f, d] = rgbDenseSift(image, grayscale_image)
    sift_step = 11; % step size
    sift_size = 10; % window size around descriptor
    
    if size(image, 3) > 1
        % convert the image to rgb color space
        [r, g, b] = convertTorgb(image);
        
        % calculate the dense sift descriptors for each channel
        [f1, d1] = vl_dsift(r, 'step', sift_step, 'size', sift_size, 'fast');
        [f2, d2] = vl_dsift(g, 'step', sift_step, 'size', sift_size, 'fast');
        [f3, d3] = vl_dsift(b, 'step', sift_step, 'size', sift_size, 'fast');
        
        % concatenate the results
        f = [f1;f2;f3];
        d = [d1;d2;d3];
    else
        % if the image is grayscale
        % calculate the dense sift descriptors
        [f_gray, d_gray] = vl_dsift(grayscale_image, 'step', sift_step, 'size', sift_size, 'fast');
        
        % concatenate the same descriptors
        f = [f_gray; f_gray; f_gray];
        d = [d_gray; d_gray; d_gray]; 
    end
end

function [f, d] = opponentSift(image, grayscale_image)
    % calculate the sift descriptors using the grayscale image in order to
    % get the frames locations
    [f_gray, d_gray] = vl_sift(grayscale_image);
    
    if size(image, 3) > 1
        % convert the image to opponent color space
        [o1, o2, o3] = convertToOpponent(image);

        % calculate the dense sift descriptors for each channel
        [f1, d1] = vl_sift(o1, 'frames', f_gray);
        [f2, d2] = vl_sift(o2, 'frames', f_gray);
        [f3, d3] = vl_sift(o3, 'frames', f_gray);
        
        % concatenate the results
        f = [f1;f2;f3];
        d = [d1;d2;d3];
    else
        % if the image is grayscale concatenate the same descriptors
        f = [f_gray; f_gray; f_gray];
        d = [d_gray; d_gray; d_gray];
    end
end

function [f, d] = opponentDenseSift(image, grayscale_image)
    sift_step = 11; % step size
    sift_size = 10; % window size around descriptor
    
    if size(image, 3) > 1
        % convert the image to opponent color space
        [o1, o2, o3] = convertToOpponent(image);
        
        % calculate the dense sift descriptors for each channel
        [f1, d1] = vl_dsift(o1, 'step', sift_step, 'size', sift_size, 'fast');
        [f2, d2] = vl_dsift(o2, 'step', sift_step, 'size', sift_size, 'fast');
        [f3, d3] = vl_dsift(o3, 'step', sift_step, 'size', sift_size, 'fast');
        
        % concatenate the results
        f = [f1;f2;f3];
        d = [d1;d2;d3];
    else
        % if the image is grayscale
        % calculate the dense sift descriptors
        [f_gray, d_gray] = vl_dsift(grayscale_image, 'step', sift_step, 'size', sift_size, 'fast');
        
        % concatenate the same descriptors
        f = [f_gray; f_gray; f_gray];
        d = [d_gray; d_gray; d_gray]; 
    end
end

function [f, d] = grayscaleLiop(grayscale_image)
    grayscale_image = imresize(grayscale_image, [299, 299]);
    f = NaN;
    d = vl_liop(grayscale_image);
end

function [f, d] = RGBLiop(image)
    image = imresize(image, [299, 299]);
    f = NaN;
    
    if size(image, 3) > 1
        d1 = vl_liop(image(:,:,1));
        d2 = vl_liop(image(:,:,2));
        d3 = vl_liop(image(:,:,3));
        
        % concatenate the results
        d = [d1;d2;d3];
    else
        d_gray = vl_liop(image);
        d = [d_gray; d_gray; d_gray];
    end
end

function [f, d] = rgbLiop(image)
    image = imresize(image, [299, 299]);
    f = NaN;
    
    if size(image, 3) > 1
        [r, g, b] = convertTorgb(image);
        
        d1 = vl_liop(r);
        d2 = vl_liop(g);
        d3 = vl_liop(b);
        
        % concatenate the results
        d = [d1;d2;d3];
    else
        d_gray = vl_liop(image);
        d = [d_gray; d_gray; d_gray];
    end
end

function [f, d] = opponentLiop(image)
    image = imresize(image, [299, 299]);
    f = NaN;
    
    if size(image, 3) > 1
        [o1, o2, o3] = convertToOpponent(image);
        
        d1 = vl_liop(o1);
        d2 = vl_liop(o2);
        d3 = vl_liop(o3);
        
        % concatenate the results
        d = [d1;d2;d3];
    else
        d_gray = vl_liop(image);
        d = [d_gray; d_gray; d_gray];
    end
end
function [ f, d ] = sift( type, image, descriptors_per_image)
    if size(image, 3) > 1
        grayscale_image = single(rgb2gray(image));
    else
        grayscale_image = single(image);
    end
        
    if strmatch(type, 'grayscale')
        [f, d] = grayscaleSift(grayscale_image);
    elseif strmatch(type, 'grayscaleDense')
    	[f, d] = grayscaleDenseSift(grayscale_image);
    elseif strmatch(type, 'RGB')
    	[f, d] = RGBSift(image, grayscale_image);
    elseif strmatch(type, 'rgb')
    	[f, d] = rgbSift(image, grayscale_image);
    elseif strmatch(type, 'opponent')
    	[f, d] = opponentSift(image, grayscale_image);
    end
    
    if descriptors_per_image ~= -1
        [f, d] = trimElements(f, d, descriptors_per_image);
    end
end

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

function [f, d] = grayscaleSift(grayscale_image)
    [f, d] = vl_sift(grayscale_image);
end

function [f, d] = grayscaleDenseSift(grayscale_image)
    step = 11;
    size = 10;
    [f, d] = vl_dsift(grayscale_image, 'step', step, 'size', size);
end

function [f, d] = RGBSift(image, grayscale_image)
    image = single(image);
    [f_gray, d_gray] = vl_sift(grayscale_image);
    
    if size(image, 3) > 1
        [f1, d1] = vl_sift(image(:,:,1), 'frames', f_gray);
        [f2, d2] = vl_sift(image(:,:,2), 'frames', f_gray);
        [f3, d3] = vl_sift(image(:,:,3), 'frames', f_gray);
        f = [f1;f2;f3];
        d = [d1;d2;d3];
    else
        f = [f_gray; f_gray; f_gray];
        d = [d_gray; d_gray; d_gray];
    end
end

function [f, d] = rgbSift(image, grayscale_image)
    image = single(image);
    
    [f_gray, d_gray] = vl_sift(grayscale_image);
    
    if size(image, 3) > 1
        %extract each channel
        R  = image(:,:,1);
        G  = image(:,:,2);
        B  = image(:,:,3);
        %convert to opponent space
        r = R ./(R+G+B);
        g = G ./(R+G+B);
        b = B ./(R+G+B);

        [f1, d1] = vl_sift(r, 'frames', f_gray);
        [f2, d2] = vl_sift(g, 'frames', f_gray);
        [f3, d3] = vl_sift(b, 'frames', f_gray);
        f = [f1;f2;f3];
        d = [d1;d2;d3];
    else
        f = [f_gray; f_gray; f_gray];
        d = [d_gray; d_gray; d_gray];
    end
end

function [f, d] = opponentSift(image, grayscale_image)
    image = single(image);
    
    [f_gray, d_gray] = vl_sift(grayscale_image);
    
    if size(image, 3) > 1
        %extract each channel
        R  = image(:,:,1);
        G  = image(:,:,2);
        B  = image(:,:,3);
        %convert to opponent space
        o1 = (R-G)./sqrt(2);
        o2 = (R+G-2*B)./sqrt(6);
        o3 = (R+G+B)./sqrt(3);

        [f1, d1] = vl_sift(o1, 'frames', f_gray);
        [f2, d2] = vl_sift(o2, 'frames', f_gray);
        [f3, d3] = vl_sift(o3, 'frames', f_gray);
        f = [f1;f2;f3];
        d = [d1;d2;d3];
    else
        f = [f_gray; f_gray; f_gray];
        d = [d_gray; d_gray; d_gray];
    end
end
function [param] = get_kernel_param(type)
    if strmatch(type, 'linear')
        param = '-t 0';
    elseif strmatch(type, 'polynomial')
        param = '-t 1';
    elseif strmatch(type, 'rbf')
    	param = '-t 2';
    elseif strmatch(type, 'sigmoid')
    	param = '-t 3';
    end
end

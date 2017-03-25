function [param] = get_kernel_param(type)
    if strmatch(type, 'linear')
        param = '-t 0 -c 10';
    elseif strmatch(type, 'rbf')
    	param = '-t 2 -c 10';
    end
end

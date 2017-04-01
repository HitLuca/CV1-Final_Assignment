% this function takes as input a kernel type and returns the parameters to
% be used in the svm training in order to use that kernel
%
% --inputs
% type: string containing the kernel type
% 
% --outputs
% param: string containing the parameters of the svm classifier

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

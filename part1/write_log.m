

function write_log(path, data)

    file_path = strcat(path, 'log.txt');
    file_id = fopen(file_path, 'wt');
    fprintf(file_id, data);
    
end

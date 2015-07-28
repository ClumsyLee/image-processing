%% bits2str: Read data into a string from bit stream
function str = bits2str(bits)
    % Read the first 32 bit for data_len.
    data_len = bin2dec(int2str(bits(1:32))');

    code_len = data_len * 8 + 32;
    if code_len > numel(bits)  % Wrong header.
        warning(['Wrong header detected, ' ...
                 'trying to read from the whole bit stream.']);
        data_len = ceil((numel(bits) - 32) / 8);
        code_len = data_len * 8 + 32;
    end

    % Read data.
    data = reshape(bits(33:code_len), [8, data_len]);
    str = char([128 64 32 16 8 4 2 1] * data);

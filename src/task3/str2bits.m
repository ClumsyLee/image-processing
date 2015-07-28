%% str2bits: Serialize string into bit stream
function bits = str2bits(str)
    data_len = length(str);

    % Use the first 32 bits to store size of following data.
    header = bitget(data_len, [32:-1:1]');

    % Serialize body.
    str = uint8(str);
    body = zeros(8, data_len);
    for row = 1:8
        body(row, :) = bitget(str, 9 - row);
    end

    bits = [header; body(:)];

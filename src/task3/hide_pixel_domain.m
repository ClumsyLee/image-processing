%% hide_pixel_domain: Hide infomations in pixel domain
%% data_str should be in ASCII.
function data_img = hide_pixel_domain(img, data_str)
    data_len = length(data_str);
    code_len = data_len * 8 + 32;
    if numel(img) < code_len
        error 'The image is too small to hide data.'
    end

    % Use the first 32 bits to store size of following data.
    header = bitget(data_len, [32:-1:1]');

    % Serialize body to binary.
    data_str = uint8(data_str);
    body = zeros(8, data_len);
    for row = 1:8
        body(row, :) = bitget(data_str, 9 - row);
    end

    % Insert into image.
    img = img';
    img(1:code_len) = bitset(img(1:code_len)', 1, [header; body(:)]);
    data_img = uint8(img');

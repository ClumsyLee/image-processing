%% read_pixel_domain: Read infomations from pixel domain
function data_str = read_pixel_domain(data_img)
    data_img = data_img';

    % Read the first 32 bit for data_len.
    data_len = (2 .^ [0:31]) * double(bitget(data_img(32:-1:1), 1)');

    % Read body.
    code_len = data_len * 8 + 32;
    if code_len > numel(data_img)  % Wrong header.
        error 'No data detected in the image.'
    end
    body = reshape(bitget(data_img(33:code_len), 1), [8, data_len]);

    data_str = char([128 64 32 16 8 4 2 1] * double(body));

%% inv_preprocess_some_dct_coeff: Inverse the preprocess
function [img, bits] = inv_preprocess_some_dct_coeff(pre_out, QTAB, ...
                                                     height, width)
    img = zeros(ceil([height width] / 8) * 8);
    bits = zeros(size(pre_out, 2), 1);

    % Scanning blocks.
    k = 1;
    for row = 1:8:height
        for col = 1:8:width
            block = zeros(8, 8);
            this_col = pre_out(:, k);

            % Recover data here.
            bits(k) = max(0, this_col(find(this_col, 1, 'last')));

            block(zigzag(8)) = this_col;          % Inverse Zig-Zag.
            block = block .* QTAB;                     % Inverse quantize.
            img(row:row+7, col:col+7) = idct2(block);  % Inverse DCT.

            k = k + 1;
        end
    end

    img = img(1:height, 1:width);  % Cut to the origin size.
    img = uint8(img + 128);

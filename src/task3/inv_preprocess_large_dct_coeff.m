%% inv_preprocess_some_dct_coeff: Inverse the preprocess
function [img, bits] = inv_preprocess_some_dct_coeff(pre_out, QTAB, ...
                                                     height, width)
    UPPER_BOUND = 4;
    LOWER_BOUND = -UPPER_BOUND - 1;

    img = zeros(ceil([height width] / 8) * 8);
    bits = [];

    % Scanning blocks.
    k = 1;
    for row = 1:8:height
        for col = 1:8:width
            block = zeros(8, 8);
            this_col = pre_out(:, k);

            % Recover data here.
            slot = this_col(this_col <= LOWER_BOUND | this_col >= UPPER_BOUND);
            bits = [bits; bitget(slot, 1, 'int8')];

            block(zigzag(8)) = this_col;          % Inverse Zig-Zag.
            block = block .* QTAB;                     % Inverse quantize.
            img(row:row+7, col:col+7) = idct2(block);  % Inverse DCT.

            k = k + 1;
        end
    end

    img = img(1:height, 1:width);  % Cut to the origin size.
    img = uint8(img + 128);

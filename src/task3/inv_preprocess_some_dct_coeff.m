%% inv_preprocess_some_dct_coeff: Inverse the preprocess
function [img, bits] = inv_preprocess_some_dct_coeff(pre_out, QTAB, ...
                                                     height, width)
    MAX_SLOT = 15;

    img = zeros(ceil([height width] / 8) * 8);
    bits = zeros(numel(img) / 64 * MAX_SLOT, 1);

    % Scanning blocks.
    k = 1;
    for row = 1:8:height
        for col = 1:8:width
            block = zeros(8, 8);
            this_col = pre_out(:, k);

            % Recover data here.
            bits_pos = MAX_SLOT * (k - 1) + 1;
            bits(bits_pos:bits_pos+MAX_SLOT-1) = ...
                bitget(this_col(1:MAX_SLOT), 1, 'int8');

            block(zigzag(8)) = this_col;          % Inverse Zig-Zag.
            block = block .* QTAB;                     % Inverse quantize.
            img(row:row+7, col:col+7) = idct2(block);  % Inverse DCT.

            k = k + 1;
        end
    end

    img = img(1:height, 1:width);  % Cut to the origin size.
    img = uint8(img + 128);

%% inv_preprocess: Inverse the preprocess
function [img] = inv_preprocess(pre_out, QTAB, height, width)
    img = zeros(ceil([height width] / 8) * 8);

    % Scanning blocks.
    k = 1;
    for row = 1:8:height
        for col = 1:8:width
            block = zeros(8, 8);

            block(zigzag(8)) = pre_out(:, k);          % Inverse Zig-Zag.
            block = block .* QTAB;                     % Inverse quantize.
            img(row:row+7, col:col+7) = idct2(block);  % Inverse DCT.

            k = k + 1;
        end
    end

    img = img(1:height, 1:width);  % Cut to the origin size.
    img = uint8(img + 128);

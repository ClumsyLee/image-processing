%% preprocess_some_dct_coeff: Block splitting, DCT & quantization
function out = preprocess_some_dct_coeff(img, bits, QTAB)
    img = double(img) - 128;  % convert to double for matrix ops later.
    bits(bits == 0) = -1;

    % Ensure row/col is a multiple of 8.
    origin_size = size(img);
    new_size = ceil(origin_size / 8) * 8;
    left = new_size - origin_size;
    img = padarray(img, left, 'replicate', 'post');

    out = zeros(64, numel(img) / 64);  % Placeholder for the answer.

    % Scanning blocks.
    k = 1;
    for row = 1:8:new_size(1)
        for col = 1:8:new_size(2)
            c = dct2(img(row:row+7, col:col+7));  % DCT.
            c = round(c ./ QTAB);                 % Quantize.
            c = c(zigzag(8));                     % Zig-Zag.

            % Insert bits here.
            if numel(bits)
                c(min(find(c, 1, 'last') + 1, 64)) = bits(1);
                bits(1) = [];
            end

            out(:, k) = c;
            k = k + 1;
        end
    end

    if numel(bits)
        warning([num2str(numel(bits)) ' bit(s) not encoded']);
    end

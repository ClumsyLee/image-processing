%% quantize_img: Quantize image
function quantized_img = quantize_img(img, L)
    row = size(img, 1);
    col = size(img, 2);
    quantized_img = zeros(row, col);
    for row = 1:row
        for col = 1:col
            pixel = img(row, col, :);
            bits = [bitget(pixel(3), 9-L:8), ...
                    bitget(pixel(2), 9-L:8), ...
                    bitget(pixel(1), 9-L:8)];
            quantized_img(row, col) = bi2de(uint32(bits));
        end
    end

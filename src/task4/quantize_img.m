%% quantize_img: Quantize image
function quantized_img = quantize_img(img, L)
    shift = 8 - L;
    quantized_img = uint32(bitshift(img, -shift));
    quantized_img = bitshift(quantized_img(:, :, 1), 2 * L) + ...
                    bitshift(quantized_img(:, :, 2),     L) + ...
                             quantized_img(:, :, 3);

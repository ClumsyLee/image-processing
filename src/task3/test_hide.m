%% test_hide: Test the result of data hiding
function test_hide(img, data, preprocessor, inv_preprocessor)
    [DC, AC, height, width] = jpeg_encode(img);
    [data_DC, data_AC] = jpeg_hide_encode(img, data, preprocessor);

    % Use normal decoder to avoid cheating.
    decoded_img = jpeg_decode(DC, AC, height, width);
    data_img = jpeg_decode(data_DC, data_AC, height, width);
    [~, recovered_data] = jpeg_hide_decode(data_DC, data_AC, height, width, ...
                                           inv_preprocessor);

    subplot 211
    imshow(decoded_img);
    title(['JPEG encoded (PSNR = ' num2str(psnr(decoded_img, img)), ...
           ' Ratio = ' num2str(compression_ratio(DC, AC, height, width)) ')']);

    subplot 212
    imshow(data_img);
    title(['JPEG encoded with data (PSNR = ' num2str(psnr(data_img, img)), ...
           ' Ratio = ' num2str(compression_ratio(data_DC, data_AC, ...
                                                 height, width)) ')']);
end

%% compression_ratio: Calculate the compression ratio of a image
function ratio = compression_ratio(DC, AC, height, width)
    ratio = (height * width * 8 + 64) / (length(DC) + length(AC) + 64);
end

%% jpeg_hide_encode: Encode an image using JPEG & hiding data in it
function [DC_stream, AC_stream, height, width] = jpeg_hide_encode(img, ...
                                                                  data, ...
                                                                  preprocessor)
    load ../../resource/JpegCoeff

    [height, width] = size(img);  % Save the origin size.
    coefficients = preprocessor(img, str2bits(data), QTAB);

    DC_stream = encode_dc(coefficients(1, :), DCTAB);
    AC_stream = encode_ac(coefficients(2:end, :), ACTAB);
end

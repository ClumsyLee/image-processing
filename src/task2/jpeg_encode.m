%% jpeg_encode: Encode an image using JPEG.
function [DC_stream, AC_stream, height, width] = jpeg_encode(img)
    load ../../resource/JpegCoeff

    [height, width] = size(img);  % Save the origin size.
    coefficients = preprocess(img, QTAB);

    DC_stream = encode_dc(coefficients(1, :), DCTAB);
    AC_stream = encode_ac(coefficients(2:end, :), ACTAB);
end

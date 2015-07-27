%% jpeg_encode: Encode an image using JPEG.
function [DC_stream, AC_stream, height, width] = jpeg_encode(img)
    load ../../resource/JpegCoeff

    [coefficients, new_size] = preprocess(img, QTAB);
    height = new_size(1);
    width  = new_size(2);

    DC_stream = encode_dc(coefficients(1, :), DCTAB);
    AC_stream = encode_ac(coefficients(2:end, :), ACTAB);
end

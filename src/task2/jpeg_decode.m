%% jpeg_decode: decode a JPEG encoded image.
function img = jpeg_decode(DC_stream, AC_stream, height, width)
    load ../../resource/JpegCoeff

    block_num = prod(ceil([height width] / 8));

    DC = decode_dc(DC_stream, DCTAB, block_num);
    AC = decode_ac(AC_stream, ACTAB, block_num);

    img = inv_preprocess([DC; AC], QTAB, height, width);
end

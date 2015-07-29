%% jpeg_hide_decode: decode a JPEG encoded image.
function [img, data] = jpeg_hide_decode(DC_stream, AC_stream, height, width, ...
                                        inv_preprocessor)
    load ../../resource/JpegCoeff

    block_num = prod(ceil([height width] / 8));

    DC = decode_dc(DC_stream, DCTAB, block_num);
    AC = decode_ac(AC_stream, ACTAB, block_num);

    [img, bits] = inv_preprocessor([DC; AC], QTAB, height, width);
    data = bits2str(bits);
end

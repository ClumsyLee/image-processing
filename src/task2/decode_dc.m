%% decode_dc: Decode DC component
function DC = decode_dc(DC_stream, DCTAB, block_num)
    DC = zeros(1, block_num);
    huffman_table = DCTAB(:, 2:end);

    for k = 1:block_num
        [index, len] = huffman_decode(DC_stream, huffman_table);
        DC_stream(1:len) = [];  % Remove decoded.
        category = index - 1;   % category == code length.

        DC(k) = decode_amp(DC_stream(1:category));
        DC_stream(1:category) = [];  % Remove decoded.
    end

    DC = cumsum([DC(1), -DC(2:end)]);

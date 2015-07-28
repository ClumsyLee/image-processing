%% huffman_decode: Decode huffman code
function [index, len] = huffman_decode(codes, huffman_table)
    candidate = 1:size(huffman_table, 1);

    len = 0;
    while length(candidate) > 1
        len = len + 1;
        for k = 1:length(candidate)
            row = candidate(k);
            if codes(len) ~= huffman_table(row, len)
                candidate(k) = 0;  % Mark as unqualified.
            end
        end
        candidate(candidate == 0) = [];  % Eliminate unqualified.
    end

    index = candidate;

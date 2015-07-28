%% decode_ac: Decode AC component
function AC = decode_ac(AC_stream, ACTAB, block_num)
    AC = zeros(63, block_num);
    huffman_table = [ACTAB(:, 4:end)
                     ones(1, 8) 0 0 1 zeros(1, 5);  % ZRL
                     1 0 1 0 zeros(1, 12)];  % EOB

    for block = 1:block_num
        k = 1;
        while k <= 63
            [index, len] = huffman_decode(AC_stream, huffman_table);
            AC_stream(1:len) = [];  % Remove decoded.
            [Run, Size] = decode_index(index);

            if Run == 0 & Size == 0  % EOB
                break  % Go to next block.
            end

            k = k + Run;  % Skip Run steps, because they are already 0s.

            AC(k, block) = decode_amp(AC_stream(1:Size));
            AC_stream(1:Size) = [];  % Remove decoded.
            k = k + 1;  % Skip amp.
        end
    end

%% decode_index: Decode index into Run & Size
function [Run, Size] = decode_index(index)
    if index <= 160
        Run = floor((index - 1) / 10);
        Size = mod(index - 1, 10) + 1;
    elseif index == 161  % ZRL
        Run = 15;
        Size = 0;
    else  % EOB
        Run = 0;
        Size = 0;
    end

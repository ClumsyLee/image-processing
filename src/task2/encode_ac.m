%% encode_ac: Encode AC component
function AC_stream = encode_ac(AC, ACTAB)
    AC_stream = [];

    for k = 1:size(AC, 2)  % For every block.
        col = AC(:, k);

        amp_index = find(col, 1);  % Find first non-zero.
        while numel(amp_index)
            amp = col(amp_index);
            Run = amp_index - 1;

            % Reduce zeros.
            while Run > 15
                AC_stream = [AC_stream 1 1 1 1 1 1 1 1 0 0 1];
                Run = Run - 16;
            end

            % Encode run/size
            Size = amp2cate(amp);
            row = Run * 10 + Size;
            huff = ACTAB(row, 4:3+ACTAB(row, 3));

            AC_stream = [AC_stream huff dec2_1s(amp)];  % Add to stream.

            col(1:amp_index) = [];  % Delete this run/amp.
            amp_index = find(col, 1);  % Find next non-zero.
        end
        if length(col)  % Zero(s) not coded, insert EOB
            AC_stream = [AC_stream 1 0 1 0];
        end
    end

    AC_stream = AC_stream';

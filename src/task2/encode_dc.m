%% encode_dc: Encode DC component
function DC_stream = encode_dc(DC, DCTAB)
    errors = diff_encode(DC);
    category = ceil(log2(abs(errors) + 1));

    DC_stream = [];

    for k = 1:length(errors)
        e = errors(k);
        row = category(k) + 1;

        huff = DCTAB(row, 2:1+DCTAB(row, 1));
        if e == 0
            mag = [];
        else
            mag = dec2bin(abs(e)) - '0';
            if e < 0
                mag = 1 - mag;  % Use 1's complement.
            end
        end

        DC_stream = [DC_stream huff mag];
    end

    DC_stream = DC_stream';
end

%% diff_encode: Encode using differential coding.
function Y = diff_encode(X)
    Y = [0 X] - [X 0];  % X(n - 1) - X(n).
    Y = [X(1), Y(2:end-1)];
end

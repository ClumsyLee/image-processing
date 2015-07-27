%% dec2_1s: Convert decimal to 1's complement
function y = dec2_1s(dec)
    y = dec2bin(abs(dec)) - '0';
    if dec < 0
        y = 1 - y;  % Use 1's complement.
    end

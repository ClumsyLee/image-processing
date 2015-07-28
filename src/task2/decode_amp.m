%% decode_amp: Decode mag/amp
function amp = decode_amp(code)
    if isempty(code)
        amp = 0;
    elseif code(1) == 0  % Nagetive.
        amp = -bin2dec(int2str(1 - code)');
    else
        amp = bin2dec(int2str(code)');
    end

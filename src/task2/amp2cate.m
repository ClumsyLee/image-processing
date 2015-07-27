%% amp2cate: Convert amp to category
function cate = amp2cate(amp)
    cate = ceil(log2(abs(amp) + 1));

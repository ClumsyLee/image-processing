% Luis' approach
function index = zigzag(r)
M = bsxfun(@plus, (1:r).', 0:r-1);
M = M + bsxfun(@times, (1:r).'/(2*r), (-1).^M);
[~, index] = sort(M(:));

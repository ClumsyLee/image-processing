%% my_dct2: My implementation of dct2
function B = my_dct2(A)
    % DCT transform matrix.
    [row, col] = size(A);
    B = trans_mat(row) * A * trans_mat(col)';

%% trans_mat: Construct NxN DCT transform matrix
function D = trans_mat(N)
    D = sqrt(2 / N) * cos([0:N-1]' * [1:2:2*N-1] * pi / (2 * N));
    D(1, :) = sqrt(1 / N);

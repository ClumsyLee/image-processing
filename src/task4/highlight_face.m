%% highlight_face: Highlight faces in an image
function varargout = highlight_face(img, faces_UL, faces_LR)
    WIDTH = 2;

    r = img(:, :, 1);
    g = img(:, :, 2);
    b = img(:, :, 3);
    row = (1:size(img, 1))';
    col = 1:size(img, 2);

    for k = 1:size(faces_UL, 2)
        row_range = (row >= faces_UL(1, k) & row <= faces_LR(1, k));
        col_range = (col >= faces_UL(2, k) & col <= faces_LR(2, k));
        row_bound = row_range & (row - faces_UL(1, k) < WIDTH | ...
                                 faces_LR(1, k) - row < WIDTH);
        col_bound = col_range & (col - faces_UL(2, k) < WIDTH | ...
                                 faces_LR(2, k) - col < WIDTH);

        frame = bsxfun(@and, row_bound, col_range) | ...
                bsxfun(@and, col_bound, row_range);

        r(frame) = 255;
        g(frame) = 0;
        b(frame) = 0;
    end

    I = cat(3, r, g, b);

    if nargout
        varargout(1) = I;
    else
        imshow(I);
    end

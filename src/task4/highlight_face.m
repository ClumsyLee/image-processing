%% highlight_face: Highlight faces in an image
function varargout = highlight_face(img, faces)
    WIDTH = 2;

    r = img(:, :, 1);
    g = img(:, :, 2);
    b = img(:, :, 3);

    for k = 1:size(faces, 3)
        face = faces(:, :, k);
        row = (1:size(img, 1))';
        col = 1:size(img, 2);

        row_range = (row >= face(1, 1) & row <= face(1, 2));
        col_range = (col >= face(2, 1) & col <= face(2, 2));
        row_bound = row_range & (row - face(1, 1) < WIDTH | ...
                                 face(1, 2) - row < WIDTH);
        col_bound = col_range & (col - face(2, 1) < WIDTH | ...
                                 face(2, 2) - col < WIDTH);

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

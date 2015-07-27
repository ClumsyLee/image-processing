%% add_chess_board_mask: Add a mask like a chess board to a image
function masked_img = add_chess_board_mask(img)
    masked_img = img;  % Copy the image.

    [y_max, x_max, ~] = size(img);

    for y = 1:y_max
        for x = 1:x_max
            if mod(ceil(x / x_max * 8) + ceil(y / y_max * 8), 2) == 0
                masked_img(y, x, :) = [0 0 0];
            end
        end
    end

%% add_chess_board_mask: Add a mask like a chess board to a image
function masked_img = add_chess_board_mask(img)
    masked_img = img;  % Copy the image.

    [x_max, y_max, ~] = size(img);

    for x = 1:x_max
        for y = 1:y_max
            if mod(ceil(x / x_max * 8) + ceil(y / y_max * 8), 2) == 0
                masked_img(x, y, :) = [0 0 0];
            end
        end
    end

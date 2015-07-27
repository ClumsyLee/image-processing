%% add_circle: Add a red circle to the center of the given image
function circled_img = add_circle(img)
    circled_img = img;  % Copy the image.

    [y_max, x_max, ~] = size(circled_img);
    r = min(x_max, y_max) / 2;
    center = [(x_max + 1) / 2, (y_max + 1) / 2];

    for y = 1:y_max
        for x = 1:x_max
            if norm([x y] - center) <= r
                circled_img(y, x, :) = [255 0 0];
            end
        end
    end

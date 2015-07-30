%% detect_face: Detect faces in an image
function varargout = detect_face(img, model, sample_size, sample_step, ...
                                 expand_ratio, expand_threshold)
    faces_UL = [];
    faces_LR = [];
    L = log2(length(model)) / 3;  % Infer L from model.
    img_size = [size(img, 1); size(img, 2)];

    % Generate samples.
    row_sample = 1:sample_step:img_size(1)-sample_size+1;
    col_sample = 1:sample_step:img_size(2)-sample_size+1;

    faces_UL = combvec(row_sample, col_sample);
    faces_LR = faces_UL + sample_size - 1;

    % Expand every sample.
    for k = 1:size(faces_UL, 2)
        k
        prev_size = [0; 0];
        now_size = faces_LR(:, k) - faces_UL(:, k);

        while any(prev_size ~= now_size)
            UL = faces_UL(:, k);
            LR = faces_LR(:, k);
            expand = ceil(now_size * expand_ratio);

            new_UL = repmat(UL, 1, 4) + ...
                [-expand(1), 0         , 0, 0
                 0         , -expand(2), 0, 0];
            new_LR = repmat(LR, 1, 4) + ...
                [0, 0, expand(1), 0
                 0, 0, 0        , expand(2)];

            % Ensure they are still inside the image.
            new_UL = max(new_UL, ones(2, 4));
            new_LR = min(new_LR, repmat(img_size, 1, 4));

            for direction = 1:4
                dir_UL = new_UL(:, direction);
                dir_LR = new_LR(:, direction);
                if dir_UL == UL & dir_LR == LR  % UL & LR not changed.
                    continue
                end

                part = img_part(img, dir_UL, dir_LR);
                part_dist(direction) = face_distance(part, model);
            end
            [value, index] = max(part_dist);
            if value >= expand_threshold
                faces_UL(:, k) = new_UL(:, index);
                faces_LR(:, k) = new_LR(:, index);
            end

            prev_size = now_size;
            now_size = faces_LR(:, k) - faces_UL(:, k);
        end
    end


    if nargout
    else
        highlight_face(img, faces_UL, faces_LR);
    end
end

%% img_part: Get part of the image
function part = img_part(img, UL, LR)
    part = img(UL(1):LR(1), UL(2):LR(2), :);
end

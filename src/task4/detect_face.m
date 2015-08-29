%% detect_face: Detect faces in an image
function varargout = detect_face(img, model, sample_size, sample_step, ...
                                 sample_threshold)
    faces_UL = [];
    faces_LR = [];
    origin_img = img;
    img = quantize_img(img, log2(length(model)) / 3);  % Infer L from model.
    img_size = [size(img, 1); size(img, 2)];

    % Generate samples.
    row_sample = 1:sample_step:img_size(1)-sample_size+1;
    col_sample = 1:sample_step:img_size(2)-sample_size+1;

    faces_UL = combvec(col_sample, row_sample);
    faces_UL = flipud(faces_UL);
    faces_LR = faces_UL + sample_size - 1;

    pick = logical(zeros(1, size(faces_UL, 2)));
    block_dist = zeros(1, size(faces_UL, 2));
    for k = 1:size(faces_UL, 2)
        d = part_distance(img, faces_UL(:, k), faces_LR(:, k), model);
        block_dist(k) = d;
        pick(k) = d <= sample_threshold;
    end

    [value, index] = sort(block_dist);
    for k1 = 2:length(pick)
        index1 = index(k1);
        if ~pick(index1)
            continue
        end
        x11 = faces_UL(1, index1);
        y11 = faces_UL(2, index1);
        x12 = faces_LR(1, index1);
        y12 = faces_LR(2, index1);
        for k2 = 1:k1-1
            index2 = index(k2);
            if ~pick(index2)
                continue
            end
            x21 = faces_UL(1, index2);
            y21 = faces_UL(2, index2);
            x22 = faces_LR(1, index2);
            y22 = faces_LR(2, index2);

            if (x11 >= x21 && x11 <= x22 || ...
                x12 >= x21 && x12 <= x22) && ...
               (y11 >= y21 && y11 <= y22 || ...
                y12 >= y21 && y12 <= y22)
               pick(index1) = 0;
               continue
           end
        end
    end

    faces_UL = faces_UL(:, pick(:));
    faces_LR = faces_LR(:, pick(:));

    if nargout
    else
        highlight_face(origin_img, faces_UL, faces_LR);
    end
end

%% part_distance: Get part of the image
function d = part_distance(img, UL, LR, model)
    d = face_distance(img(UL(1):LR(1), UL(2):LR(2), :), model);
end

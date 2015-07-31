%% detect_face: Detect faces in an image
function varargout = detect_face(img, model, sample_size, sample_step, ...
                                 sample_threshold, expand_ratio, expand_threshold)
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
    for k = 1:size(faces_UL, 2)
        pick(k) = part_distance(img, faces_UL(:, k), faces_LR(:, k), model) <= sample_threshold;
    end
    pick = reshape(pick, length(col_sample), length(row_sample))';

    % faces_UL = faces_UL(:, pick);
    % faces_LR = faces_LR(:, pick);

    for row = 1:length(row_sample)
        for col = 1:length(col_sample)
            if ~(pick(row, col))
                continue
            end

            col_offset = 0;
            while col + col_offset + 3 <= length(col_sample)
                if pick(row, col + col_offset + 3)
                    col_offset = col_offset + 3;
                elseif pick(row, col + col_offset + 2)
                    col_offset = col_offset + 2;
                elseif pick(row, col + col_offset + 1)
                    col_offset = col_offset + 1;
                else
                    break
                end
                % faces_LR(:, k) = faces_LR(:, k + col_offset);
                pick(row, col+1:col+col_offset) = false;
            end

            row_offset = 0;
            while row + row_offset + 3 <= length(row_sample)
                if any(pick(row + row_offset + 3, col:col+col_offset))
                    row_offset = row_offset + 3;
                elseif any(pick(row + row_offset + 2, col:col+col_offset))
                    row_offset = row_offset + 2;
                elseif any(pick(row + row_offset + 1, col:col+col_offset))
                    row_offset = row_offset + 1;
                else
                    break
                end
                % faces_LR(:, k) = faces_LR(:, start_k + col_offset);
                pick(row+1:row+row_offset, col:col+col_offset) = false;
            end

            faces_LR(:, (row - 1) * length(col_sample) + col) = ...
                faces_LR(:, (row + row_offset - 1) * length(col_sample) + col + col_offset);
        end
    end

    pick = pick';
    pick = pick(:);

    for k = 1:length(pick)
        if pick(k)
            part_dist(k) = part_distance(img, faces_UL(:, k), faces_LR(:, k), model) <= expand_threshold;
        end
    end


    for k1 = 1:length(pick)-1
        if ~pick(k1)
            continue
        end
        UL1 = faces_UL(:, k1);
        LR1 = faces_LR(:, k1);
        for k2 = k1+1:length(pick)
            if ~pick(k2)
                continue
            end
            UL2 = faces_UL(:, k2);
            LR2 = faces_LR(:, k2);
            if UL1 <= UL2 & LR1 >= UL2 | ...
               UL1 <= LR2 & LR1 >= LR2 | ...
               UL2 <= UL1 & LR2 >= UL1 | ...
               UL2 <= LR1 & LR2 >= LR1
                pick(k2) = false;
            end
        end
    end

    % Expand every sample.
    % new_count = 1;
    % new_UL = [];
    % new_LR = [];
    % for k = 1:size(faces_UL, 2)
    %     k
    %     UL = faces_UL(:, k);
    %     LR = faces_LR(:, k);
    %     prev_size = [0; 0];
    %     now_size = LR - UL;
        % if part_distance(img, UL, LR, model) > sample_threshold
        %     continue
        % end

        % step = faces_UL(:, )
        % for step = 1:2
        %     UL_diff = faces_UL(:, k + step) - UL;
        %     if UL_diff(2) == 0 & UL_diff(1) <= 2 * sample_step


        % min_value = intmax;
        % while any(prev_size ~= now_size)
            % % expand = ceil(now_size * expand_ratio);
            % expand = 10 * ones(2, 1);

            % new_UL = repmat(UL, 1, 4) + ...
            %     [-expand(1), 0         , 0, 0
            %      0         , -expand(2), 0, 0];
            % new_LR = repmat(LR, 1, 4) + ...
            %     [0, 0, expand(1), 0
            %      0, 0, 0        , expand(2)];

            % % Ensure they are still inside the image.
            % new_UL = max(new_UL, ones(2, 4));
            % new_LR = min(new_LR, repmat(img_size, 1, 4));

            % for direction = 1:4
            %     dir_UL = new_UL(:, direction);
            %     dir_LR = new_LR(:, direction);
            %     if dir_UL == UL & dir_LR == LR  % UL & LR not changed.
            %         continue
            %     end

            %     part_dist(direction) = part_distance(img, dir_UL, dir_LR, model);
            % end

            % [value, index] = min(part_dist);
            % if value <= 0
            %     faces_UL(:, k) = new_UL(:, index);
            %     faces_LR(:, k) = new_LR(:, index);
            %     min_value = value;
            % end


            % UL = faces_UL(:, k);
            % LR = faces_LR(:, k);
            % prev_size = now_size;
            % now_size = LR - UL;
        % end
    % end



    for k = 1:size(faces_UL, 2)
        if (pick(k))
            pick(k) = part_distance(img, faces_UL(:, k), faces_LR(:, k), model) <= expand_threshold;
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

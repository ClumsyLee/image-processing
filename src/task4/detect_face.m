%% detect_face: Detect faces in an image
function varargout = detect_face(img, model, sample_size, sample_step, ...
                                 expand_ratio)
    faces_UL = [];
    faces_LR = [];
    L = log2(length(model)) / 3;  % Infer L from model.
    img_size = size(img)';

    row_sample = 1:sample_step:img_size(1)-sample_size+1;
    col_sample = 1:sample_step:img_size(2)-sample_size+1;

    faces_UL = combvec(row_sample, col_sample);
    faces_LR = faces_UL + sample_size;

    if nargout
    else
        highlight_face(img, faces_UL, faces_LR);
    end

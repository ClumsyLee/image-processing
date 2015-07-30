%% train_face_model: Train a face model using face images
%% L is the bits used for every color component
function v = train_face_model(imgs, L)
    bin_num = 2 ^ (3 * L);
    v = zeros(bin_num, 1);
    pixels = 0;
    for k = 1:length(imgs)
        img = quantize_img(imgs{k}, L);
        v = v + histc(img(:), 0:bin_num-1) / numel(img);
    end
    v = v / length(imgs);
end

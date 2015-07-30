%% face_distance: Distance between a quantized region and a face model
function d = face_distance(region, model)
    u = histc(region(:), 0:length(model)-1) / numel(region);
    d = 1 - sum(sqrt(u .* model));

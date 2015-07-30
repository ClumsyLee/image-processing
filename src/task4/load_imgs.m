%% load_imgs: Load images under a directory.
function imgs = load_imgs(directory)
    imgs = {};
    files = dir(directory);
    for file = files'
        name = [directory '/' file.name];
        if ~isdir(name)
            imgs = [imgs; imread(name)];
        end
    end

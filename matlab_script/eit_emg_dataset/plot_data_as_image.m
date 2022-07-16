%

cur_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(cur_dir, '..', '..', 'data', 'EIT_EMG_Gait_210127');

saved_file = fullfile(root_dir, 'Dataset_TIME_SIZE_150.mat');
data = load(saved_file, 'X_train').X_train;
label = load(saved_file, 'Y_train').Y_train;

for i = randperm(size(label, 1), 5)
    img = data(i, :, :);
    img = squeeze(img)';
    img = imresize(img, 5);
    y = label(i);
    
    saved_img = fullfile(root_dir, sprintf('sample_img%d_G%d.png', i, y));
    imwrite(img, saved_img, 'PNG');
end

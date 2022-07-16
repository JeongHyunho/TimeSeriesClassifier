

% setup
N_CATS = 2;
N_TARGETS = 72;
TIME_SIZE = 80;
INTERVAL = 10;
VAL_RATIO = 0.2;
TEST_RATIO = 0.2;
SEED = 41;

rng(SEED)

cur_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(cur_dir, '..', '..', 'data', 'Gait_Phase_220218');

% mean, standard deviation of all data
data_files = dir(fullfile(root_dir, 'S*.mat'));
n_trials = length(data_files);
data_all = [];

for i = 1:n_trials
    filename = fullfile(data_files(i).folder, data_files(i).name);
    data = load(filename, 'Data_con').Data_con;
    data_all = [data_all, data(1:N_TARGETS, :)];
end
target_mean = mean(data_all, 2)';
target_std = std(data_all, 1, 2)';

% store all dataset, 3 for split
X_data = nan(3, 2000, TIME_SIZE, N_TARGETS);
Y_data = nan(3, 2000, N_CATS);
X_test_stream = {};
Y_test_stream = {};
idx_stream = 1;

cur_idx = [1, 1, 1];
trial_idx = randperm(n_trials);
for i = trial_idx
    filename = fullfile(data_files(i).folder, data_files(i).name);
    name_enc = split(data_files(i).name, ["_", "."]);
    sub_id = str2num(name_enc{1}(2:end));
    trial = str2num(name_enc{3});
    
    data = load(filename, 'Data_con').Data_con;
    n_frames = size(data, 2);
    
    if n_frames < TIME_SIZE
            warning('S%d T%d has %d frames less than %d.\n', ...
                sub_id, trial, n_frames, TIME_SIZE)
    end
    
    if i <= n_trials * (1 - VAL_RATIO - TEST_RATIO)
        split_idx = 1;
    elseif i <= n_trials * (1 - TEST_RATIO)
        split_idx = 2;
    else
        split_idx = 3;
        field = num2str(cur_idx(3));
        X_test_stream{idx_stream} = ...
            (data(1:N_TARGETS, :)' - reshape(target_mean, 1, N_TARGETS)) ...
            ./ reshape(target_std, 1, N_TARGETS);
        Y_test_stream{idx_stream} = data(76, :)';
        idx_stream = idx_stream + 1;
    end
    
    one_hot_label = zeros(n_frames, 2);
    label = data(76, :);
    hot_idx = sub2ind([n_frames, 2], 1:n_frames, label + 1);
    one_hot_label(hot_idx) = 1;
    
    start_idx = 1;
    while start_idx + TIME_SIZE < n_frames
        range = start_idx:start_idx + TIME_SIZE - 1;
        X_data(split_idx, cur_idx(split_idx), :, :) = ...
            data(1:N_TARGETS, range)';
        Y_data(split_idx, cur_idx(split_idx), :) = one_hot_label(range(end), :);
        
        start_idx = start_idx + INTERVAL;
        cur_idx(split_idx) = cur_idx(split_idx) + 1;
    end
end

% normalization
X_data = (X_data - reshape(target_mean, 1, 1, 1, N_TARGETS)) ...
    ./ reshape(target_std, 1, 1, 1, N_TARGETS);

% remove nan and split train/ val/ test
X_train = squeeze(X_data(1, 1:cur_idx(1) - 1, :, :));
Y_train = squeeze(Y_data(1, 1:cur_idx(1) - 1, :));

X_val = squeeze(X_data(2, 1:cur_idx(2) - 1, :, :));
Y_val = squeeze(Y_data(2, 1:cur_idx(2) - 1, :));

X_test = squeeze(X_data(3, 1:cur_idx(3) - 1, :, :));
Y_test = squeeze(Y_data(3, 1:cur_idx(3) - 1, :));

% save
saved_file = fullfile(root_dir, ...
    sprintf('Dataset_TIME_SIZE_%d.mat', TIME_SIZE));
save(saved_file, 'X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test', ...
    'X_test_stream', 'Y_test_stream')

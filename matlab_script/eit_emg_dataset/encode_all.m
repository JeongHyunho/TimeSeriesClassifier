% 

% setup
N_CATS = 6;
N_TARGETS = 72;
TIME_SIZE = 150;
N_BATCHES = 20;
VAL_RATIO = 0.2;
TEST_RATIO = 0.2;
SEED = 41;

rng(SEED)

cur_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(cur_dir, '..', '..', 'data', 'EIT_EMG_Gait_210127');

% mean, standard deviation of all conditions
data_files = dir(fullfile(root_dir, 'S*.mat'));
data_all = [];

for i = 1:length(data_files)
    filename = fullfile(data_files(i).folder, data_files(i).name);
    data = load(filename, 'Data').Data;
    data_all = [data_all, data(1:N_TARGETS, :)];
end
target_mean = mean(data_all, 2)';
target_std = std(data_all, 1, 2)';

% store all dataset, 3 for split
X_data = nan(3, 2000, TIME_SIZE, N_TARGETS);
Y_data = nan(3, 2000, 1);

cur_idx = [1, 1, 1];
for i = 0:N_CATS-1
    target_files = dir(fullfile(root_dir, 'S*_' + string(i) +'_*.mat'));
    
    % split train/val/test based on trial
    n_total_trials = length(target_files);
    rnd_trials = randperm(n_total_trials);
    
    for j = 1:length(target_files)
        filename = fullfile(target_files(j).folder, target_files(j).name);
        name_enc = split(target_files(j).name, ["_", "."]);
        sub_id = str2num(name_enc{1}(2:end));
        label = str2num(name_enc{2});
        trial = str2num(name_enc{3});

        data = load(filename, 'Data').Data;
        n_frames = size(data, 2);
        
        % check minimum length
        if n_frames < TIME_SIZE
            warning('S%d G%d T%d has %d frames less than %d.\n', ...
                sub_id, label, trial, n_frames, TIME_SIZE)
        end
        
        % store every TIME_SIZE chunk as one batch with INTERVAL overlap
        if j <= n_total_trials * (1 - VAL_RATIO - TEST_RATIO)
            split_idx = 1;
        elseif j <= n_total_trials * (1 - TEST_RATIO)
            split_idx = 2;
        else
            split_idx = 3;
        end
        
        start_indices = linspace(1, n_frames - TIME_SIZE + 1, N_BATCHES);
        for idx = round(start_indices)
            range = idx:idx + TIME_SIZE - 1;
            % do normalization
            X_data(split_idx, cur_idx(split_idx), :, :) = ...
                data(1:N_TARGETS, range)';
            Y_data(split_idx, cur_idx(split_idx)) = label;
            cur_idx(split_idx) = cur_idx(split_idx) + 1;
        end
    end
end

% normalization
X_data = (X_data - reshape(target_mean, 1, 1, 1, N_TARGETS)) ...
    ./ reshape(target_std, 1, 1, 1, N_TARGETS);

% remove nan and split train/ val/ test
X_train = squeeze(X_data(1, 1:cur_idx(1) - 1, :, :));
Y_train = Y_data(1, 1:cur_idx(1) - 1)';

X_val = squeeze(X_data(2, 1:cur_idx(2) - 1, :, :));
Y_val = Y_data(2, 1:cur_idx(2) - 1)';

X_test = squeeze(X_data(3, 1:cur_idx(3) - 1, :, :));
Y_test = Y_data(3, 1:cur_idx(3) - 1)';

% save
saved_file = fullfile(root_dir, ...
    sprintf('Dataset_TIME_SIZE_%d.mat', TIME_SIZE));
save(saved_file, 'X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test')

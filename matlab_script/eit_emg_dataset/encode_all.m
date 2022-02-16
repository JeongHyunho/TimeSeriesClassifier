% 

% setup
N_CATS = 6;
N_TARGETS = 72;
TIME_SIZE = 150;
INTERVAL = 20;
VAL_RATIO = 0.2;
TEST_RATIO = 0.2;
SEED = 41;

rng(SEED)

cur_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(cur_dir, '..', '..', 'data', 'EIT_EMG_Gait_210127');

% mean, standard deviation of each condition
cond_mean = zeros(N_CATS, N_TARGETS);
cond_std = zeros(N_CATS, N_TARGETS);

for i = 1:N_CATS
    cond_files = dir(fullfile(root_dir, sprintf('S*_%d_*.mat', i-1)));
    cond_data_all = [];
    
    for j = 1:length(cond_files)
        filename = fullfile(cond_files(j).folder, cond_files(j).name);
        data = load(filename, 'Data').Data;
        cond_data_all = [cond_data_all, data(1:N_TARGETS, :)];
    end
    
    cond_mean(i, :) = mean(cond_data_all, 2)';
    cond_std(i, :) = std(cond_data_all, 1, 2)';
end

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
        
        start_idx = 1;
        while start_idx + TIME_SIZE < n_frames
            range = start_idx:start_idx + TIME_SIZE - 1;
            % do normalization
            X_data(split_idx, cur_idx(split_idx), :, :) = ...
                (data(1:N_TARGETS, range)' - cond_mean(label + 1, :)) ./ cond_std(label + 1, :);
            Y_data(split_idx, cur_idx(split_idx)) = label;
            
            start_idx = start_idx + INTERVAL;
            cur_idx(split_idx) = cur_idx(split_idx) + 1;
        end
    end
end

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

% This code loads and processes all trials.
% It saves train, validation dataset in root data directory.
% Validation set are a part of total time steps data.

% datset setups (time length, intervals, limits, val set size)
TIME_SIZE = 150;
INTERVAL = 20;
KNEE_RNG = [0, 70];
ANK_RNG = [-10, 30];
VAL_RATIO = 0.2;

% root dataset path
root_dir = fullfile('C:', 'Users', 'biomechanics', 'Dropbox', 'SNU_DATASET');
folders = dir(root_dir);

% stored matrix X(B x T x N), Y(B x T)
targets = ["R_contact", "R_knee_flex_ang", "R_ank_rot_ang", "R_vastus_medialis", ...
    "R_medial_ham", "R_vastus_lateralis", "R_gastrocnemius"];
labels = "Phase";

cur_idx = 1;
n_targets = length(targets);
n_categories = 4;

X_data = nan(1000, TIME_SIZE, n_targets);
Y_data = nan(1000, TIME_SIZE, n_categories);

for i = 1: length(folders)
    % check vaild path
    folder = folders(i);
    folder_name = folder.name;
    if ~contains(folder_name, 'Pilot')
        continue
    else
        fprintf('Processing %s ...\n', folder_name)
    end
    
    % find all strides
    folder_dir = fullfile(root_dir, folder_name, 'trial*.mat');
    filenames = dir(folder_dir);
    
    % iterate trials data and store them
    for j = 1:length(filenames)
        filename = filenames(j).name;
        load(fullfile(root_dir, folder_name, filename), 'ex_data')
        
        t_length = size(ex_data.Phase, 1); 
        data_mat = zeros(t_length, n_targets);
        
        % check minimum length
        if t_length < TIME_SIZE
            warning('Trial %s has less than %d time steps. Ignored ... \n', ...
            filename, TIME_SIZE)
        end
        
        for k = 1:n_targets
            target = targets(k);
            
            % make EMG sinal have a uniform scale
            switch(target)
                case {'R_vastus_medialis', 'R_medial_ham', ...
                        'R_vastus_lateralis', 'R_gastrocnemius'}
                    emg_signal = ex_data.(target);
                    scale = 2 * std(emg_signal);
                    data_mat(:, k) = 1 / scale * emg_signal;
                case 'R_knee_flex_ang'
                    knee_ang = ex_data.(target);
                    data_mat(:, k) = 2 * (knee_ang - KNEE_RNG(1)) / (KNEE_RNG(2) - KNEE_RNG(1)) - 1;
                case 'R_ank_rot_ang'
                    ank_ang = ex_data.(target);
                    data_mat(:, k) = 2 * (ank_ang - ANK_RNG(1)) / (ANK_RNG(2) - ANK_RNG(1)) - 1;
                otherwise
                    data_mat(:, k) = ex_data.(target);
            end
        end
        label_mat = ex_data.(labels);
        
        % sotore every TIME_SIZE chunk as one batch
        start_idx = 1;
        while start_idx + TIME_SIZE < t_length
            range = start_idx:start_idx + TIME_SIZE - 1;
            X_data(cur_idx, :, :) = data_mat(range, :);
            Y_data(cur_idx, :, :) = label_mat(range, :);
            
            start_idx = start_idx + INTERVAL;
            cur_idx = cur_idx + 1;
        end
    end
    
    % delete pre-assigned nan's
    X_data(isnan(X_data)) = [];
    X_data = reshape(X_data, [], TIME_SIZE, n_targets);
    
    Y_data(isnan(Y_data)) = [];
    Y_data = reshape(Y_data, [], TIME_SIZE, n_categories);
end

% split train, validataion set
n_samples = size(X_data, 1);
n_val = floor(VAL_RATIO * n_samples);
val_idx = randsample(n_samples, n_val);
train_idx = setdiff(1:n_samples, val_idx);

X_train = X_data(train_idx, :, :);
Y_train = Y_data(train_idx, :, :);
X_val = X_data(val_idx, :, :);
Y_val = Y_data(val_idx, :, :);

% save data sets
saved_file = fullfile(root_dir, 'SNUH_data_total.mat');
save(saved_file, "X_train", "Y_train", "X_val", "Y_val")

fprintf('(Train: %d, Val: %d) Saved to %s \n', ...
    n_samples - n_val, n_val, saved_file)

% This code loads and processes all trials.
% It saves train, validation dataset in root data directory.
% Validation set one trial of each subject.
% It assumes prosthesis user, so only contains sEMG from thigh.

% datset setups (time length, intervals, limits, val set size)
TIME_SIZE = 150;
INTERVAL = 20;
KNEE_RNG = [0, 70];
ANK_RNG = [-10, 30];
VAL_N_TRIAL = 1;

% root dataset path
root_dir = fullfile('C:', 'Users', 'biomechanics', 'Dropbox', 'SNU_DATASET');
folders = dir(root_dir);

% stored matrix X(B x T x N), Y(B x T)
targets = ["R_contact", "R_knee_flex_ang", "R_ank_rot_ang", "R_vastus_medialis", ...
    "R_medial_ham", "R_vastus_lateralis"];
labels = "Phase";

cur_train_idx = 1;
cur_val_idx = 1;
n_targets = length(targets);
n_categories = 4;

X_train = nan(1000, TIME_SIZE, n_targets);
Y_train = nan(1000, TIME_SIZE, n_categories);

X_val = nan(500, TIME_SIZE, n_targets);
Y_val = nan(500, TIME_SIZE, n_categories);

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
        % split validation set based on trial
        start_idx = 1;
        while start_idx + TIME_SIZE < t_length
            range = start_idx:start_idx + TIME_SIZE - 1;
            start_idx = start_idx + INTERVAL;
            
            if  j <= length(filenames) - VAL_N_TRIAL
                X_train(cur_train_idx, :, :) = data_mat(range, :);
                Y_train(cur_train_idx, :, :) = label_mat(range, :);
                cur_train_idx = cur_train_idx + 1;
            else
                X_val(cur_val_idx, :, :) = data_mat(range, :);
                Y_val(cur_val_idx, :, :) = label_mat(range, :);
                cur_val_idx = cur_val_idx + 1;
            end
        end
    end
    
    % delete pre-assigned nan's
    X_train(isnan(X_train)) = [];
    X_train = reshape(X_train, [], TIME_SIZE, n_targets);
    Y_train(isnan(Y_train)) = [];
    Y_train = reshape(Y_train, [], TIME_SIZE, n_categories);
    
    X_val(isnan(X_val)) = [];
    X_val = reshape(X_val, [], TIME_SIZE, n_targets);
    Y_val(isnan(Y_val)) = [];
    Y_val = reshape(Y_val, [], TIME_SIZE, n_categories);
end

% save data sets
saved_file = fullfile(root_dir, 'SNUH_data_amputee_trial.mat');
save(saved_file, "X_train", "Y_train", "X_val", "Y_val")

fprintf('(Train: %d, Val: %d) Saved to %s \n', ...
    length(X_train), length(X_val), saved_file)

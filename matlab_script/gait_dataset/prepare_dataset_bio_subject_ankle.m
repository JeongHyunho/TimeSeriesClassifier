% This code loads and processes all trials.
% It saves train, validation dataset in root data directory.
% Validation set are from separted subjects.
% It only contains bio-signal (sEMG).
% Target value are right ankle angle

% datset setups (time length, intervals, limits, val set size)
TIME_SIZE = 150;
INTERVAL = 20;
VAL_N_SUJ = 6;
TEST_N_SUJ = 6;
EMG_SCALE = 2000;

setup = {'time_size', TIME_SIZE, 'interval', INTERVAL, ...
    'val_n_suj', VAL_N_SUJ, 'test_n_suj', TEST_N_SUJ, 'emg_scale', EMG_SCALE};

% root dataset path and check valid folder
root_dir = fullfile('C:', 'Users', 'biomechanics', 'Dropbox', 'SNU_DATASET', 'GAIT');
folders = dir(root_dir);

for i = length(folders):-1:1
    foldername = folders(i).name;
    if ~ contains(foldername, 'Pilot')
        folders(i) = [];
    end
end

% stored matrix X(B x T x N), Y(B x T)
sources = ["R_vastus_medialis", "R_medial_ham", "R_vastus_lateralis", "R_gastrocnemius", ...
    "L_vastus_medialis", "L_medial_ham", "L_vastus_lateralis", "L_gastrocnemius"];
labels = "R_ank_rot_ang";
HS = "R_HS";

n_targets = length(sources);
n_categories = 1;

X_train = nan(1000, TIME_SIZE, n_targets);
Y_train = nan(1000, TIME_SIZE, n_categories);
Y_train_means = nan(1000, n_categories);
Y_train_sigmas = nan(1000, n_categories);

X_val = nan(500, TIME_SIZE, n_targets);
Y_val = nan(500, TIME_SIZE, n_categories);
Y_val_means = nan(500, n_categories);
Y_val_sigmas = nan(500, n_categories);

X_test = nan(500, TIME_SIZE, n_targets);
Y_test = nan(500, TIME_SIZE, n_categories);
Y_test_means = nan(500, n_categories);
Y_test_sigmas = nan(500, n_categories);

% not chunked test data
X_test_stream = nan(50, 500, n_targets);
Y_test_stream = nan(50, 500, n_categories);
Y_test_stream_means = nan(50, n_categories);
Y_test_stream_sigmas = nan(50, n_categories);

Test_stream_R_HS = nan(50, 5);
Test_stream_L_HS = nan(50, 5);
Test_stream_R_TO = nan(50, 5);
Test_stream_L_TO = nan(50, 5);
Test_stream_id = nan(50);

% val_suj_ind = sort(randsample(length(folders), VAL_N_SUJ));
% val_sub_name = cellfun(@string, {folders(val_suj_ind).name}');
val_sub_name = ["Pilot05", "Pilot07", "Pilot12", "Pilot23-M", ...
    "Pilot30-M", "Pilot33-M"]';
val_sub_id = arrayfun(@(s) str2num(regexp(s,'\d*','Match')), val_sub_name);

% remained = setdiff(1:length(folders), val_suj_ind)';
% test_suj_ind = sort(randsample(remained, TEST_N_SUJ));
% test_sub_name = cellfun(@string, {folders(test_suj_ind).name}');
test_sub_name = ["Pilot09", "Pilot17", "Pilot18-F", "Pilot29-M", ...
    "Pilot32-F", "Pilot35-M"]';
test_sub_id = arrayfun(@(s) str2num(regexp(s,'\d*','Match')), test_sub_name);

cur_train_idx = 1;
cur_val_idx = 1;
cur_test_idx = 1;
cur_stream_idx = 1;
for i = 1: length(folders)
    % find all strides
    folder_name = folders(i).name;
    fprintf('Processing %s ...\n', folder_name)
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
            source = sources(k);
            
            % make EMG sinal have a uniform scale
            emg_signal = ex_data.(source);
            data_mat(:, k) = emg_signal / EMG_SCALE;
        end
        
        % get target label and normalize it
        ank_ang = ex_data.(labels);
        HS_indices = ex_data.(HS);
        label_mean = mean(ank_ang(HS_indices(1):HS_indices(2)));
        label_sigma = sqrt(var(ank_ang(HS_indices(1):HS_indices(2))));
        label_mat = (ank_ang - label_mean) / label_sigma;
        
        % store every TIME_SIZE chunk as one batch
        % split validation set based on subject
        start_idx = 1;
        while start_idx + TIME_SIZE < t_length
            range = start_idx:start_idx + TIME_SIZE - 1;
            
            if  any(i == val_suj_ind)
                X_val(cur_val_idx, :, :) = data_mat(range, :);
                Y_val(cur_val_idx, :, :) = label_mat(range, :);
                Y_val_means(cur_val_idx, :) = label_mean;
                Y_val_sigmas(cur_val_idx, :) = label_sigma;
                cur_val_idx = cur_val_idx + 1;
            elseif any(i == test_suj_ind)
                X_test(cur_test_idx, :, :) = data_mat(range, :);
                Y_test(cur_test_idx, :, :) = label_mat(range, :);
                Y_test_means(cur_test_idx, :) = label_mean;
                Y_test_sigmas(cur_test_idx, :) = label_sigma;
                cur_test_idx = cur_test_idx + 1;
                
                % not chunked test data
                if start_idx == 1
                    T = size(data_mat, 1);
                    X_test_stream(cur_stream_idx, 1:T, :) = data_mat;
                    Y_test_stream(cur_stream_idx, 1:T, :) = label_mat;
                    
                    Test_stream_R_HS(cur_stream_idx, 1:length(ex_data.R_HS)) = ex_data.R_HS;
                    Test_stream_L_HS(cur_stream_idx, 1:length(ex_data.L_HS)) = ex_data.L_HS;
                    Test_stream_R_TO(cur_stream_idx, 1:length(ex_data.R_TO)) = ex_data.R_TO;
                    Test_stream_L_TO(cur_stream_idx, 1:length(ex_data.L_TO)) = ex_data.L_TO;
                    Test_stream_id(cur_stream_idx) = find(test_sub_name == folder_name);
                    
                    Y_test_stream_means(cur_stream_idx, :) = label_mean;
                    Y_test_stream_sigmas(cur_stream_idx, :) = label_sigma;
                    cur_stream_idx = cur_stream_idx + 1;
                end
            else
                X_train(cur_train_idx, :, :) = data_mat(range, :);
                Y_train(cur_train_idx, :, :) = label_mat(range, :);
                Y_train_means(cur_train_idx, :) = label_mean;
                Y_train_sigmas(cur_train_idx, :) = label_sigma;
                cur_train_idx = cur_train_idx + 1;
            end
            
            start_idx = start_idx + INTERVAL;
        end
    end    
end

% delete pre-assigned nan's
X_train(isnan(X_train)) = [];
X_train = reshape(X_train, [], TIME_SIZE, n_targets);
Y_train(isnan(Y_train)) = [];
Y_train = reshape(Y_train, [], TIME_SIZE, n_categories);
Y_train_means(isnan(Y_train_means)) = [];
Y_train_means = reshape(Y_train_means, [], n_categories);
Y_train_sigmas(isnan(Y_train_sigmas)) = [];
Y_train_sigmas = reshape(Y_train_sigmas, [], n_categories);

X_val(isnan(X_val)) = [];
X_val = reshape(X_val, [], TIME_SIZE, n_targets);
Y_val(isnan(Y_val)) = [];
Y_val = reshape(Y_val, [], TIME_SIZE, n_categories);
Y_val_means(isnan(Y_val_means)) = [];
Y_val_means = reshape(Y_val_means, [], n_categories);
Y_val_sigmas(isnan(Y_val_sigmas)) = [];
Y_val_sigmas = reshape(Y_val_sigmas, [], n_categories);

X_test(isnan(X_test)) = [];
X_test = reshape(X_test, [], TIME_SIZE, n_targets);
Y_test(isnan(Y_test)) = [];
Y_test = reshape(Y_test, [], TIME_SIZE, n_categories);
Y_test_means(isnan(Y_test_means)) = [];
Y_test_means = reshape(Y_test_means, [], n_categories);
Y_test_sigmas(isnan(Y_test_sigmas)) = [];
Y_test_sigmas = reshape(Y_test_sigmas, [], n_categories);

X_test_stream(cur_stream_idx:end, :, :) = [];
Y_test_stream(cur_stream_idx:end, :, :) = [];
Y_test_stream_means(isnan(Y_test_stream_means)) = [];
Y_test_stream_means = reshape(Y_test_stream_means, [], n_categories);
Y_test_stream_sigmas(isnan(Y_test_stream_sigmas)) = [];
Y_test_stream_sigmas = reshape(Y_test_stream_sigmas, [], n_categories);

Test_stream_R_HS(cur_stream_idx:end, :, :) = [];
Test_stream_L_HS(cur_stream_idx:end, :, :) = [];
Test_stream_R_TO(cur_stream_idx:end, :, :) = [];
Test_stream_L_TO(cur_stream_idx:end, :, :) = [];
Test_stream_id(cur_stream_idx:end) = [];

% save data sets
saved_file = fullfile(root_dir, 'SNUH_data_bio_subject_ankle.mat');
save(saved_file, "X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test", ...
    "X_test_stream", "Y_test_stream", "Test_stream_R_HS", "Test_stream_L_HS", ...
    "Test_stream_R_TO", "Test_stream_L_TO", "Test_stream_id", ...
    "val_sub_id", "test_sub_id", "setup", ...
    "Y_train_means", "Y_train_sigmas", "Y_val_means", "Y_val_sigmas", ...
    "Y_test_means", "Y_test_sigmas", "Y_test_stream_means", "Y_test_stream_sigmas")

fprintf('(Train: %d, Val: %d, Test: %d) Saved to %s \n', ...
    size(X_train, 1), size(X_val, 1), size(X_test, 1), saved_file)
fprintf('Val #%s: %s\n', [(1:VAL_N_SUJ)', val_sub_name]')
fprintf('Test #%s: %s\n', [(1:TEST_N_SUJ)', test_sub_name]')

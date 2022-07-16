% This code plot emg data

% root dataset path and check valid folder
root_dir = fullfile('C:', 'Users', 'biomechanics', 'Dropbox', 'SNU_DATASET', 'GAIT');
folders = dir(root_dir);

data_type = 'knee';
saved_file = fullfile(root_dir, ...
    ['SNUH_data_bio_subject_', data_type, '.mat']);
loaded = load(saved_file);

% target emg name, subject id
target_emg = "R_gastrocnemius";
target_id = 3;

emg_name = ["R_vastus_medialis", "R_medial_ham", "R_vastus_lateralis", "R_gastrocnemius", ...
    "L_vastus_medialis", "L_medial_ham", "L_vastus_lateralis", "L_gastrocnemius"];
emg_ind = find(emg_name == target_emg);
id_ind = find(loaded.Test_stream_id == target_id);
emg_data = loaded.X_test_stream(:, :, target_ind);
[n_trial, ] = size(emg_data);

N_pts = 100;
emg_ph = nan(100, N_pts);
step_id = 1;

for i_id = id_ind
    emg_trial = loaded.X_test_stream(i_id, :, emg_ind);
    emg_trial(isnan(emg_trial)) = [];
    
    hs = loaded.Test_stream_R_HS(i_id, :);
    hs(isnan(hs)) = [];
    
    emg_step = emg_trial(hs(1):hs(2));
    emg_ph(step_id, :) = interp1(1:numel(emg_step), ...
        emg_step, linspace(1, numel(emg_step), N_pts));
    step_id = step_id + 1;
end
   
emg_ph(isnan(emg_ph)) = [];
emg_ph = reshape(emg_ph, [], 100);

emg_mean = mean(emg_ph, 1);
emg_std = std(emg_ph, 1);

% plot
figure();
plot(emg_mean, 'Color', '#77AC30', 'LineWidth', 1.5); hold on
emg_mean_std = [emg_mean + emg_std, fliplr(emg_mean - emg_std)];
fill([1:100, 100:-1:1], emg_mean_std, 'b', 'FaceAlpha', 0.6, 'EdgeColor', 'none', ...
    'FaceColor', '#77AC30')

axe = gca;
axe.Color = 'none';
axe.Position(4) = 0.5;
axe.LineWidth = 1.0;
xticks([0, 50, 100])
xticklabels({'', '', ''})
yticks(0:ceil(max(emg_mean)))
yticklabels({'', ''})
ylim([0., 1.2 * max(emg_mean)]);
box off

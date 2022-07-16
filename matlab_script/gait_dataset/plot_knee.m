% This code plot knee angle data

% root dataset path and check valid folder
root_dir = fullfile('C:', 'Users', 'biomechanics', 'Dropbox', 'SNU_DATASET', 'GAIT');
folders = dir(root_dir);

data_type = 'knee';
saved_file = fullfile(root_dir, ['SNUH_data_bio_subject_knee.mat']);
loaded = load(saved_file);

% target subject id
target_id = 3;

id_ind = find(loaded.Test_stream_id == target_id);

N_pts = 100;
angle_ph = nan(100, N_pts);
step_id = 1;

for i_id = id_ind
    angle_trial = loaded.Y_test_stream(i_id, :);
    angle_trial(isnan(angle_trial)) = [];
    
    hs = loaded.Test_stream_R_HS(i_id, :);
    hs(isnan(hs)) = [];
    
    angle_step = angle_trial(hs(1):hs(2));
    angle_ph(step_id, :) = interp1(1:numel(angle_step), ...
        angle_step, linspace(1, numel(angle_step), N_pts));
    step_id = step_id + 1;
end
   
angle_ph(isnan(angle_ph)) = [];
angle_ph = reshape(angle_ph, [], 100);

angle_mean = mean(angle_ph, 1);
angle_std = std(angle_ph, 1);

% plot
figure();
plot(angle_mean, 'Color', '#A2142F', 'LineWidth', 1.5); hold on
emg_mean_std = [angle_mean + angle_std, fliplr(angle_mean - angle_std)];
fill([1:100, 100:-1:1], emg_mean_std, 'b', 'FaceAlpha', 0.6, 'EdgeColor', 'none', ...
    'FaceColor', '#A2142F')

axe = gca;
axe.Color = 'none';
axe.Position(4) = 0.5;
axe.LineWidth = 1.0;
xticks([0, 50, 100])
xticklabels({'', '', ''})
yticks([])
ylim([1.2 * min(angle_mean), 1.2 * max(angle_mean)]);
box off

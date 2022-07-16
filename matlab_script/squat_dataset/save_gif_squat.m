function save_gif_squat(pilot, trial)
% save a gif animation for squat motion of specified subject and trial


root_dir = fullfile('C:', 'Users', 'biomechanics', 'Dropbox', 'SNU_DATASET', 'Squat');
foldername = sprintf('Pilot%02d', pilot);
filename = sprintf('trial%02d.mat', trial);
load(fullfile(root_dir, foldername, filename), 'ex_data');

gif_file = sprintf('trial%02d.gif', trial);
gif_filename = fullfile(root_dir, foldername, gif_file);

% set joints of motion skeleton
% [base(1), root(2), r_hip(3), r_knee(4), r_ankle(5), r_toe(6),
% l_hip(7), l_knee(8), l_ankle(9), l_toe(10), trunk(11)]
parents = [0, 1, 2, 3, 4, 2, 6, 7, 8, 9, 1];
seg_len = [0, 0, ...
    0.5, 0.5, 0.15, 0, ...
    0.5, 0.5, 0.15, 0, ...
    0.3];

n_joints = numel(parents);
skel_offsets = zeros(2, n_joints);
skel_offsets(1, [5, 9]) = seg_len([5, 9]);                  % forward foots
skel_offsets(2, [3, 4, 7, 8]) = - seg_len([3, 4, 7, 8]);    % upright pose
skel_offsets(2, 11) = seg_len(11);
skel_rotations = rotm_from_angle(zeros(1, n_joints));
[skel_positions, ~] = calc_global_positions(skel_rotations, skel_offsets, parents);

% initialize visualizer
fh = figure('Visible', 'off'); hold on
xlim(0.8 * [-1, 1])
ylim([0, 1.6])
axis equal

nodes_h = gobjects(n_joints);
lines_h = gobjects(n_joints);
for i = 1:n_joints
    nodes_h(i) = plot(skel_positions(1, i), skel_positions(2, i), 'o');
    
    if parents(i) ~= 0
        parent_id = parents(i);
        x = [skel_positions(1, i), skel_positions(1, parent_id)];
        y = [skel_positions(2, i), skel_positions(2, parent_id)];
        lines_h(i) = line(x, y);
    end
end

% drawing data
n_frame = size(ex_data.Phase, 1);
fprintf('Saving gif to %s ... \n', gif_filename);

for i = 1:n_frame
    % set global points from data
    angles = [0, ...
        - ex_data.Pelvis_fwd_tilt(i), ...
        ex_data.R_hip_flex_ang(i), ...
        - ex_data.R_knee_flex_ang(i), ...
        ex_data.R_ank_flex_ang(i), ...
        0, ...
        ex_data.L_hip_flex_ang(i), ...
        - ex_data.L_knee_flex_ang(i), ...
        ex_data.L_ank_flex_ang(i), ...
        0, ...
        - 90  - ex_data.Trunk_fwd_tilt(i)];
    skel_ratations = rotm_from_angle(angles);
    [post_positions, id_contact] = calc_global_positions(skel_ratations, skel_offsets, parents);
    
    % preserve contact point in x
    x_diff = post_positions(1, id_contact) - skel_positions(1, id_contact);
    skel_positions = post_positions - [x_diff; 0];
    
    % change the point of nodes and lines
    for j = 1:n_joints
        parent_id = parents(j);
        set(nodes_h(j), 'XData', skel_positions(1, j))
        set(nodes_h(j), 'YData', skel_positions(2, j))
        
        if parent_id ~= 0
            set(lines_h(j), 'XData', [skel_positions(1, j), skel_positions(1, parent_id)])
            set(lines_h(j), 'YData', [skel_positions(2, j), skel_positions(2, parent_id)])
        end
    end
    
    drawnow()
    
    % capture the plot as an image
    frame = getframe(fh);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    
    % write to the gif file
    if i == 1
        imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf);
    elseif mod(i, 4) == 0
        imwrite(imind, cm ,gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 1/30)
    end
end
end


function [positions, id_contact] = calc_global_positions(rotations, offsets, parents)
% calculates joint positions in global
% assumes that one lowest joint is on contact with ground (y = 0)

n_joints = numel(parents);
trans_mats = repmat(eye(3), 1, 1, n_joints);

for i = 1:n_joints
    trans_mats(1:2, :, i) = [rotations(:, :, i), ...
        rotations(:, :, i) * offsets(:, i)];
end

for i = 1:n_joints
    parent_id = parents(i);
    if parent_id == 0
        continue
    end
    
    trans_mats(:, :, i) = trans_mats(:, :, parent_id) * trans_mats(:, :, i);
end

positions = trans_mats(1:2, 3, :) ./ trans_mats(3, 3, :);
positions = squeeze(positions);

[min_y, id_contact] = min(positions(2, :));
positions(2, :) = positions(2, :) - min_y;
end


function rotations = rotm_from_angle(angles)
% constructs 2d rotation matrix from angles in degree

angles = angles * pi / 180;
angles = reshape(angles, 1, []);
rotations = [cos(angles); sin(angles); -sin(angles); cos(angles)];
rotations = reshape(rotations, 2, 2, []);
end
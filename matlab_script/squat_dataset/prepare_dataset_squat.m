% This code loads and processes all squat trials
% It saves train, validataion dataset in root data directory
% Validataion set is splited in terms of time steps

% datset setups (time length, intervals, limits, val set size)
TIME_SIZE = 150;
INTERVAL = 20;
KNEE_RNG = [0, 70];
ANK_RNG = [-10, 30];
VAL_RATIO = 0.2;

% root dataset path
root_dir = fullfile('C:', 'Users', 'biomechanics', 'Dropbox', 'SNU_DATASET', 'Squat');
folders = dir(root_dir);

% stored matrix X(B x T x N), Y(B x T)
targets = ["R_contact", "R_knee_flex_ang", "R_ank_rot_ang", "R_vastus_medialis", ...
    "R_medial_ham", "R_vastus_lateralis", "R_gastrocnemius"];
labels = "Phase";

cur_idx = 1;
n_targets = length(targets);
n_categories = 2;
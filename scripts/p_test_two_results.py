import os
import pickle
from scipy import stats


# shared setting
target = 'Knee'
best_dir = 'best_0'
mse_filename = 'mse_all_trials.pkl'

# load two experiments
logdir0 = os.path.join('/', 'home', 'user', 'ray_results5210')
algo0 = 'LSTM'
exp_date0 = '2021-11-10_18-40-07'
exp_name0 = 'R' + target + 'FromEmg' + algo0 + '_' + exp_date0

logdir1 = os.path.join('/', 'home', 'user', 'ray_results')
algo1 = 'Snail'
exp_date1 = '2021-11-10_22-06-58'
exp_name1 = 'R' + target + 'FromEmg' + algo1 + '_' + exp_date1

with open(os.path.join(logdir0, exp_name0, best_dir, mse_filename), 'rb') as f:
    mse0 = pickle.load(f)

with open(os.path.join(logdir1, exp_name1, best_dir, mse_filename), 'rb') as f:
    mse1 = pickle.load(f)

_, p = stats.ttest_ind(mse0, mse1, equal_var=False)
print(f'p: {p}')

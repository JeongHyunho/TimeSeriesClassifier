import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# seaborn setup
sns.set_theme(style="white")
plt.show()

# target angle
target = 'Knee'

# snail
if target.lower() == 'knee':
    ray_folder0 = 'ray_results'
    exp_date0 = '2021-11-10_22-06-58'
else:
    ray_folder0 = 'ray_results7124'
    exp_date0 = '2021-11-10_16-36-17'
algo0 = 'Snail'
exp_name0 = 'R' + target + 'FromEmg' + algo0 + '_' + exp_date0
logdir0 = os.path.join('/', 'home', 'user', ray_folder0, exp_name0, 'best_0')

# lstm
if target.lower() == 'knee':
    ray_folder1 = 'ray_results7124'
    exp_date1 = '2021-12-13_15-56-44'
else:
    ray_folder1 = 'ray_results5210'
    exp_date1 = '2021-11-11_12-25-22'
algo1 = 'LSTM'
exp_name1 = 'R' + target + 'FromEmg' + algo1 + '_' + exp_date1
logdir1 = os.path.join('/', 'home', 'user', ray_folder1, exp_name1, 'best_0')

for sub_id in range(5):
    grad0 = pd.read_csv(logdir0 + f'/sub{sub_id}_grad.csv')
    grad1 = pd.read_csv(logdir1 + f'/sub{sub_id}_grad.csv')

    grad0['algo'] = ['snail'] * len(grad0)
    grad1['algo'] = ['lstm'] * len(grad0)

    df = grad1.append(grad0, ignore_index=True)

    sns.relplot(data=df, kind='line', x='phase', y='grad', hue='algo', palette='tab10', aspect=2.0)
    plt.xticks([0, 50, 100])
    plt.xlim([0, 100])
    if target.lower() == 'knee':
        plt.yticks(np.linspace(0, 0.4, 5))
        plt.ylim([0., 0.3])
    else:
        plt.yticks(np.linspace(0, 0.5, 6))
        plt.ylim([0., 0.5])
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig(os.path.join(logdir0, f'sub{sub_id}_cf_grad_'), dpi=1200)
    plt.savefig(os.path.join(logdir1, f'sub{sub_id}_cf_grad'), dpi=1200)
    plt.close()

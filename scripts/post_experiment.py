import json
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ray.tune import Analysis
from torch.utils.data import DataLoader

from data.snuh_dataset import SnuhEmgForAngle, SnuhEmgForAngleTestStream
from model.ankle_estimator import LSTMEstimator, SnailEstimator, ExtendedLSTMEstimator


def main(ray_folder, exp_date, target, algo):
    logdir = os.path.join('/', 'home', 'user', ray_folder)
    exp_name = 'R' + target + \
               'FromEmg' + algo + '_' + exp_date
    device = 'cpu'
    loaded = Analysis(os.path.join(logdir, exp_name))

    if algo.lower() == 'snail':
        estimator = SnailEstimator
    elif algo.lower() == 'lstm':
        estimator = LSTMEstimator
    elif algo.lower() == 'extendedlstm':
        estimator = ExtendedLSTMEstimator
    else:
        raise NotImplementedError

    # df_best_5 = loaded.dataframe().sort_values('val_loss').head(5)
    df_best_5 = loaded.dataframe().sort_values('val_loss').head(1)

    test_losses = []
    test_rmse_means = []
    test_rmse_stds = []
    test_nrmse_means = []
    test_nrmse_stds = []
    r_squareds = []

    dataset = SnuhEmgForAngle(motion='gait', target=target, validation=True, device=device)
    test_set = DataLoader(dataset, shuffle=True)

    # seaborn setup
    sns.set_theme(style="white")
    plt.show()

    n_phase_pts = 100
    test_stream = SnuhEmgForAngleTestStream(motion='gait', target=target, device=device)
    for rank, trial_dir in enumerate(df_best_5['logdir']):
        # load
        with open(os.path.join(trial_dir, 'params.json'), 'r') as f:
            trial_config = json.load(f)
        model = estimator.load_from_config(trial_config, os.path.join(trial_dir, 'model.pt'), map_location=device)
        test_loss = model.train_model(-1, test_set, evaluation=True, verbose=True)
        test_losses.append(test_loss)

        model.eval()
        r_squared = []
        test_mse = []
        test_nmse = []
        for id in range(test_stream.n_ids + 1):
            Y_phase = pd.DataFrame()
            X_grad = pd.DataFrame()
            plot_dir = os.path.join(logdir, exp_name, f'best_{rank}')
            os.makedirs(plot_dir, exist_ok=True)

            for t_idx in test_stream.get_trial_indices_by_id(id):
                # load trial data
                X_data, Y_data, HS, _, _ = test_stream[t_idx]
                Y_data = Y_data.cpu().numpy()
                ph_range = np.arange(HS[0], HS[1] + 1)
                n_pts = len(ph_range)
                x_intp = np.linspace(0, n_pts - 1, n_phase_pts)
                with torch.no_grad():
                    Y_pred = model(X_data.unsqueeze(dim=0)).squeeze(dim=0).cpu().numpy()

                # calculate mse
                Y_pred_ang = test_stream.unscale(Y_pred, t_idx, test_stream=True)
                Y_data_ang = test_stream.unscale(Y_data, t_idx, test_stream=True)
                test_mse.append(np.mean((Y_pred_ang - Y_data_ang) ** 2))
                test_nmse.append(test_mse[-1] / (Y_data_ang.max() - Y_data_ang.min()) ** 2)

                # calculate R^2
                pred, data = Y_pred[ph_range], Y_data[ph_range]
                SSres, SStot = np.sum((pred - data) ** 2), np.sum((data - data.mean()) ** 2)
                r_squared.append(1.0 - SSres / SStot)

                # interpolation and save prediction/ data to DataFrame to be plotted
                if len(HS) not in [2, 3]:
                    warnings.warn(f'Not considered HS: {HS}, id: {id}.')
                else:
                    # comparison of prediction and data
                    _Y_pred_ph = np.interp(x_intp, np.arange(n_pts), Y_pred[ph_range])
                    _Y_data_ph = np.interp(x_intp, np.arange(n_pts), Y_data[ph_range])
                    _Y_pred_ang_ph = test_stream.unscale(_Y_pred_ph, t_idx, test_stream=True)
                    _Y_data_ang_ph = test_stream.unscale(_Y_data_ph, t_idx, test_stream=True)

                    _df_data = pd.DataFrame({'phase': np.arange(n_phase_pts), 'type': 'data', 'angle': _Y_data_ang_ph},
                                            index=range(len(Y_phase), len(Y_phase) + n_phase_pts))
                    Y_phase = Y_phase.append(_df_data)
                    _df_pred = pd.DataFrame({'phase': np.arange(n_phase_pts), 'type': 'pred', 'angle': _Y_pred_ang_ph},
                                            index=range(len(Y_phase), len(Y_phase) + n_phase_pts))
                    Y_phase = Y_phase.append(_df_pred)

                    # gradient of input in phase
                    X_grad_on = torch.nn.Parameter(X_data, requires_grad=True)
                    last_ang = model(X_grad_on.unsqueeze(dim=0)).squeeze(dim=0)[ph_range[-1]]
                    last_ang.backward()

                    _X_grad_ph = np.interp(x_intp,
                                           np.arange(n_pts),
                                           torch.norm(X_grad_on.grad[ph_range], dim=-1).cpu().numpy())
                    _df_grad = pd.DataFrame({'phase': np.arange(n_phase_pts), 'grad': _X_grad_ph},
                                            index=range(len(Y_phase), len(Y_phase) + n_phase_pts))
                    X_grad = X_grad.append(_df_grad)

            # save 'pred vs. data' plots (sorted by val_loss, test subjects) and DataFrame
            if algo.lower() == 'lstm':
                sns.relplot(data=Y_phase, kind='line', x='phase', y='angle', hue='type', style='type',
                            palette=['k', 'b'], aspect=2.0)
            else:
                sns.relplot(data=Y_phase, kind='line', x='phase', y='angle', hue='type', style='type',
                            palette=['k', 'darkorange'], aspect=2.0)
            plt.xlim([0, 100])
            plt.xticks([0, 50, 100])
            if target.lower() == 'knee':
                plt.yticks(np.arange(0, Y_phase['angle'].max() + 20, 20))
            else:
                plt.yticks([-5, 0, 5], ['-10', '0', '10'])
                plt.plot([0, 100], [0, 0], c='k', lw=0.5, alpha=0.5)
            # plt.title(f'Subject #{id:02d}')
            plt.xlabel('')
            plt.ylabel('')
            plt.savefig(os.path.join(plot_dir, f'sub{id}_result'), dpi=1200)
            plt.close()
            Y_phase.to_csv(os.path.join(plot_dir, f'sub{id}_result.csv'))

            # save grad plot
            sns.relplot(data=X_grad, kind='line', x='phase', y='grad', palette=['k'], aspect=2.0)
            plt.xticks([0, 50, 100])
            plt.yticks(np.linspace(0, 0.4, 3))
            plt.xlim([0, 100])
            plt.ylim([0., 0.3])
            # plt.title(f'Subject #{id:02d}')
            plt.xlabel('')
            plt.ylabel('')
            plt.savefig(os.path.join(plot_dir, f'sub{id}_grad'), dpi=1200)
            plt.close()
            X_grad.to_csv(os.path.join(plot_dir, f'sub{id}_grad.csv'))

        # save mse (whole trials)
        with open(os.path.join(plot_dir, 'mse_all_trials.pkl'), 'wb') as f:
            pickle.dump(test_mse, f)

        r_squareds.append(np.mean(r_squared))
        test_rmse_means.append(np.mean(np.sqrt(test_mse)))
        test_rmse_stds.append(np.std(np.sqrt(test_mse)))
        test_nrmse_means.append(np.mean(np.sqrt(test_nmse)))
        test_nrmse_stds.append(np.std(np.sqrt(test_nmse)))

    # save statistics of best 5 results
    df_best_5['test_loss'] = test_losses
    df_best_5['test_rmse_mean'] = test_rmse_means
    df_best_5['test_rmse_sigma'] = test_rmse_stds
    df_best_5['test_nrmse_mean'] = test_nrmse_means
    df_best_5['test_nrmse_sigma'] = test_nrmse_stds
    df_best_5['r_squared'] = r_squareds
    print(f'Test_loss: {test_losses} \n'
          f'Test_rmse_mean: {test_rmse_means} \n'
          f'Test_rmse_std: {test_rmse_stds} \n'
          f'Test_nrmse_mean: {test_nrmse_means} \n'
          f'Test_nrmse_std: {test_nrmse_stds} \n'
          f'Test_R2: {r_squareds}')
    df_best_5.to_csv(os.path.join(logdir, exp_name, 'summary_best5.csv'))


if __name__ == '__main__':
    # knee + snail
    # ray_folder = 'ray_results'
    # exp_date = '2021-11-10_22-06-58'
    # target = 'Knee'
    # algo = 'Snail'

    # knee + lstm
    ray_folder = 'ray_results7124'
    exp_date = '2021-12-13_15-56-44'
    target = 'Knee'
    algo = 'LSTM'

    # ankle + snail
    # ray_folder = 'ray_results7124'
    # exp_date = '2021-11-10_16-36-17'
    # target = 'Ankle'
    # algo = 'Snail'

    # ankle + lstm
    # ray_folder = 'ray_results5210'
    # exp_date = '2021-11-11_12-25-22'
    # target = 'Ankle'
    # algo = 'LSTM'

    main(ray_folder, exp_date, target, algo)

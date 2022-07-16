import json
import os.path
import time
from collections import defaultdict

import numpy as np

import torch
from ray.tune import Analysis

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from models.gait_phase_classifier import CNNClassifier
from torchsummary import summary_string

from data import EitEmgGaitDetection, EitEmgGaitPhaseDataset
from models.gait_detector import LSTMDetector, CNNDetector

sns.set_theme(style='white')
plt.show()


def main(exp_folder, classifier):
    loaded = Analysis(exp_folder)
    loaded.get_best_logdir('val_loss', 'min')
    df_best_10 = loaded.dataframe().sort_values('val_loss').head(10)

    dataset = EitEmgGaitPhaseDataset(test=True)

    total_acc = []
    cases_acc = defaultdict(list)
    fps = []
    delay_mean = []
    delay_std = []

    for rank, trial_dir in enumerate(df_best_10['logdir']):
        with open(os.path.join(trial_dir, 'params.json'), 'r') as f:
            trial_config = json.load(f)

        model = torch.load(
            os.path.join(trial_dir, 'best_model.pt'),
            map_location='cpu',
        )
        # 총/ 조건 별 정확도 계산
        total_acc.append(model.calc_acc(dataset.samples, dataset.labels))
        for c in range(dataset.n_categories):
            idx = (torch.argmax(dataset.labels, -1) == c)
            acc = model.calc_acc(dataset.samples[idx], dataset.labels[idx])
            cases_acc[c].append(acc)

        # confusion matrix 계산 및 저장
        plot_dir = os.path.join(exp_folder, f'best_{rank}')
        os.makedirs(plot_dir, exist_ok=True)

        p_pred = model.forward(dataset.samples)
        y_pred = torch.argmax(p_pred, -1)
        conf_mat = confusion_matrix(torch.argmax(dataset.labels, -1), y_pred.flatten())
        conf_mat = conf_mat / np.sum(conf_mat, -1, keepdims=True)

        fig = plt.figure()
        plt.set_cmap('Greys_r')
        ax = fig.gca()
        cax = ax.matshow(conf_mat)
        cax.set_clim(vmin=0., vmax=1.)
        fig.colorbar(cax)

        ax.set_xticklabels([''] + dataset.categories)
        ax.set_yticklabels([''] + dataset.categories)

        for idx_r, row in enumerate(conf_mat):
            for idx_c, el in enumerate(row):
                text_c = np.ones(3) if el < 0.5 else np.zeros(3)
                ax.text(idx_c, idx_r, f'{100 * el:.1f}',
                        va='center', ha='center', c=text_c, size='x-large')

        fig.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
        plt.close(fig)

        # 전환 구간 plot
        model.eval()
        win_len = 20
        trans_pred_up = []
        trans_pred_down = []
        frame_diff = []
        for x_stream, y_stream in zip(dataset.X_stream, dataset.Y_stream):
            x_batch = []
            y_batch = []
            n_frames = len(x_stream)
            for t_idx in range(n_frames):
                if t_idx + 80 > n_frames:
                    continue
                win_range = torch.arange(t_idx, t_idx + 80)
                x_batch.append(x_stream[win_range, :])
                y_batch.append(y_stream[win_range[-1]])

            x_batch = torch.stack(x_batch, 0)
            y_batch = torch.stack(y_batch, 0)

            with torch.no_grad():
                pred = model(x_batch)
                p_label = torch.argmax(pred, -1)

            y_diff = torch.diff(y_batch.squeeze(-1))
            p_diff = torch.diff(p_label)
            y_trans = torch.nonzero(y_diff).squeeze(-1)
            p_trans = torch.nonzero(p_diff).squeeze(-1)
            frame_diff.append(torch.abs(y_trans - p_trans[..., None]).flatten().sort()[0][:len(y_trans)])

            trans_indices = torch.nonzero(y_diff).squeeze(-1)
            for trans_idx in trans_indices:
                if trans_idx < win_len or trans_idx + win_len >= len(pred):
                    pass
                else:
                    scope = torch.arange(trans_idx - win_len, trans_idx + win_len + 1)
                    if int(y_diff[trans_idx]) == 1:
                        trans_pred_up.append(p_label[scope])
                    else:
                        trans_pred_down.append(p_label[scope])

        trans_true_up = np.zeros(2 * win_len + 1)
        trans_true_up[win_len + 1:] = 1
        trans_true_down = np.ones(2 * win_len + 1)
        trans_true_down[win_len + 1:] = 0

        xticks = [0, win_len, 2 * win_len]
        xlabels = [
            r'$t_{trans}$' + f' - {1 / 120 * win_len:.1f} sec',
            r'$t_{trans}$',
            r'$t_{trans}$' + f' + {1 / 120 * win_len:.1f} sec',
        ]

        trans_pred_up = torch.stack(trans_pred_up, dim=0)
        trans_pred_down = torch.stack(trans_pred_down, dim=0)

        fh = plt.figure()
        for line in trans_pred_up:
            plt.plot(line, '-b')
        plt.plot(trans_true_up, '--k')
        plt.yticks([0, 1], ['stance', 'swing'], size='x-large')
        plt.xticks(xticks, xlabels, size='x-large')
        plt.tight_layout()
        fh.savefig(os.path.join(plot_dir, 'p_label_up.png'))
        # plt.show()
        plt.close(fh)

        fh = plt.figure()
        for line in trans_pred_down:
            plt.plot(line, '-b')
        plt.plot(trans_true_down, '--k')
        plt.yticks([0, 1], ['stance', 'swing'], size='x-large')
        plt.xticks(xticks, xlabels, size='x-large')
        plt.tight_layout()
        fh.savefig(os.path.join(plot_dir, 'p_label_down.png'))
        # plt.show()
        plt.close(fh)

        # delay 계산
        delay = torch.cat(frame_diff).float()
        delay_mean.append(torch.mean(delay).item() / 120)
        delay_std.append(torch.std(delay).item() / 120)

        # fps 계산
        n_samples = 100
        samples = dataset.samples[torch.randint(high=len(dataset.samples), size=(n_samples,))]

        with torch.no_grad():
            model.eval()
            start_time = time.time()
            for sample in samples:
                model.forward(sample.unsqueeze(0))
            time_per_t = (time.time() - start_time) / samples.shape[0]
        fps.append(1 / time_per_t)

        # model summary 저장
        model_summary, _ = summary_string(model, (80, 72), -1, device='cpu')
        with open(os.path.join(plot_dir, 'summary.txt'), 'w') as f:
            print(model_summary, file=f)

    df_best_10['test_acc'] = total_acc
    print(f'Test accuracy: {total_acc}\n')

    for c in range(dataset.n_categories):
        df_best_10[f'test_acc_{c}'] = cases_acc[c]
        print(f'Test accuracy of {c}: {cases_acc[c]}')

    df_best_10['fps'] = fps
    print(f'Test FPS: {fps}')

    df_best_10['delay_mean'] = delay_mean
    print(f'Delay Mean: {delay_mean}')

    df_best_10['delay_std'] = delay_std
    print(f'Delay Std: {delay_std}')

    df_best_10.to_csv(os.path.join(exp_folder, 'summary_best10.csv'))


if __name__ == '__main__':
    ray_folder = 'ray_results7124'
    exp_date = '2022-02-24_23-57-38'
    model_type = 'CNN'

    if model_type == 'CNN':
        exp_folder = os.path.join('/', 'home', 'user', ray_folder, 'CNNClassifier_' + exp_date)
        classifier = CNNClassifier
    else:
        raise ValueError

    assert os.path.exists(exp_folder)
    main(exp_folder, classifier)

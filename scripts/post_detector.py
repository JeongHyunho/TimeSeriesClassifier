import json
import os.path
from collections import defaultdict

import numpy as np

import torch
from ray.tune import Analysis

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchsummary import summary_string

from data import EitEmgGaitDetection
from models.gait_detector import LSTMDetector, CNNDetector

sns.set_theme(style='white')
plt.show()


def main(exp_folder, detector):
    loaded = Analysis(exp_folder)
    loaded.get_best_logdir('val_loss', 'min')
    df_best_10 = loaded.dataframe().sort_values('val_loss').head(10)

    dataset = EitEmgGaitDetection(test=True)

    total_acc = []
    cases_acc = defaultdict(list)
    fps = []

    for rank, trial_dir in enumerate(df_best_10['logdir']):
        with open(os.path.join(trial_dir, 'params.json'), 'r') as f:
            trial_config = json.load(f)

        try:
            model = detector.load_from_config(
                trial_config,
                os.path.join(trial_dir, 'best_model.pt'),
                map_location='cpu',
            )
        except NotImplementedError:
            model = torch.load(
                os.path.join(trial_dir, 'best_model.pt'),
                map_location='cpu',
            )
        # 총/ 조건 별 정확도 계산
        total_acc.append(model.calc_acc(dataset.samples, dataset.labels, method='vote'))
        for c in range(dataset.n_categories):
            idx = (dataset.labels == c).squeeze(dim=-1)
            acc = model.calc_acc(dataset.samples[idx], dataset.labels[idx], method='vote')
            cases_acc[c].append(acc)

        # confusion matrix 계산 및 저장
        plot_dir = os.path.join(exp_folder, f'best_{rank}')
        os.makedirs(plot_dir, exist_ok=True)

        p_pred = model.forward(dataset.samples)
        if isinstance(model, LSTMDetector):
            y_pred, _ = torch.mode(torch.argmax(p_pred, -1), -1)
        elif isinstance(model, CNNDetector):
            y_pred = torch.argmax(p_pred, -1)
        conf_mat = confusion_matrix(dataset.labels.flatten(), y_pred.flatten())
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
                        va='center', ha='center', fontsize='large', c=text_c)

        fig.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
        plt.close(fig)

        # transition 구간 저장
        if isinstance(model, LSTMDetector):
            model.eval()
            for c in range(dataset.n_categories):
                c_samples = dataset.samples[dataset.labels.squeeze(-1) == c]
                with torch.no_grad():
                    p_pred = model.forward(c_samples)
                p_labels = torch.argmax(p_pred, dim=-1)
                fh = plt.figure()
                plt.plot(p_labels.t())
                plt.title(f'true condition: {c} ({dataset.categories[c]})')
                fh.savefig(os.path.join(plot_dir, f'transition_{c}.png'))
                plt.close(fh)

        # fps 계산
        # n_samples = 20
        # samples = dataset.samples[torch.randint(high=len(dataset.samples), size=(n_samples,))]
        #
        # with torch.no_grad():
        #     model.eval()
        #     start_time = time.time()
        #     for sample in samples:
        #         for t, t_inp in enumerate(sample):
        #             t_inp = t_inp.unsqueeze(dim=0).unsqueeze(dim=0)
        #             model.forward(t_inp) if t == 0 else model.forward(t_inp, model.hc_n)
        #     time_per_t = (time.time() - start_time) / (samples.shape[0] * samples.shape[1])
        # fps.append(1 / time_per_t)

        # model summary 저장
        model_summary, _ = summary_string(model, (150, 72), -1, device='cpu')
        with open(os.path.join(plot_dir, 'summary.txt'), 'w') as f:
            print(model_summary, file=f)

    df_best_10['test_acc'] = total_acc
    print(f'Test accuracy: {total_acc}\n')

    for c in range(dataset.n_categories):
        df_best_10[f'test_acc_{c}'] = cases_acc[c]
        print(f'Test accuracy of {c}: {cases_acc[c]}')

    # df_best_10['fps'] = fps
    # print(f'Test FPS: {fps}')

    df_best_10.to_csv(os.path.join(exp_folder, 'summary_best10.csv'))


if __name__ == '__main__':
    ray_folder = 'ray_results'
    exp_date = '2022-02-22_14-27-38'
    model_type = 'CNN'

    if model_type == 'LSTM':
        exp_folder = os.path.join('/', 'home', 'user', ray_folder, 'LSTMDetector_' + exp_date)
        detector = LSTMDetector
    elif model_type == 'CNN':
        exp_folder = os.path.join('/', 'home', 'user', ray_folder, 'CNNDetector_' + exp_date)
        detector = CNNDetector
    else:
        raise ValueError

    assert os.path.exists(exp_folder)
    main(exp_folder, detector)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from data.snuh_dataset import SnuhGaitPhase, SnuhEmgForAngle
from models.ankle_estimator import LSTMEstimator, ExtendedLSTMEstimator, SnailEstimator
from models.phase_classifiers import LSTMClassifier


def test_model_train():
    dataset = SnuhGaitPhase('trial', normalization=True)
    loader = DataLoader(dataset, shuffle=True, batch_size=128)

    model = LSTMClassifier(input_dim=dataset.sample_dim,
                           output_dim=dataset.label_dim,
                           n_out_layers=1,
                           device='cuda')

    for epoch in range(10):
        model.train_classifier(epoch, loader)


def test_val_conf_mat():
    dataset = SnuhGaitPhase('trial', validation=True)
    model = LSTMClassifier(input_dim=dataset.sample_dim,
                           output_dim=dataset.label_dim,
                           device='cpu')

    val_samples = model.fit_data(dataset.samples)
    val_labels = model.fit_data(dataset.labels)
    p_pred = model.forward(val_samples)
    y_pred = torch.argmax(p_pred, -1).flatten()
    y_true = torch.argmax(val_labels, -1).flatten()
    conf_mat = confusion_matrix(y_true, y_pred)
    conf_mat = conf_mat / np.sum(conf_mat, -1)[..., np.newaxis]

    fig = plt.figure()
    ax = fig.gca()
    cax = ax.matshow(conf_mat)
    cax.set_clim(vmin=0., vmax=1.)
    fig.colorbar(cax)

    cats = ['R Stance', 'L Swing', 'L Stance', 'R Swing']
    ax.set_xticklabels([''] + cats)
    ax.set_yticklabels([''] + cats)

    for idx_r, row in enumerate(conf_mat):
        for idx_c, el in enumerate(row):
            ax.text(idx_c, idx_r, f'{el:.2f}',
                    va='center', ha='center', fontsize='large')

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.array(buf)

    plt.close(fig)
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def test_emg_model_train():
    dataset = SnuhEmgForAngle(motion='gait', target='ankle', device='cuda')
    loader = DataLoader(dataset, shuffle=True, batch_size=128)

    model = LSTMEstimator(input_dim=dataset.sample_dim,
                          output_dim=dataset.label_dim,
                          feature_dim=20,
                          n_lstm_layers=2,
                          pre_layers=[64, 64],
                          post_layers=[64, 64],
                          layer_norm=True,
                          device='cuda')

    # for epoch in range(10):
    #     model.train_model(epoch, loader)

    torch.save(model.state_dict(), 'model.pt')
    state_dict = torch.load('./model.pt')
    model.load_state_dict(state_dict)


def test_extended_emg_model_train():
    dataset = SnuhEmgForAngle(motion='gait', target='ankle', device='cuda')
    loader = DataLoader(dataset, shuffle=True, batch_size=128)

    model = ExtendedLSTMEstimator(input_dim=dataset.sample_dim,
                                  output_dim=dataset.label_dim,
                                  n_locals=10,
                                  feature_dim=20,
                                  n_lstm_layers=2,
                                  pre_layers=[64, 64],
                                  post_layers=[64, 64],
                                  device='cuda')

    for epoch in range(10):
        model.train_model(epoch, loader)


def test_snail_emg_model_train():
    dataset = SnuhEmgForAngle(motion='gait', target='ankle', device='cuda')
    loader = DataLoader(dataset, shuffle=True, batch_size=128)

    model = SnailEstimator(input_dim=dataset.sample_dim,
                           output_dim=dataset.label_dim,
                           key_dims=[64, 64],
                           value_dims=[64, 64],
                           filter_dims=[32],
                           target_length=32,
                           layer_norm=True,
                           device='cuda')

    for epoch in range(10):
        model.train_model(epoch, loader, verbose=True)

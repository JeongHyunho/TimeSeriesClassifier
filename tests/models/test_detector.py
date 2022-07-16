import pytest
import torch

from torch.utils.data import DataLoader

from data.eit_emg_dataset import EitEmgGaitDetection
from models.gait_detector import LSTMDetector, CNNDetector


@pytest.mark.parametrize("fc_norm", ['none', 'layer'])
def test_train_lstm_detector(fc_norm):
    dataset = EitEmgGaitDetection(device='cuda')
    val_dataset = EitEmgGaitDetection(validation=True, device='cuda')
    loader = DataLoader(dataset, shuffle=True, batch_size=32)

    model = LSTMDetector(input_dim=dataset.sample_dim,
                         output_dim=dataset.n_categories,
                         feature_dim=20,
                         n_lstm_layers=2,
                         bidirectional=True,
                         pre_layers=[32],
                         post_layers=[32],
                         fc_norm=fc_norm,
                         device='cuda')

    for epoch in range(10):
        model.train_model(epoch, loader, verbose=True)

    for t, t_inp in enumerate(dataset.samples[0]):
        t_inp = t_inp.unsqueeze(dim=0).unsqueeze(dim=0)
        model.forward(t_inp) if t == 0 else model.forward(t_inp, model.hc_n)

    acc = model.calc_acc(val_dataset.samples, val_dataset.labels, method='vote')
    print(f'acc: {acc}')

    torch.save(model, 'model.pt')
    del model
    model = torch.load('./model.pt')


def test_train_cnn_detector():
    dataset = EitEmgGaitDetection(device='cuda')
    val_dataset = EitEmgGaitDetection(validation=True, device='cuda')
    loader = DataLoader(dataset, shuffle=True, batch_size=32)

    model = CNNDetector(
        input_width=dataset.win_size,
        input_channels=dataset.sample_dim,
        kernel_sizes=[3, 3, 3],
        n_channels=[72, 72, 72],
        groups=dataset.sample_dim,
        strides=[1, 1, 1],
        paddings=[0, 0, 0],
        fc_layers=[32, 32],
        output_dim=dataset.n_categories,
        normalization_type='batch',
        pool_type='max',
        pool_sizes=[0, 0, 10],
        pool_strides=[0, 0, 10],
        pool_paddings=[0, 0, 0],
    )

    for epoch in range(10):
        model.train_model(epoch, loader, verbose=True)

    acc = model.calc_acc(val_dataset.samples, val_dataset.labels, method='vote')
    print(f'acc: {acc}')

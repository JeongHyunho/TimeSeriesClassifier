from torch.utils.data import DataLoader

from data.eit_emg_phase_dataset import EitEmgGaitPhaseDataset
from models.gait_phase_classifier import CNNClassifier


def test_train_cnn_detector():
    dataset = EitEmgGaitPhaseDataset(device='cuda')
    val_dataset = EitEmgGaitPhaseDataset(validation=True, device='cuda')
    loader = DataLoader(dataset, shuffle=True, batch_size=32)

    model = CNNClassifier(
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

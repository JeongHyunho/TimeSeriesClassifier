import os

import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


cur_dir, _ = os.path.split(__file__)
root_dir = os.path.join(cur_dir, 'Gait_Phase_220218')


class EitEmgGaitPhaseDataset(Dataset):
    def __init__(
            self,
            time_size=80,
            validation=False,
            test=False,
            device='cpu',
    ):
        super().__init__()
        assert not validation or not test, 'validation or test 중 하나만 true 여야 함'

        loaded = loadmat(os.path.join(root_dir, f'Dataset_TIME_SIZE_{time_size}.mat'))

        if validation:
            samples = loaded['X_val']
            labels = loaded['Y_val']
        elif test:
            samples = loaded['X_test']
            labels = loaded['Y_test']

            self.X_stream = [torch.tensor(mat).float().to(device) for mat in loaded['X_test_stream'][0].tolist()]
            self.Y_stream = [torch.tensor(mat).float().to(device) for mat in loaded['Y_test_stream'][0].tolist()]
        else:
            samples = loaded['X_train']
            labels = loaded['Y_train']

        self.samples = torch.tensor(samples).float().to(device)
        self.labels = torch.tensor(labels).long().to(device)

    @property
    def sample_dim(self):
        return self.samples.shape[2]

    @property
    def win_size(self):
        return self.samples.shape[1]

    @property
    def n_categories(self):
        return 2

    @property
    def categories(self):
        return ['stand', 'swing']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

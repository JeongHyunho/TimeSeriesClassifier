import torch.nn.functional as F

from data.base_dataset import BaseDataset


def pros_input_dim_from_signal_type(signal_type: str) -> int:

    if signal_type == 'all':
        input_dim = 10
    elif signal_type == 'emg':
        input_dim = 4
    elif signal_type == 'eim':
        input_dim = 4
    elif signal_type == 'imu':
        input_dim = 2
    elif signal_type == 'bio':
        input_dim = 8
    else:
        raise ValueError(f'{signal_type} not in ["all", "emg", "eim", "imu", "bio"]')

    return input_dim


class ProsDataset(BaseDataset):

    val_ratio = 0.2
    test_ratio = 0.2
    split_seed = 42

    output_dim = 1

    def __init__(
            self,
            log_dir,
            window_size,
            overlap_ratio,
            num_classes,
            validation=False,
            test=False,
            signal_type='all',
            out_prefix='trial',
            device='cpu',
            preprocess='preprocess.json',
    ):
        self.signal_type = signal_type
        if self.signal_type == 'all':
            signal_rng = slice(0, 10)
        elif self.signal_type == 'emg':
            signal_rng = slice(0, 4)
        elif self.signal_type == 'eim':
            signal_rng = slice(4, 8)
        elif self.signal_type == 'imu':
            signal_rng = slice(8, 10)
        elif self.signal_type == 'bio':
            signal_rng = slice(0, 8)
        else:
            raise ValueError(f'{self.signal_type} not in ["all", "emg", "eim", "imu", "bio"]')

        super().__init__(
            log_dir=log_dir,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            signal_type=signal_type,
            signal_rng=signal_rng,
            validation=validation,
            test=test,
            out_prefix=out_prefix,
            preprocess=preprocess,
        )

        xb, yb = self.load_and_split(num_classes)
        yb = yb.long()
        cb_tensor = F.one_hot(yb, num_classes=num_classes)[:, :, 0, :]

        # change device
        self.samples = xb.to(device)
        self.labels = cb_tensor.to(device)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

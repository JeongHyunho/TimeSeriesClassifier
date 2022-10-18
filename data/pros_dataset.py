import torch.nn.functional as F

from data.base_dataset import BaseDataset


class ProsDataset(BaseDataset):

    val_ratio = 0.2
    test_ratio = 0.2
    split_seed = 42

    input_dim = 8
    output_dim = 1

    def __init__(
            self,
            log_dir,
            window_size,
            overlap_ratio,
            validation=False,
            test=False,
            num_classes=None,
            out_prefix='trial',
            device='cpu',
    ):
        super().__init__(
            log_dir=log_dir,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            validation=validation,
            test=test,
            out_prefix=out_prefix,
        )

        xb, yb = self.load_and_split()
        yb = yb[..., 0].long()

        if num_classes:
            cb_tensor = F.one_hot(yb, num_classes=num_classes)
        else:
            cb_tensor = F.one_hot(yb)

        # change device
        self.samples = xb.to(device)
        self.labels = cb_tensor.to(device)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

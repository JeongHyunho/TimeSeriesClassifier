from data.base_dataset import BaseDataset


class ArmCurlDataset(BaseDataset):

    val_ratio = 0.2
    test_ratio = 0.2
    split_seed = 42

    input_dim = 2
    output_dim = 2

    def __init__(
            self,
            log_dir,
            window_size,
            overlap_ratio,
            validation=False,
            test=False,
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

        # change device
        self.samples = xb.to(device)
        self.labels = yb.to(device)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

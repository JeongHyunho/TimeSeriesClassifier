import json
import re
import traceback
import warnings
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List


class BaseDataset(Dataset):
    """ Base dataset class for data collected using tcp """

    test_indices = None
    stand_file = None

    def __init__(
            self,
            log_dir: Path,
            window_size: int,
            overlap_ratio: float,
            signal_type: str,
            signal_rng: slice,
            validation=False,
            test=False,
            out_prefix='trial',
            preprocess='preprocess.json',
    ):
        super().__init__()
        self.log_dir = log_dir
        self.window_size = window_size
        self.overlap_ration = overlap_ratio
        self.signal_type = signal_type
        self.signal_rng = signal_rng
        self.interval = int(window_size * (1 - overlap_ratio))

        assert not validation or not test, ValueError("only one of 'test' and 'validation' can be true")
        self.validation = validation
        self.test = test

        assert self.log_dir.exists(), FileExistsError(f"{self.log_dir} doesn't exist")
        self.trials = list(self.log_dir.rglob(out_prefix + '[0-9]*.csv'))
        self.num_trials = len(self.trials)
        assert self.num_trials > 0, FileExistsError(f"no trials found in {self.log_dir}")

        preproc_file = self.log_dir / preprocess
        if preproc_file.exists():
            self.preproc_config = json.loads(preproc_file.read_text())

            exist_keys = self.preproc_config.keys()
            for trial in self.trials:
                trial_number = re.findall("\d+", trial.name)[0]
                if trial_number in exist_keys and \
                        len(self.preproc_config[trial_number]) % 2 != 0:
                    warnings.warn(f"expected a list of even number of elements for cut indices,"
                                  f" but got {self.preproc_config[trial_number]}, ignored and set to whole range")
                    self.preproc_config[trial_number] = []
        else:
            self.preproc_config = None

    def load_and_split(self, num_classes=None) -> (torch.Tensor, torch.Tensor):
        """ Load csv files and split all data into train/val/test batch tensors """

        # load data from csv files
        inp_list, label_list = [], []

        for trial in self.trials:
            if trial.suffix == '.csv':
                from_csv = np.loadtxt(str(trial), delimiter=',', skiprows=1)
                new_slice = slice(self.signal_rng.start+1, self.signal_rng.stop+1)

                trial_number = re.findall("\d+", trial.name)[0]
                if self.preproc_config and trial_number in self.preproc_config.keys():
                    cut_indices = self.preproc_config[trial_number]
                    cut_indices.insert(0, 0)
                    cut_indices.append(-1)

                    try:
                        inp = []
                        for start, end in zip(cut_indices[::2], cut_indices[1::2]):
                            inp.append(from_csv[start:end, new_slice])
                        inp = np.vstack(inp)
                    except IndexError:
                        warnings.warn(f'index error for preprocessing {trial}, ignored and set to whole range')
                        warnings.warn(traceback.format_exc())
                        inp = from_csv[:, new_slice]
                else:
                    inp = from_csv[:, new_slice]
            else:
                raise NotImplementedError

            label = from_csv[:, -self.output_dim:]
            inp_list.append(inp)
            label_list.append(label)

        # split train/validation/test indices
        num_val_trials = int(self.num_trials * self.val_ratio)
        num_test_trials = int(self.num_trials * self.test_ratio)
        assert num_val_trials > 0, f"too small trials, make trials larger or val_ratio smaller: total {self.num_trials}"
        assert num_test_trials > 0, f"too small trials, make trials larger or test_ratio smaller: total {self.num_trials}"
        rng = np.random.default_rng(self.split_seed)
        val_indices = rng.choice(np.arange(self.num_trials), num_val_trials, replace=False)
        test_indices = rng.choice(np.setdiff1d(np.arange(self.num_trials), val_indices), num_test_trials, replace=False)
        train_indices = np.setdiff1d(np.arange(self.num_trials), np.hstack([val_indices, test_indices]))
        self.test_indices = test_indices

        # standardization parameters from train trials
        # save parameters
        self.stand_file = Path(self.log_dir).joinpath(f'stand_{self.signal_type}.json')
        if self.stand_file.exists():
            s_dict = json.loads(self.stand_file.read_text())
            inp_mean = np.array(s_dict['mean'])
            inp_std = np.array(s_dict['std'])

        else:
            train_array = np.vstack([inp_list[idx] for idx in train_indices])
            inp_mean = np.mean(train_array, axis=0)
            inp_std = np.std(train_array, axis=0)
            s_dict = json.dumps({'mean': inp_mean.tolist(), 'std': inp_std.tolist()})
            self.stand_file.write_text(s_dict)

        # make batch data by cropping windows
        if self.validation:
            indices = val_indices
        elif self.test:
            indices = test_indices
        else:
            indices = train_indices

        x_list = [inp_list[idx] for idx in indices]
        y_list = [label_list[idx] for idx in indices]

        xb_list, yb_list = [], []
        for x, y in zip(x_list, y_list):
            time_length = len(x)
            if time_length < self.window_size:
                warnings.warn(f'this trial has too short time-length: {time_length} < {self.window_size}')
                continue

            start_t = 0
            while start_t + self.window_size <= time_length:
                xb_list.append(x[start_t:start_t + self.window_size, :])
                yb_list.append(y[start_t:start_t + self.window_size])
                start_t += self.interval

        xb = (np.stack(xb_list, axis=0) - inp_mean) / (inp_std + 1e-6)
        yb = np.stack(yb_list, axis=0)

        # change array to torch tensor
        return torch.FloatTensor(xb), torch.FloatTensor(yb)

    @property
    def val_ratio(self):
        raise NotImplementedError

    @property
    def test_ratio(self):
        raise NotImplementedError

    @property
    def split_seed(self):
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

    def get_test_stream(self, device='cpu') -> (List[torch.Tensor], List[torch.Tensor]):
        assert self.test_indices is not None, "run load_and_split first"

        # load data from csv files
        x_list, y_list = [], []

        for trial in self.trials:
            try:
                from_csv = np.loadtxt(str(trial), delimiter=',', skiprows=0)
                x_list.append(from_csv[:, self.signal_rng])
            except ValueError:
                from_csv = np.loadtxt(str(trial), delimiter=',', skiprows=1)
                new_slice = slice(self.signal_rng.start+1, self.signal_rng.stop+1)
                x_list.append(from_csv[:, new_slice])
            y_list.append(from_csv[:, -self.output_dim:])

        x_list = [x_list[idx] for idx in self.test_indices]
        y_list = [y_list[idx] for idx in self.test_indices]

        # standardization
        s_dict = json.loads(self.stand_file.read_text())
        x_mean = np.array(s_dict['mean'])
        x_std = np.array(s_dict['std'])
        x_list = [(x - x_mean) / (x_std + 1e-6) for x in x_list]

        # to tensor
        xt_list = [torch.FloatTensor(x).to(device) for x in x_list]
        yt_list = [torch.FloatTensor(y).to(device) for y in y_list]

        return xt_list, yt_list

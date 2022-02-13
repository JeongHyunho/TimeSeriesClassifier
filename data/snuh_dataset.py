import os
import getpass
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


if os.name == 'nt':
    snuh_root = 'C:\\' + os.path.join('Users', 'biomechanics', 'Dropbox', 'SNU_DATASET')
else:
    snuh_root = os.path.join('/', 'home', getpass.getuser(), 'Dropbox', 'SNU_DATASET')
snuh_gait_root = os.path.join(snuh_root, 'Gait')
snuh_total = os.path.join(snuh_gait_root, 'SNUH_data_total.mat')
snuh_trial = os.path.join(snuh_gait_root, 'SNUH_data_trial.mat')
snuh_subject = os.path.join(snuh_gait_root, 'SNUH_data_subject.mat')
snuh_amp_trial = os.path.join(snuh_gait_root, 'SNUH_data_amputee_trial.mat')
snuh_amp_subject = os.path.join(snuh_gait_root, 'SNUH_data_amputee_subject.mat')
snuh_mech_trial = os.path.join(snuh_gait_root, 'SNUH_data_mech_trial.mat')
snuh_mech_subject = os.path.join(snuh_gait_root, 'SNUH_data_mech_subject.mat')
snuh_bio_trial = os.path.join(snuh_gait_root, 'SNUH_data_bio_trial.mat')
snuh_bio_subject = os.path.join(snuh_gait_root, 'SNUH_data_bio_subject.mat')


class SnuhGaitPhase(Dataset):
    def __init__(self, split_type='total', validation=False, normalization=False, eps=1e-4):
        self.split_type = split_type
        self.normalization = normalization
        self.eps = 1e-4

        if split_type == 'total':
            loaded_mat = loadmat(snuh_total)
        elif split_type == 'trial':
            loaded_mat = loadmat(snuh_trial)
        elif split_type == 'subject':
            loaded_mat = loadmat(snuh_subject)
        elif split_type == 'amputee_trial':
            loaded_mat = loadmat(snuh_amp_trial)
        elif split_type == 'amputee_subject':
            loaded_mat = loadmat(snuh_amp_subject)
        elif split_type == 'mech_trial':
            loaded_mat = loadmat(snuh_mech_trial)
        elif split_type == 'mech_subject':
            loaded_mat = loadmat(snuh_mech_subject)
        elif split_type == 'bio_trial':
            loaded_mat = loadmat(snuh_bio_trial)
        elif split_type == 'bio_subject':
            loaded_mat = loadmat(snuh_bio_subject)
        else:
            raise Exception(f"{split_type} is not allowd for split type.")

        if validation:
            self.samples = loaded_mat['X_val']
            self.labels = loaded_mat['Y_val']
        else:
            self.samples = loaded_mat['X_train']
            self.labels = loaded_mat['Y_train']

        if normalization:
            x_data = np.concatenate([loaded_mat['X_val'], loaded_mat['X_train']])
            self.x_mean = np.mean(x_data.reshape(-1, self.sample_dim), axis=0)
            self.x_std = np.std(x_data.reshape(-1, self.sample_dim), axis=0)

    @property
    def sample_dim(self):
        return self.samples.shape[2]

    @property
    def label_dim(self):
        return self.labels.shape[2]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        if self.normalization:
            return (self.samples[idx] - self.x_mean) / (self.x_std + self.eps), self.labels[idx]
        else:
            return self.samples[idx], self.labels[idx]


def _struct_to_dict(array):
    out_dict = {}
    for k, v in zip(array[::2], array[1::2]):
        out_dict[k.ravel()[0]] = v.ravel() if len(v.ravel()) > 1 else v.ravel()[0]

    return out_dict


class SnuhEmgForAngle(Dataset):
    def __init__(self, target='ankle', motion='gait', validation=False, test=False, device='cpu'):
        if motion.lower() == 'gait':
            if target.lower() == 'ankle':
                path_to_file = os.path.join(snuh_root, 'Gait', 'SNUH_data_bio_subject_ankle')
                loaded_mat = loadmat(path_to_file)
            elif target.lower() == 'knee':
                path_to_file = os.path.join(snuh_root, 'Gait', 'SNUH_data_bio_subject_knee')
                loaded_mat = loadmat(path_to_file)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if validation:
            self.samples = torch.tensor(loaded_mat['X_val']).float().to(device)
            self.labels = torch.tensor(loaded_mat['Y_val']).float().to(device)
        elif test:
            self.samples = torch.tensor(loaded_mat['X_test']).float().to(device)
            self.labels = torch.tensor(loaded_mat['Y_test']).float().to(device)
        else:
            self.samples = torch.tensor(loaded_mat['X_train']).float().to(device)
            self.labels = torch.tensor(loaded_mat['Y_train']).float().to(device)

        self.setup = _struct_to_dict(loaded_mat['setup'][0])

    @property
    def sample_dim(self):
        return self.samples.shape[2]

    @property
    def label_dim(self):
        return 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class SnuhEmgForAngleTestStream(Dataset):
    def __init__(self, target='ankle', motion='gait', device='cpu'):
        if motion.lower() == 'gait':
            if target.lower() == 'ankle':
                path_to_file = os.path.join(snuh_root, 'Gait', 'SNUH_data_bio_subject_ankle')
                loaded_mat = loadmat(path_to_file)
            elif target.lower() == 'knee':
                path_to_file = os.path.join(snuh_root, 'Gait', 'SNUH_data_bio_subject_knee')
                loaded_mat = loadmat(path_to_file)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.device = device

        self.X_data = loaded_mat['X_test_stream'].astype('f')
        self.Y_data = loaded_mat['Y_test_stream'].astype('f')
        self.HS = loaded_mat['Test_stream_R_HS']
        self.TO = loaded_mat['Test_stream_R_HS']
        self.id = loaded_mat['Test_stream_id'].ravel() - 1      # for 0-based indexing
        self.setup = _struct_to_dict(loaded_mat['setup'][0])

        self.Y_test_stream_means = loaded_mat['Y_test_stream_means'].ravel()
        self.Y_test_stream_sigmas = loaded_mat['Y_test_stream_sigmas'].ravel()

    @property
    def n_ids(self):
        return self.id[-1]

    @property
    def n_trials(self):
        return self.Y_test_stream_means.shape[0]

    def get_trial_indices_by_id(self, id):
        return np.where(self.id == id)[0]

    def unscale(self, np_array, idx, test_stream=True):
        if test_stream:
            mean = self.Y_test_stream_means[idx]
            sigma = self.Y_test_stream_sigmas[idx]
            return sigma * np_array + mean
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        valid_length = np.argmax(np.isnan(self.X_data[idx][:, 0]))
        assert valid_length > 0

        X_data = torch.tensor(self.X_data[idx][:valid_length, :]).to(self.device)
        Y_data = torch.tensor(self.Y_data[idx][:valid_length]).to(self.device)

        _HS = self.HS[idx]
        HS = _HS[~ np.isnan(_HS)].astype('i')
        _TO = self.TO[idx]
        TO = _TO[~ np.isnan(_TO)].astype('i')
        id = self.id[idx].astype('i')

        return X_data, Y_data, HS, TO, id

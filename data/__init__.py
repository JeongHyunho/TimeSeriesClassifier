import itertools

from torch.utils.data import DataLoader

from data.armcurl_dataset import ArmCurlDataset
from data.eit_emg_dataset import EitEmgGaitDetection
from data.eit_emg_phase_dataset import EitEmgGaitPhaseDataset
from data.pros_dataset import ProsDataset
from data.snuh_dataset import SnuhEmgForAngle


def load_estimator_data(batch_size, target, device='cuda'):
    dataset = SnuhEmgForAngle(motion='gait', target=target, device=device)
    train_set = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    dataset = SnuhEmgForAngle(motion='gait', target=target, validation=True, device=device)
    val_set = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    dataset = SnuhEmgForAngle(motion='gait', target=target, validation=True, device=device)
    test_set = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return train_set, val_set, test_set


def load_detector_data(batch_size, device='cuda'):
    dataset = EitEmgGaitDetection(device=device)
    train_set = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    dataset = EitEmgGaitDetection(validation=True, device=device)
    val_set = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    dataset = EitEmgGaitDetection(test=True, device=device)
    test_set = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return train_set, val_set, test_set


def load_classifier_data(batch_size, device='cuda'):
    dataset = EitEmgGaitPhaseDataset(device=device)
    train_set = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    dataset = EitEmgGaitPhaseDataset(validation=True, device=device)
    val_set = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    dataset = EitEmgGaitPhaseDataset(test=True, device=device)
    test_set = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return train_set, val_set, test_set


def get_ordered_combinations(dim_list, n):
    ordered = []

    for subset in itertools.product(*([dim_list] * len(dim_list))):
        checked = sorted(subset[:n])
        if checked in ordered:
            pass
        else:
            ordered += [list(checked)]

    return ordered


def load_prosthesis_loaders(batch_size, log_dir, window_size, overlap_ratio, num_classes, signal_type,
                            device='cuda') -> (DataLoader, DataLoader, DataLoader):
    ds_kwargs = {'log_dir': log_dir, 'window_size': window_size, 'overlap_ratio': overlap_ratio,
                 'num_classes': num_classes, 'signal_type': signal_type, 'device': device}
    dl_kwargs = {'shuffle': True, 'batch_size': batch_size}

    train_ds = ProsDataset(**ds_kwargs)
    train_dl = DataLoader(train_ds, **dl_kwargs)

    val_ds = ProsDataset(**ds_kwargs, validation=True)
    val_dl = DataLoader(val_ds, **dl_kwargs)

    test_ds = ProsDataset(**ds_kwargs, test=True)
    test_dl = DataLoader(test_ds, **dl_kwargs)

    return train_dl, val_dl, test_dl


def load_armcurl_loaders(batch_size, log_dir, window_size, overlap_ratio, device='cuda')\
        -> (DataLoader, DataLoader, DataLoader):
    ds_kwargs = {'log_dir': log_dir, 'window_size': window_size, 'overlap_ratio': overlap_ratio, 'device': device}
    dl_kwargs = {'shuffle': True, 'batch_size': batch_size}

    train_ds = ArmCurlDataset(**ds_kwargs)
    train_dl = DataLoader(train_ds, **dl_kwargs)

    val_ds = ArmCurlDataset(**ds_kwargs, validation=True)
    val_dl = DataLoader(val_ds, **dl_kwargs)

    test_ds = ArmCurlDataset(**ds_kwargs, test=True)
    test_dl = DataLoader(test_ds, **dl_kwargs)

    return train_dl, val_dl, test_dl

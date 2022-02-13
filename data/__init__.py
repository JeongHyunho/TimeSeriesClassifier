import itertools

from torch.utils.data import DataLoader

from data.snuh_dataset import SnuhEmgForAngle


def load_data(batch_size, target, device='cuda'):
    dataset = SnuhEmgForAngle(motion='gait', target=target, device=device)
    train_set = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    dataset = SnuhEmgForAngle(motion='gait', target=target, validation=True, device=device)
    val_set = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    dataset = SnuhEmgForAngle(motion='gait', target=target, validation=True, device=device)
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

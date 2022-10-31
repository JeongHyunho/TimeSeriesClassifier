import json
from pathlib import Path

import pytest

import numpy as np


@pytest.fixture(scope='session')
def time_length() -> int:
    return 500


@pytest.fixture(scope='session')
def input_dim(signal_type) -> int:
    if signal_type == 'all':
        return 8
    elif signal_type == 'emg':
        return 4
    elif signal_type == 'eim':
        return 4


@pytest.fixture(scope='session')
def signal_dim(signal_type) -> int:
    return 8


@pytest.fixture(scope='session')
def output_dim() -> int:
    return 5


@pytest.fixture(scope='session')
def batch_size() -> int:
    return 4


@pytest.fixture(scope='session')
def overlap_ratio() -> float:
    return 0.2


@pytest.fixture(scope='session')
def stream_data(time_length, signal_dim) -> np.ndarray:
    signal = np.random.randn(signal_dim, time_length)
    footswitch = np.cos(np.arange(time_length) * 5 * 2 * np.pi / time_length) > 0.
    angle = np.random.randn(2, time_length)
    is_terminal = np.hstack([np.ones(time_length-1), 0])
    speed = int(np.random.randn() > 0.) * np.ones(time_length)

    phase = []
    last_idx = 0
    trs = np.sort(np.random.choice(time_length, 4, replace=False))
    for p, idx in enumerate(trs):
        phase = np.hstack([phase, p * np.ones(idx - last_idx)])
        last_idx = idx
    phase = np.hstack([phase, 4 * np.ones(time_length - last_idx)])

    data = np.vstack([signal, *[[footswitch] * 3], angle, is_terminal, speed, phase]).T

    return data


@pytest.fixture(scope='session')
def train_config(signal_type, time_length) -> dict:
    json_file = Path(__file__).parent.joinpath('../../exp_scripts/pros_train_config.json')
    assert json_file.exists(), FileNotFoundError(f"{json_file} doesn't exist")
    config = json.loads(json_file.read_text())

    # test setup
    config['signal_type'] = signal_type
    config['num_samples'] = 5
    config['epoch'] = 10

    return config

import json
from pathlib import Path

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--signal_type", type=str, choices=['all'], default='all',
                     help='type of signal, it determines input_dim')


@pytest.fixture(scope="session")
def signal_type(pytestconfig):
    return pytestconfig.getoption("signal_type")


@pytest.fixture(scope='session')
def time_length() -> int:
    return 500


@pytest.fixture(scope='session')
def input_dim(signal_type) -> int:
    if signal_type == 'all':
        return 3


@pytest.fixture(scope='session')
def output_dim() -> int:
    return 2


@pytest.fixture(scope='session')
def batch_size() -> int:
    return 4


@pytest.fixture(scope='session')
def overlap_ratio() -> float:
    return 0.2


@pytest.fixture(scope='session')
def stream_data(time_length, input_dim) -> np.ndarray:
    signal = np.random.randn(input_dim, 10*time_length)
    angle = np.random.randn(1, 10*time_length)
    torque = np.random.randn(1, 10*time_length)
    is_terminal = np.hstack([np.ones(10*time_length-1), 0])
    data = np.vstack([signal, angle, torque, is_terminal]).T

    return data


@pytest.fixture(scope='session')
def train_config(overlap_ratio, signal_type, use_gpu) -> dict:
    json_file = Path(__file__).parent.joinpath('../../exp_scripts/armcurl_train_config.json')
    assert json_file.exists(), FileNotFoundError(f"{json_file} doesn't exist")
    config = json.loads(json_file.read_text())

    # test setup
    config['overlap_ratio'] = overlap_ratio
    config['signal_type'] = signal_type
    config['device'] = 'cuda' if use_gpu else 'cpu'
    config['num_samples'] = 5
    config['epoch'] = 10

    return config

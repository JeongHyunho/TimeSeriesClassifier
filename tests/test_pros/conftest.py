import json
from pathlib import Path

import pytest

import numpy as np


@pytest.fixture(scope='session')
def time_length() -> int:
    return 200


@pytest.fixture(scope='session')
def input_dim() -> int:
    return 8


@pytest.fixture(scope='session')
def output_dim() -> int:
    return 6


@pytest.fixture(scope='session')
def buffer_config() -> dict:
    json_file = Path(__file__).parent.joinpath('../../exp_scripts/pros_log_config.json')
    assert json_file.exists(), FileNotFoundError(f"{json_file} doesn't exist")

    return json.loads(json_file.read_text())


@pytest.fixture(scope='session')
def stream_data(time_length, input_dim) -> np.ndarray:
    signal = np.random.randn(input_dim, time_length)
    footswitch = np.cos(np.arange(time_length) * 5 * 2 * np.pi/ time_length) > 0.
    angle = np.random.randn(2, time_length)
    is_terminal = np.hstack([np.ones(time_length-1), 0])
    data = np.vstack([signal, *[[footswitch] * 3], angle, is_terminal]).T

    return data


@pytest.fixture(scope='session')
def train_config() -> dict:
    json_file = Path(__file__).parent.joinpath('../../exp_scripts/pros_train_config.json')
    assert json_file.exists(), FileNotFoundError(f"{json_file} doesn't exist")
    config = json.loads(json_file.read_text())

    # test setup
    config['num_samples'] = 5
    config['epoch'] = 10

    return config

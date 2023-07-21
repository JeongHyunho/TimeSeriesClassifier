import json
from pathlib import Path

import pytest


@pytest.fixture(scope='session')
def time_length() -> int:
    return 80


@pytest.fixture(scope='session')
def input_dim() -> int:
    return 4


@pytest.fixture(scope='session')
def output_dim() -> int:
    return 1


@pytest.fixture(scope='session')
def batch_size() -> int:
    return 4


@pytest.fixture(scope='session')
def overlap_ratio() -> float:
    return 0.2


@pytest.fixture(scope='session')
def data_path() -> Path:
    return Path(__file__).parent.joinpath('../../output/golf_2307/log')


@pytest.fixture(scope='session')
def train_config(overlap_ratio, use_gpu) -> dict:
    json_file = Path(__file__).parent.joinpath('../../exp_scripts/golf_train_config.json')
    assert json_file.exists(), FileNotFoundError(f"{json_file} doesn't exist")
    config = json.loads(json_file.read_text())

    # test setup
    config['overlap_ratio'] = overlap_ratio
    config['device'] = 'cuda' if use_gpu else 'cpu'
    config['num_samples'] = 5
    config['epoch'] = 10

    return config

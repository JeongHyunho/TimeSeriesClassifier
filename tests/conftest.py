import json
import os
from pathlib import Path

import pytest
import logging


os.environ["MKL_THREADING_LAYER"] = "GNU"

logging.basicConfig(
    format='[%(name)s] %(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    datefmt='%y/%m/%d %H:%M:%S',
)


def pytest_addoption(parser):
    parser.addoption("--use_gpu", action="store_true", default=False, help='use gpu or not')
    parser.addoption("--log_debug", action="store_true", default=False, help='set debug logging')


@pytest.fixture(scope="session")
def use_gpu(pytestconfig):
    return pytestconfig.getoption("use_gpu")


@pytest.fixture(scope="session")
def log_debug(pytestconfig):
    return pytestconfig.getoption("log_debug")


@pytest.fixture(scope="session")
def address() -> str:
    return 'localhost'


@pytest.fixture(scope="session")
def recv_port() -> int:
    return 9105


@pytest.fixture(scope="session")
def send_port() -> int:
    return 9106


@pytest.fixture(scope='package')
def cluster_config() -> dict:
    json_file = Path(__file__).parent.joinpath('../exp_scripts/cluster_config.json')
    assert json_file.exists(), FileNotFoundError(f"{json_file} doesn't exist")
    config = json.loads(json_file.read_text())

    # test setup
    config['sbatch_kwargs']['partition'] = 'debug'

    return config

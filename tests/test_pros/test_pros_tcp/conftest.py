import json
import logging
import textwrap

import pytest
import socket
import struct

from typing import Callable, List
from pathlib import Path

import torch

from core.util import sample_config, dot_map_dict_to_nested_dict
from models.gait_phase_classifier import CNNClassifier, LSTMClassifier


@pytest.fixture(scope='package')
def data_len(stream_data) -> int:
    return stream_data.shape[-1]


@pytest.fixture(scope='package')
def control_len() -> int:
    return 1


@pytest.fixture(scope='package')
def create_client_fn(stream_data, address, port, data_len, control_len) -> Callable:
    def _create_client(type='log'):
        logger = logging.getLogger('test.client')

        cli_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cli_sock.connect((address, port))

        for i, array in enumerate(stream_data):
            data_send = struct.pack(f'>{data_len}f', *array)
            cli_sock.send(data_send)

            if type == 'log':
                pass
            elif type == 'control':
                data_recv = cli_sock.recv(4 * data_len)
                signal = struct.unpack(f'>{control_len}i', data_recv)
                logger.debug(f'#{i} received: {signal}')
            else:
                raise ValueError(f"type should be 'log' or 'control', but got {type}")

        cli_sock.close()

    return _create_client


@pytest.fixture
def basic_train_config_json(tmp_path) -> Path:
    config = """
    {
        "task": ["sleep", "count"],
        "sleep.time": [5, 6],
        "count.max_number": [10, 20, 30],
        
        "num_samples": 5
    }
    """

    json_file = tmp_path.joinpath('train_config.json')
    json_file.write_text(config)

    return json_file


@pytest.fixture
def basic_cluster_config_json(tmp_path) -> Path:
    config = """
    {
        "report": "report.json",
        "sbatch_kwargs": {
            "partition": "debug" 
        }
    }
    """

    json_file = tmp_path.joinpath('cluster_config.json')
    json_file.write_text(config)

    return json_file


@pytest.fixture
def basic_train_py(tmp_path) -> Path:
    py = textwrap.dedent("""
    import time
    import json
    from pathlib import Path
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--job_dir', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--report', type=str)
    args = parser.parse_args()
    
    config = json.loads(args.config)
    
    report = {'report.task': config['task']}
    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    job_dir.joinpath(args.report).write_text(json.dumps(report))
    
    if config['task'] == 'sleep':
        time.sleep(config['sleep']['time'])
    elif config['task'] == 'count':
        n_list = [str(n) for n in range(config['count']['max_number'])]
        print(", ".join(n_list))
    else:
        raise ValueError(f"unexpected task: {config['task']}")
    
    """)

    py_file = tmp_path.joinpath('train_py.py')
    py_file.write_text(py)

    return py_file


@pytest.fixture(scope='package')
def create_train_dir_fcn(input_dim, train_config) -> Callable:
    def create_train_dir(session_dir: Path, n_jobs=5) -> List[str]:
        s_dict = json.dumps({'mean': [0] * input_dim, 'std': [1] * input_dim})
        (session_dir / 'train').mkdir(parents=True)
        (session_dir / 'train' / 'stand.json').write_text(s_dict)

        for i_job in range(n_jobs):
            config = dot_map_dict_to_nested_dict(train_config)
            config = sample_config(config)

            if config['arch'] == 'cnn':
                kwargs = CNNClassifier.kwargs_from_config(config)
                model = CNNClassifier(**kwargs)
            else:   # lstm
                kwargs = LSTMClassifier.kwargs_from_config(config)
                model = LSTMClassifier(**kwargs)

            (session_dir / 'train' / f'job{i_job}').mkdir()
            torch.save(model, session_dir / 'train' / f'job{i_job}' / 'model.pt')
            (session_dir / 'train' / f'job{i_job}' / 'variant.json').write_text(json.dumps(config))

        return [f'job{i}' for i in range(n_jobs)]

    return create_train_dir

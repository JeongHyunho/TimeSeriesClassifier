import json
from collections.abc import Callable
from pathlib import Path

import logging
import socket

import pytest
from typing import List

import struct
import torch

from core.util import dot_map_dict_to_nested_dict, sample_config
from models.armcurl_estimator import CNNEstimator


@pytest.fixture(scope='package')
def data_len(stream_data) -> int:
    return stream_data.shape[-1]


@pytest.fixture(scope='package')
def control_len() -> int:
    return 0


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
                if control_len > 0:
                    data_recv = cli_sock.recv(4 * data_len)
                    signal = struct.unpack(f'>{control_len}i', data_recv)
                    logger.debug(f'#{i} received: {signal}')
            else:
                raise ValueError(f"type should be 'log' or 'control', but got {type}")

        cli_sock.close()

    return _create_client


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
                kwargs = CNNEstimator.kwargs_from_config(config)
                model = CNNEstimator(**kwargs)
            else:   # lstm
                from models.armcurl_estimator import LSTMEstimator
                kwargs = LSTMEstimator.kwargs_from_config(config)
                model = LSTMEstimator(**kwargs)

            (session_dir / 'train' / f'job{i_job}').mkdir()
            torch.save(model, session_dir / 'train' / f'job{i_job}' / 'model.pt')
            (session_dir / 'train' / f'job{i_job}' / 'variant.json').write_text(json.dumps(config))

        return [f'job{i}' for i in range(n_jobs)]

    return create_train_dir

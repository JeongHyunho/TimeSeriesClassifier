import pytest
from typing import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from core.tcp_buffer import ProsthesisBuffer


@pytest.fixture(scope='package')
def in_tensor(time_length, input_dim, output_dim, use_gpu) -> (torch.Tensor, torch.Tensor):
    batch_size = 16

    inp_tensor = torch.randn(batch_size, time_length, input_dim)
    label = F.one_hot(torch.arange(time_length) % output_dim, num_classes=output_dim)
    label = label[None, ...].expand(batch_size, -1, -1)

    if use_gpu:
        inp_tensor = inp_tensor.cuda()
        label = label.cuda()

    return inp_tensor, label


@pytest.fixture(scope='package')
def make_session_fcn(buffer_config, stream_data) -> Callable:
    def make_session(session_name, output_dir, num_trials=5):
        for _ in range(num_trials):
            buffer = ProsthesisBuffer(
                config=buffer_config,
                session_name=session_name,
                output_dir=output_dir,
            )

            for data in stream_data:
                buffer.receive(data)

            buffer.save()

    return make_session


@pytest.fixture(scope='package')
def cnn_basic_kwargs(time_length, input_dim, output_dim, use_gpu) -> dict:
    kwargs = {
        "input_width": time_length,
        "input_channels": input_dim,
        "kernel_sizes": [3, 3],
        "n_channels": [input_dim, 2 * input_dim],
        "groups": input_dim,
        "strides": [1, 1],
        "paddings": ['same', 'same'],
        "fc_layers": [10, 10],
        "output_dim": output_dim,
        "cnn_norm": 'none',
        "pool_type": 'max',
        "pool_sizes": [None, 5],
        "pool_strides": [None, 5],
        "pool_paddings": [None, 0],
        "fc_norm": 'none',
        "device": 'cuda' if use_gpu else 'cpu',
    }

    return kwargs


@pytest.fixture(scope='package')
def lstm_basic_kwargs(time_length, input_dim, output_dim, use_gpu) -> dict:
    kwargs = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "feature_dim": 10,
        "hidden_dim": 20,
        "n_lstm_layers": 2,
        "pre_layers": [5, 5],
        "act_fcn": "relu",
        "lstm_norm": "none",
        "p_drop": 0.2,
        "fc_norm": "none",
        "device": "cuda" if use_gpu else "cpu",
    }

    return kwargs


@pytest.fixture(scope='package')
def train_py() -> Path:
    py_file = Path(__file__).parent.joinpath('../../../exp_scripts/pros_train_py.py')
    assert py_file.exists(), FileNotFoundError(f"{py_file} doesn't exist")

    return py_file
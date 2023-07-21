from pathlib import Path

import pytest
import torch


@pytest.fixture(scope='package')
def in_tensor(time_length, input_dim, output_dim, use_gpu) -> (torch.Tensor, torch.Tensor):
    batch_size = 16

    inp_tensor = torch.randn(batch_size, time_length, input_dim)
    label = torch.randn(batch_size, time_length, output_dim)

    if use_gpu:
        inp_tensor = inp_tensor.cuda()
        label = label.cuda()

    return inp_tensor, label


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
def lstm_basic_kwargs(input_dim, output_dim, use_gpu) -> dict:
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
def mlp_basic_kwargs(time_length, input_dim, output_dim, use_gpu) -> dict:
    kwargs = {
        "input_dim": input_dim,
        "input_width": time_length,
        "hidden_nodes": [64, 32],
        "output_dim": output_dim,
        "act_fcn": "relu",
        "norm": "none",
        "device": "cuda" if use_gpu else "cpu",
    }

    return kwargs


@pytest.fixture(scope='package')
def train_py() -> Path:
    py_file = Path(__file__).parent.joinpath('../../../exp_scripts/golf_train_py.py')
    assert py_file.exists(), FileNotFoundError(f"{py_file} doesn't exist")

    return py_file


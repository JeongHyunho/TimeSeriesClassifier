import pytest
import math
import torch

from core.util import dot_map_dict_to_nested_dict, sample_config
from exp_scripts.armcurl_train_py import load_model_cnn, load_model_lstm, load_model_mlp
from models.armcurl_estimator import CNNEstimator, LSTMEstimator, MLPEstimator


@pytest.mark.parametrize("cnn_norm", ['none', 'batch', 'layer'])
@pytest.mark.parametrize("fc_norm", ['none', 'batch', 'layer'])
def test_cnn_estimator_init(in_tensor, cnn_basic_kwargs, cnn_norm, fc_norm):
    cnn_basic_kwargs['cnn_norm'] = cnn_norm
    cnn_basic_kwargs['fc_norm'] = fc_norm

    model = CNNEstimator(**cnn_basic_kwargs)
    loss, _ = model.calc_loss(*in_tensor)
    acc = model.calc_acc(*in_tensor)

    assert torch.isfinite(loss)
    assert all([math.isfinite(v) for v in acc])


@pytest.mark.parametrize("lstm_norm", ['none', 'layer'])
@pytest.mark.parametrize("fc_norm", ['none', 'layer'])
def test_lstm_estimator_init(in_tensor, lstm_basic_kwargs, lstm_norm, fc_norm):
    lstm_basic_kwargs['lstm_norm'] = lstm_norm
    lstm_basic_kwargs['fc_norm'] = fc_norm

    model = LSTMEstimator(**lstm_basic_kwargs)
    loss, _ = model.calc_loss(*in_tensor)
    acc = model.calc_acc(*in_tensor)

    assert torch.isfinite(loss)
    assert all([math.isfinite(v) for v in acc])


@pytest.mark.parametrize("mlp_norm", ['none', 'batch', 'layer'])
def test_mlp_estimator_init(in_tensor, mlp_basic_kwargs, mlp_norm):
    mlp_basic_kwargs['norm'] = mlp_norm

    model = MLPEstimator(**mlp_basic_kwargs)
    loss, _ = model.calc_loss(*in_tensor)
    acc = model.calc_acc(*in_tensor)

    assert torch.isfinite(loss)
    assert all([math.isfinite(v) for v in acc])


# @pytest.mark.parametrize("arch", ['cnn', 'lstm', 'mlp'])
@pytest.mark.parametrize("arch", ['mlp'])
def test_estimator_train(make_session_fcn, train_config, arch, use_gpu, tmp_path):
    make_session_fcn(session_name='test', output_dir=tmp_path, num_trials=200)
    train_dir = tmp_path / 'test' / 'log'

    config = dot_map_dict_to_nested_dict(train_config)
    sampled = sample_config(config)

    max_iter = 100
    iter = 1
    while sampled['arch'] != arch and iter < max_iter:
        sampled = sample_config(config)
        iter += 1
    assert(iter < max_iter, f'cannot sample arch={arch}')

    if arch == 'cnn':
        load_fcn = load_model_cnn
    elif arch == 'lstm':
        load_fcn = load_model_lstm
    else:
        load_fcn = load_model_mlp

    model, train_dl, val_dl, test_dl = load_fcn(train_dir, sampled)

    epoch = 1
    loss, _ = model.train_model(epoch, train_dl)
    val_loss, _ = model.train_model(epoch, val_dl, evaluation=True)
    test_acc = model.calc_acc(test_dl.dataset.samples, test_dl.dataset.labels)

    assert math.isfinite(loss)
    assert math.isfinite(val_loss)
    assert all([math.isfinite(v) for v in test_acc])

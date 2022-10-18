import pytest
import math
import torch

from models.armcurl_estimator import CNNEstimator, LSTMEstimator


@pytest.mark.parametrize("cnn_norm", ['none', 'batch', 'layer'])
@pytest.mark.parametrize("fc_norm", ['none', 'batch', 'layer'])
def test_cnn_estimator_init(in_tensor, cnn_basic_kwargs, cnn_norm, fc_norm):
    cnn_basic_kwargs['cnn_norm'] = cnn_norm
    cnn_basic_kwargs['fc_norm'] = fc_norm

    model = CNNEstimator(**cnn_basic_kwargs)
    loss = model.calc_loss(*in_tensor)
    acc = model.calc_acc(*in_tensor)

    assert torch.isfinite(loss)
    assert all([math.isfinite(v) for v in acc])


@pytest.mark.parametrize("lstm_norm", ['none', 'layer'])
@pytest.mark.parametrize("fc_norm", ['none', 'layer'])
def test_lstm_estimator_init(in_tensor, lstm_basic_kwargs, lstm_norm, fc_norm):
    lstm_basic_kwargs['lstm_norm'] = lstm_norm
    lstm_basic_kwargs['fc_norm'] = fc_norm

    model = LSTMEstimator(**lstm_basic_kwargs)
    loss = model.calc_loss(*in_tensor)
    acc = model.calc_acc(*in_tensor)

    assert torch.isfinite(loss)
    assert all([math.isfinite(v) for v in acc])

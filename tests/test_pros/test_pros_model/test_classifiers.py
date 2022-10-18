import math

import pytest
import torch

from models.gait_phase_classifier import CNNClassifier, LSTMClassifier


@pytest.mark.parametrize("cnn_norm", ['none', 'batch', 'layer'])
@pytest.mark.parametrize("fc_norm", ['none', 'batch', 'layer'])
def test_cnn_classifier_basic(in_tensor, cnn_basic_kwargs, cnn_norm, fc_norm):
    cnn_basic_kwargs['cnn_norm'] = cnn_norm
    cnn_basic_kwargs['fc_norm'] = fc_norm

    model = CNNClassifier(**cnn_basic_kwargs)
    loss = model.calc_loss(*in_tensor)
    acc = model.calc_acc(*in_tensor)

    assert torch.isfinite(loss)
    assert math.isfinite(acc)


@pytest.mark.parametrize("lstm_norm", ['none', 'layer'])
@pytest.mark.parametrize("fc_norm", ['none', 'layer'])
def test_lstm_classifier_basic(in_tensor, lstm_basic_kwargs, lstm_norm, fc_norm):
    lstm_basic_kwargs['lstm_norm'] = lstm_norm
    lstm_basic_kwargs['fc_norm'] = fc_norm

    model = LSTMClassifier(**lstm_basic_kwargs)
    loss = model.calc_loss(*in_tensor)
    acc = model.calc_acc(*in_tensor)

    assert torch.isfinite(loss)
    assert math.isfinite(acc)

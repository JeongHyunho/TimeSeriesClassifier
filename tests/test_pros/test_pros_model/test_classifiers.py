import math

import pytest
import torch

from core.util import sample_config, dot_map_dict_to_nested_dict
from models.gait_phase_classifier import CNNClassifier, LSTMClassifier


@pytest.mark.parametrize("cnn_norm", ['none', 'batch', 'layer'])
@pytest.mark.parametrize("fc_norm", ['none', 'batch', 'layer'])
def test_cnn_classifier_basic(in_tensor, cnn_basic_kwargs, cnn_norm, fc_norm):
    cnn_basic_kwargs['cnn_norm'] = cnn_norm
    cnn_basic_kwargs['fc_norm'] = fc_norm

    model = CNNClassifier(**cnn_basic_kwargs)
    loss = model.calc_loss(*in_tensor)
    acc = model.calc_acc(*in_tensor)
    model.confusion_matrix_figure(*in_tensor)

    assert torch.isfinite(loss)
    assert math.isfinite(acc)


@pytest.mark.parametrize("lstm_norm", ['none', 'layer'])
@pytest.mark.parametrize("fc_norm", ['none', 'layer'])
def test_lstm_classifier_basic(in_tensor, lstm_basic_kwargs, lstm_norm, fc_norm):
    lstm_basic_kwargs['lstm_norm'] = lstm_norm
    lstm_basic_kwargs['fc_norm'] = fc_norm

    model = LSTMClassifier(**lstm_basic_kwargs)
    loss = model.calc_loss(*in_tensor)
    loss.backward()
    acc = model.calc_acc(*in_tensor)
    model.confusion_matrix_figure(*in_tensor)

    assert torch.isfinite(loss)
    assert math.isfinite(acc)


def test_model_from_sampled_config(in_tensor, train_config, time_length):
    config = sample_config(train_config)
    config['cnn.input_width'] = time_length
    config['lstm.window_size'] = time_length

    config = dot_map_dict_to_nested_dict(config)
    in_tensor = (in_tensor[0].to(config['device']), in_tensor[1].to(config['device']))

    if config['arch'] == 'cnn':
        model_cls = CNNClassifier
    else:
        model_cls = LSTMClassifier
    kwargs = model_cls.kwargs_from_config(config)

    model = model_cls(**kwargs)
    loss = model.calc_loss(*in_tensor)
    acc = model.calc_acc(*in_tensor)
    model.confusion_matrix_figure(*in_tensor)

    assert torch.isfinite(loss)
    assert math.isfinite(acc)

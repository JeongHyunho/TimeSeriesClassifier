from torch import nn


def identity(x):
    return x

_str_to_activation = {
    'identity': identity,
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
}


def activation_from_string(string):
    return _str_to_activation[string]

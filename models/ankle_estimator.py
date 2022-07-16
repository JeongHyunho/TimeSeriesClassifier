import math

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models.networks import AttentionBlock, TCBlock, LayerNormLSTM


class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()

    def calc_loss(self, input, label):
        """
        :param input: shape of B x T x N
        :param label: shape of B x T
        :return: loss
        """
        pred = self.forward(input)
        loss = self.criterion(pred, label)

        return loss

    def train_model(self, epoch, loader: DataLoader, evaluation=False, verbose=False):
        train_loss = 0.
        if evaluation:
            self.eval()
        else:
            self.train()

        for data, label in loader:

            if evaluation:
                with torch.no_grad():
                    loss = self.calc_loss(data, label)
            else:
                loss = self.calc_loss(data, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss += loss.item()

        train_loss /= len(loader)
        if verbose:
            print(f'({"Eval" if evaluation else "Train"}) Loss: {train_loss:.4f}')
        return train_loss

    @classmethod
    def load_from_config(cls, config, state_dict_file, map_location='cpu'):
        raise NotImplementedError

    @staticmethod
    def _construct_module(layers, input_dim, output_dim, layer_norm=False, act_fcn=nn.ReLU):
        module = nn.Sequential()
        if layers is None or layers == []:
            module.add_module('layer0', nn.Linear(input_dim, output_dim))
            if layer_norm:
                module.add_module('ln0', nn.LayerNorm(output_dim))
        else:
            mlp_in = input_dim
            for idx, mlp_out in enumerate(layers):
                module.add_module(f'layer{idx}', nn.Linear(mlp_in, mlp_out))
                if layer_norm:
                    module.add_module(f'ln{idx}', nn.LayerNorm(mlp_out))
                module.add_module(f'act{idx}', act_fcn())
                mlp_in = mlp_out
            module.add_module('layer_out', nn.Linear(mlp_in, output_dim))
            if layer_norm:
                module.add_module('last_ln', nn.LayerNorm(output_dim))

        return module


class LSTMEstimator(Estimator):
    def __init__(self,
                 input_dim,
                 output_dim,
                 feature_dim,
                 hidden_dim=256,
                 n_lstm_layers=1,
                 pre_layers=None,
                 post_layers=None,
                 act_fcn=nn.ReLU,
                 lr=1e-3,
                 p_drop=0.2,
                 layer_norm=False,
                 device='cuda'):
        super(LSTMEstimator, self).__init__()
        self.pre_module = self._construct_module(pre_layers, input_dim, feature_dim, layer_norm, act_fcn)

        lstm_cls = LayerNormLSTM if layer_norm else nn.LSTM
        lstm_in = feature_dim
        self.lstm = lstm_cls(
            input_size=lstm_in,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            dropout=p_drop,
            batch_first=True
            )

        mlp_in = hidden_dim
        self.post_module = self._construct_module(post_layers, mlp_in, output_dim, layer_norm, act_fcn)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, input):
        """
        :param input: shape of B x T x N
        :return: shape of B x T
        """
        B, T, _ = input.shape
        pre_processed = self.pre_module(input)      # B x T x N

        h, _ = self.lstm(pre_processed)             # B x T x H
        post_processed = self.post_module(h)        # B x T x 1

        return torch.squeeze(post_processed, dim=-1)

    @classmethod
    def load_from_config(cls, config, state_dict_file, map_location='cpu'):
        model = cls(input_dim=config['input_dim'],
                    output_dim=config['output_dim'],
                    feature_dim=config['feature_dim'],
                    hidden_dim=config['hidden_dim'],
                    n_lstm_layers=config['n_lstm_layers'],
                    pre_layers=[config['n_pre_nodes']] * config['n_pre_layers'],
                    post_layers=[config['n_post_nodes']] * config['n_post_layers'],
                    p_drop=config['p_drop'],
                    layer_norm=config['layer_norm'],
                    lr=config['lr'],
                    device=map_location)
        state_dict = torch.load(state_dict_file)
        model.load_state_dict(state_dict)

        return model


class ExtendedLSTMEstimator(LSTMEstimator):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_locals,
                 hidden_dim=256,
                 post_layers=None,
                 act_fcn=nn.ReLU,
                 layer_norm=False,
                 device='cuda',
                 *args,
                 **kwargs):
        super(ExtendedLSTMEstimator, self).__init__(input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    hidden_dim=hidden_dim,
                                                    post_layers=post_layers,
                                                    act_fcn=act_fcn,
                                                    layer_norm=layer_norm,
                                                    device=device,
                                                    *args,
                                                    **kwargs)
        self.n_locals = n_locals

        mlp_in = hidden_dim + input_dim * self.n_locals
        self.post_module = self._construct_module(post_layers, mlp_in, output_dim, layer_norm, act_fcn)

        self.to(device)

    def forward(self, input):
        """
        :param input: shape of B x T x N
        :return: shape of B x T
        """
        B, T, N = input.shape
        pre_processed = self.pre_module(input)              # B x T x N_pre

        extended_input = torch.cat([torch.zeros(B, self.n_locals - 1, N).to(input.device), input], dim=1)
        local_input = torch.cat(
            [extended_input[:, idx:idx + self.n_locals, :].view(B, 1, -1) for idx in range(T)],
            dim=1)                                          # B x T x (N x nL)

        h, _ = self.lstm(pre_processed)
        post_in = torch.cat([h, local_input], dim=-1)       # B x T x (H + N * nL)
        post_processed = self.post_module(post_in)          # B x T x 1

        return torch.squeeze(post_processed, dim=-1)

    @classmethod
    def load_from_config(cls, config, state_dict_file, map_location='cpu'):
        model = cls(input_dim=config['input_dim'],
                    output_dim=config['output_dim'],
                    n_locals=config['n_locals'],
                    feature_dim=config['feature_dim'],
                    n_lstm_layers=config['n_lstm_layers'],
                    pre_layers=[config['n_pre_nodes']] * config['n_pre_layers'],
                    post_layers=[config['n_post_nodes']] * config['n_post_layers'],
                    p_drop=config['p_drop'],
                    layer_norm=config['layer_norm'],
                    lr=config['lr'],
                    device=map_location)
        state_dict = torch.load(state_dict_file)
        model.load_state_dict(state_dict)

        return model


class SnailEstimator(Estimator):
    def __init__(self,
                 input_dim,
                 output_dim,
                 key_dims,
                 value_dims,
                 filter_dims,
                 target_length,
                 layer_norm=False,
                 lr=1e-3,
                 device='cuda'):
        assert len(key_dims) == len(value_dims) and len(key_dims) - 1 == len(filter_dims)
        super(SnailEstimator, self).__init__()
        self.n_stacks = len(key_dims)

        n_layers = int(math.ceil(math.log(target_length, 2)))
        in_channel_dim = input_dim
        for idx, (k_dim, v_dim) in enumerate(zip(key_dims, value_dims)):
            _SA = AttentionBlock(in_channel_dim, k_dim, v_dim, ln=layer_norm)
            self.__setattr__(f'SA{idx}', _SA)

            if idx < self.n_stacks - 1:
                _TC = TCBlock(in_channel_dim + v_dim, n_layers, filter_dims[idx])
                self.__setattr__(f'TC{idx}', _TC)
                in_channel_dim += v_dim + n_layers * filter_dims[idx]
            else:
                in_channel_dim += v_dim

        self.output_linear = nn.Linear(in_channel_dim, output_dim)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, input):
        """
        :param input: shape of B x T x Di
        :return output: shape of B x T x Do
        """
        h = input
        for idx in range(self.n_stacks):
            h = self.__getattr__(f'SA{idx}')(h)

            if idx < self.n_stacks - 1:
                h = self.__getattr__(f'TC{idx}')(h)
        output = self.output_linear(h).squeeze(-1)

        return output

    @classmethod
    def load_from_config(cls, config, state_dict_file, map_location='cpu'):
        model = cls(input_dim=config['input_dim'],
                    output_dim=config['output_dim'],
                    key_dims=config['key_value_dims'][0],
                    value_dims=config['key_value_dims'][1],
                    filter_dims=[config['filter_dim']] * (len(config['key_value_dims'][0]) - 1),
                    target_length=config['target_length'],
                    layer_norm=config['layer_norm'],
                    lr=config['lr'],
                    device=map_location)
        state_dict = torch.load(state_dict_file)
        model.load_state_dict(state_dict)

        return model

import abc
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader

from models import identity
from models.networks import LayerNormLSTM, construct_mlp, Basic1DCNN


class Classifier(nn.Module, abc.ABC):

    def __init__(self, reduce_time):
        super().__init__()
        self.reduce_time = reduce_time

    def calc_loss(self, input, label):
        """ Calculate loss of model on a data

        Args:
            input (torch.Tensor): input date tensor of (B, T, D)
            label (torch.Tensor): one-hot vector tensor of (B, T, C)

        Returns:
             torch.Tensor: loss of scalar tensor

        """

        if self.reduce_time:
            label = label[:, -1, :]

        pred = self.forward(input)
        y_pred = torch.sum(pred * label,  dim=-1)

        loss = - (y_pred - torch.logsumexp(pred, dim=-1)).mean()

        return loss

    @torch.no_grad()
    def calc_acc(self, input, label):
        """ Calculate accuracy on a data

        Args:
            input (torch.Tensor): input data tensor of (B, T, D)
            label (torch.Tensor):  one-hot vector tensor of (B, T, C)

        Returns:
            float: accuracy scalar

        """

        self.eval()
        if self.reduce_time:
            label = label[:, -1, :]

        pred = self.forward(input)
        p_label = torch.argmax(pred, dim=-1)
        label_idx = torch.argmax(label, dim=-1)
        acc = torch.sum(p_label == label_idx) / p_label.nelement()

        return acc.item()

    def train_model(self, epoch, loader, evaluation=False, verbose=False):
        """ Iterate data loader to train/evaluate model

        Args:
            epoch (int): current number of epoch
            loader (DataLoader): loader for train/evaluaion
            evaluation (bool): flag for evaluation mode
            verbose (bool): print loss or not

        """

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
            print(f'[Epoch {epoch}] ({"Eval" if evaluation else "Train"}) Loss: {train_loss:.4f}')

        return train_loss

    @torch.no_grad()
    def confusion_matrix_figure(self, inp, label) -> plt.Figure:
        """ draw confusion matrix

        Args:
            inp (torch.Tensor): input data tensor of (B, T, D)
            label (torch.Tensor): one-hot vector tensor of (B, T, C)

        Returns: matplotlib figure handle

        """

        self.eval()
        if self.reduce_time:
            label = label[:, -1, :]

        idx_label = torch.argmax(label, dim=-1)
        pred = self.forward(inp)
        p_label = torch.argmax(pred, dim=-1)

        conf_mat = confusion_matrix(idx_label.cpu().flatten(), p_label.cpu().flatten(), labels=np.arange(label.size(-1)))
        conf_mat = conf_mat / np.sum(conf_mat, -1, keepdims=True)

        fig = plt.figure(figsize=(5,5))
        plt.set_cmap('Greys_r')
        ax = fig.gca()
        cax = ax.matshow(conf_mat)
        cax.set_clim(vmin=0., vmax=1.)
        fig.colorbar(cax)

        for idx_r, row in enumerate(conf_mat):
            for idx_c, el in enumerate(row):
                text_c = np.ones(3) if el < 0.5 else np.zeros(3)
                ax.text(idx_c, idx_r, f'{100 * el:.1f}',
                        va='center', ha='center', c=text_c, size='x-large')

        return fig

    @abc.abstractmethod
    def forward(self, inp, *args, **kwargs):
        """ predict via model

        Args:
            inp (torch.Tensor): input date tensor of (B, T, D)

        Returns:
            torch.Tensor: result tensor of (B, C) if reduce_time is true, else (B, T, C)

        """
        raise NotImplementedError

    @staticmethod
    def kwargs_from_config(config):
        """ return kwargs for init method

        Args:
            config (dict): more easy-to-read configuration

        Returns:
            dict: keyword arguments for init method

        """
        raise NotImplementedError


class CNNClassifier(Classifier):
    def __init__(
            self,

            input_width,
            input_channels,
            kernel_sizes,
            n_channels,
            groups,
            strides,
            paddings,

            fc_layers,
            output_dim,

            cnn_norm='none',
            hidden_init=None,
            hidden_activation='relu',
            output_activation=identity,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,

            lr=1e-3,
            device='cuda',

            fc_norm='none',
            fc_act_fcn='relu',
    ):
        super().__init__(reduce_time=True)
        self.cnn = Basic1DCNN(
            input_width=input_width,
            input_channels=input_channels,
            kernel_sizes=kernel_sizes,
            n_channels=n_channels,
            groups=groups,
            strides=strides,
            paddings=paddings,
            normalization_type=cnn_norm,
            hidden_init=hidden_init,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            pool_type=pool_type,
            pool_sizes=pool_sizes,
            pool_strides=pool_strides,
            pool_paddings=pool_paddings,
        )
        self.fc = construct_mlp(
            fc_layers,
            math.prod(self.cnn.output_shape),
            output_dim,
            fc_norm,
            fc_act_fcn,
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, inp, **kwargs):
        """ predict via CNN

        Args:
            inp (torch.Tensor): input date tensor of (B, T, D)

        Returns:
            torch.Tensor: result tensor of (B, C)

        """

        inp_for_cnn = torch.transpose(inp, 1, 2)
        h = self.cnn(inp_for_cnn)
        oup = self.fc(torch.flatten(h, start_dim=1))

        return oup

    @staticmethod
    def kwargs_from_config(config):
        c_cnn = config['cnn']

        if config['signal_type'] == 'all':
            input_channels = 8
        elif config['signal_type'] == 'emg':
            input_channels = 4
        elif config['signal_type'] == 'eim':
            input_channels = 4
        else:
            raise ValueError(f"{config['signal_type']} not in ['all', 'emg', 'eim']")

        # assume two rounds cnn architecture
        kernel_sizes = [c_cnn['kernel_size0']] * c_cnn['n_conv_layer0'] \
                       + [c_cnn['kernel_size1']] * c_cnn['n_conv_layer1']
        n_channels = [input_channels * c_cnn['k_channel0']] * c_cnn['n_conv_layer0'] \
                     + [input_channels * c_cnn['k_channel1']] * c_cnn['n_conv_layer1']
        pool_sizes = [0] * (c_cnn['n_conv_layer0'] - 1) + [c_cnn['pool_size']] \
                     + [0] * (c_cnn['n_conv_layer1'] - 1) + [c_cnn['pool_size']]
        pool_strides = [None] * (c_cnn['n_conv_layer0'] - 1) + [c_cnn['pool_stride']] \
                       + [None] * (c_cnn['n_conv_layer1'] - 1) + [c_cnn['pool_stride']]
        pool_paddings = [None] * (c_cnn['n_conv_layer0'] - 1) + [0] \
                       + [None] * (c_cnn['n_conv_layer1'] - 1) + [0]

        kwargs = {
            'input_width': c_cnn['input_width'],
            'input_channels': input_channels,
            'kernel_sizes': kernel_sizes,
            'n_channels': n_channels,
            'groups': input_channels,
            'strides': [1] * (c_cnn['n_conv_layer0'] + c_cnn['n_conv_layer1']),
            'paddings': ['same'] * (c_cnn['n_conv_layer0'] + c_cnn['n_conv_layer1']),
            'fc_layers': [c_cnn['n_fc_units']] * c_cnn['n_fc_layers'],
            'output_dim': config['output_dim'],
            'cnn_norm': c_cnn['cnn_norm'],
            'pool_type': 'max',
            'pool_sizes': pool_sizes,
            'pool_strides': pool_strides,
            'pool_paddings': pool_paddings,
            'fc_norm': c_cnn['fc_norm'],
            'lr': c_cnn['lr'],
            'device': config['device'],
        }

        return kwargs


class LSTMClassifier(Classifier):
    def __init__(
            self,

            input_dim,
            output_dim,
            feature_dim,
            hidden_dim=256,
            n_lstm_layers=1,
            pre_layers=None,
            post_layers=None,
            act_fcn='relu',
            lstm_norm='none',
            lr=1e-3,
            p_drop=0.2,
            fc_norm='none',
            device='cuda',
    ):
        super().__init__(reduce_time=False)
        assert lstm_norm in ['none', 'layer']
        assert fc_norm in ['none', 'layer']

        self.pre_module = construct_mlp(pre_layers, input_dim, feature_dim, fc_norm, act_fcn)
        self.lstm_norm = lstm_norm

        if lstm_norm == 'none':
            lstm_cls = nn.LSTM
        elif lstm_norm == 'layer':
            lstm_cls = LayerNormLSTM
        else:
            raise ValueError(f'unexpected normalization_type: {lstm_norm}')

        self.lstm = lstm_cls(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            dropout=p_drop,
            batch_first=True
        )
        self.post_module = construct_mlp(post_layers, hidden_dim, output_dim, fc_norm, act_fcn)

        self.hc_n = None

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, inp, **kwargs):
        """ predict via LSTM

        Args:
            inp (torch.Tensor): input data tensor of (B, T, D)

        Returns:
            torch.Tensor: result tensor of (B, T, C)

        """

        hc0 = kwargs.get('hc0')     # initial latent  variable

        pre_processed = self.pre_module(inp)
        h, self.hc_n = self.lstm(pre_processed, hc0)
        oup = self.post_module(h)

        return oup

    @staticmethod
    def kwargs_from_config(config: dict) -> dict:
        c_lstm = config['lstm']
        pre_layers = [c_lstm['n_pre_nodes']] * c_lstm['n_pre_layers']
        post_layers = [c_lstm['n_post_nodes']] * c_lstm['n_post_layers']

        if config['signal_type'] == 'all':
            input_dim = 8
        elif config['signal_type'] == 'emg':
            input_dim = 4
        elif config['signal_type'] == 'eim':
            input_dim = 4
        else:
            raise ValueError(f"{config['signal_type']} not in ['all', 'emg', 'eim']")

        kwargs = {
            'input_dim': input_dim,
            'output_dim': config['output_dim'],
            'feature_dim': c_lstm['feature_dim'],
            'hidden_dim': c_lstm['hidden_dim'],
            'n_lstm_layers': c_lstm['n_lstm_layers'],
            'lstm_norm': c_lstm['lstm_norm'],
            'pre_layers': pre_layers,
            'post_layers': post_layers,
            'p_drop': c_lstm['p_drop'],
            'fc_norm': c_lstm['fc_norm'],
            'lr': c_lstm['lr'],
            'device': config['device'],
        }

        return kwargs


class MLPClassifier(Classifier):

    def __init__(
            self,

            input_dim,
            output_dim,
            input_width,
            hidden_nodes,
            act_fcn='relu',
            lr=1e-3,
            norm='none',
            device='cuda',
    ):
        super().__init__(reduce_time=True)
        assert norm in ['none', 'batch', 'layer']

        self.mlp = construct_mlp(
            hidden_nodes,
            input_width * input_dim,
            output_dim,
            norm_type=norm,
            act_fcn=act_fcn,
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, inp, **kwargs):
        """ predict via MLP

        Args:
            inp (torch.Tensor): input data tensor of (B, T, D)

        Returns:
            torch.Tensor: result tensor of (B, C)

        """

        inp_for_mlp = torch.flatten(inp, start_dim=1)
        oup = self.mlp(inp_for_mlp)

        return oup

    @staticmethod
    def kwargs_from_config(config):
        c_mlp = config['mlp']

        if config['signal_type'] == 'all':
            input_dim = 8
        elif config['signal_type'] == 'emg':
            input_dim = 4
        elif config['signal_type'] == 'eim':
            input_dim = 4
        else:
            raise ValueError(f"{config['signal_type']} not in ['all', 'emg', 'eim']")

        kwargs = {
            'input_dim': input_dim,
            'output_dim': config['output_dim'],
            'input_width': c_mlp['input_width'],
            'hidden_nodes': c_mlp['hidden_nodes'],
            'act_fcn': 'relu',
            'norm': c_mlp['norm'],
            'lr': c_mlp['lr'],
            'device': config['device'],
        }

        return kwargs

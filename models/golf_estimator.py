from __future__ import annotations

import abc
import math
import numpy as np
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from typing import List

from core.util import batch_by_window
from data.base_dataset import BaseDataset
from models import identity
from models.networks import LayerNormLSTM, construct_mlp, Basic1DCNN


class Estimator(nn.Module, abc.ABC):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            reduce_time: bool,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reduce_time = reduce_time

    def calc_loss(self, input, label) -> torch.Tensor:
        """ Calculate loss of model on a data

        Args:
            input (torch.Tensor): input tensor of (B, T, D)
            label (torch.Tensor): output tensor of (B, T, S)

        Returns:
             torch.Tensor: loss of scalar tensor

        """

        if self.reduce_time:
            label = label[:, -1, :]

        pred = self.__call__(input)
        loss = torch.mean((pred - label) ** 2)

        return loss

    @torch.no_grad()
    def calc_acc(self, input, label) -> float:
        """ Calulate accuracy on a data

        Args:
            input (torch.Tensor): input tensor of (B, T, D)
            label (torch.Tensor):  output tensor of (B, T, S)

        Returns:
            float: accuracy

        """

        self.eval()
        if self.reduce_time:
            label = label[:, -1, :]

        pred = self.forward(input)
        errors = (pred - label) ** 2
        acc = torch.mean(errors).item()

        return acc

    def train_model(self, epoch, loader, evaluation=False, verbose=False) -> float:
        """ Iterate data loader to train/evaluate model

        Args:
            epoch (int): current number of epoch
            loader (DataLoader): loader for train/evaluation
            evaluation (bool): flag for evaluation mode
            verbose (bool): print loss or not

        Returns:
            float: train loss of this epoch

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

    @abc.abstractmethod
    def forward(self, inp, *args, **kwargs):
        """ predict via model

        Args:
            inp (torch.Tensor): input date tensor of (B, T, D)

        Returns:
            torch.Tensor: prediction tensor of (B, S) if 'reduce_time' is true, else (B, T, S)

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

    @torch.no_grad()
    def post_process(self, dataset: BaseDataset, post_dir, w_size=None, y_labels=None, device='cpu') -> List[Path]:
        """ evaluate model and save results

        Args:
            dataset (Dataset): dataset of processed data
            post_dir (Path): directory for saving results
            w_size (int): required if model input is window of tensor
            y_labels (str): y-axis labels in plot
            device (str): device for input/output tensors of model

        Returns:
            List[Path]: result file path

        """

        files = []
        samples, labels = dataset.get_test_stream(device=device)

        self.eval()
        for idx, (x, y) in enumerate(zip(samples, labels)):
            if self.reduce_time:
                assert w_size is not None, f"'w_size' is required for this model {type(self)}"
                inp = batch_by_window(x, w_size)
                pred = self(inp)
                nan_tensor = torch.nan * torch.ones(w_size -1, pred.size(-1))
                pred = torch.cat([nan_tensor.to(device), pred], dim=0)
            else:
                pred = self(x[None, ...])[0, ...]

            p_np, y_np = pred.cpu().numpy(), y.cpu().numpy()
            mse = np.nanmean((p_np - y_np) ** 2, axis=0)
            rmse = np.sqrt(mse)
            pm_np = np.nanmean(p_np, axis=0, keepdims=True)
            ym_np = np.nanmean(y_np, axis=0, keepdims=True)
            r2 = 1 - np.nansum((p_np - y_np) ** 2, axis=0) / np.nansum((y_np - ym_np) ** 2, axis=0)
            vaf = 100*np.nansum((y_np-ym_np)*(p_np-pm_np), axis=0) / np.nansum((y_np-ym_np) ** 2, axis=0)

            fh = plt.figure(figsize=(4, 2 + 2 * self.output_dim))
            plt.subplot(self.output_dim, 1, 1)

            plt.subplot()
            plt.plot(np.hstack([p_np, y_np]), label=['pred', 'data'])
            plt.ylabel(y_labels)

            summary = f'\nRMSE: {rmse[0]:.2f}, R2: {r2[0]:.2f}, VAF: {vaf[0]:.1f}%'
            plt.title(f'#{idx}' + summary)
            plt.legend()
            plt.xlabel('index')
            plt.tight_layout()

            img_filename = post_dir.joinpath(f'postprocess{idx}.png')
            files.append(img_filename)
            fh.savefig(img_filename)
            plt.close()

            pd_data = pd.DataFrame(np.hstack([p_np, y_np]), columns=['prediction', 'label'])
            csv_filename = post_dir.joinpath(f'test_resutls{idx}.csv')
            pd_data.to_csv(csv_filename)
        self.train()

        return files


class CNNEstimator(Estimator):
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
        super().__init__(
            input_dim=input_channels,
            output_dim=output_dim,
            reduce_time=True,
        )
        self.input_width = input_width

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
            torch.Tensor: prediction tensor of (B, S)

        """

        inp_for_cnn = torch.transpose(inp, 1, 2)
        h = self.cnn(inp_for_cnn)
        oup = self.fc(torch.flatten(h, start_dim=1))

        return oup

    @staticmethod
    def kwargs_from_config(config):
        c_cnn = config['cnn']

        input_channels = 4

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


class LSTMEstimator(Estimator):
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
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            reduce_time=False,
        )
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
            torch.Tensor: prediction tensor of (B, T, S)

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

        input_dim = 4
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


class MLPEstimator(Estimator):
    """ Multi-layered perceptron estimator """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            input_width: int,
            hidden_nodes: List[int],
            act_fcn: str = 'relu',
            lr: float = 1e-3,
            norm: str = 'none',
            device: str = 'cuda',
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            reduce_time=True,
        )
        assert norm in ['none', 'batch', 'layer']
        self.input_width = input_width

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
        """ predict vial MLP

        Args:
            inp: (torch.Tensor): input data tensor of (B, T, D)

        Returns:
            torch.Tensor: result tensor of (B, O)

        """

        inp_for_mlp = torch.flatten(inp, start_dim=1)
        oup = self.mlp(inp_for_mlp)

        return oup

    @staticmethod
    def kwargs_from_config(config):
        c_mlp = config['mlp']

        input_dim = 4
        kwargs = {
            'input_dim': input_dim,
            'output_dim': config['output_dim'],
            'input_width': c_mlp['input_width'],
            'hidden_nodes': c_mlp['hidden_nodes'],
            'act_fcn': c_mlp['act_fcn'],
            'norm': c_mlp['norm'],
            'lr': c_mlp['lr'],
            'device': config['device'],
        }

        return kwargs

import abc
import math

import torch
from torch import nn, optim

from models import identity
from models.networks import LayerNormLSTM, construct_mlp, Basic1DCNN


class Detector(nn.Module, abc.ABC):
    def __init__(self, criterion):
        super().__init__()
        assert criterion in ['bce', 'cce'], ValueError
        self.criterion = criterion

    def calc_loss(self, input, label):
        """ Binary cross entropy 로스를 계산하여 반환

        Args:
            input: 데이터 텐서 (B, T, D)
            label: 라벨 텐서 (B, 1)

        Returns: 스칼라 텐서

        """

        pred = self.forward(input)  # (B, T, C)
        label = label.unsqueeze(dim=1).expand(-1, pred.size(1), -1)

        if self.criterion == 'bce':
            target = torch.zeros_like(pred).scatter_(-1, label, 1)
            target.scatter_(-1, label, 1)
            loss = - torch.sum(target * torch.log(pred) + (1 - target) * torch.log(1 - pred), -1).mean()
        else:
            y_pred = torch.gather(pred, -1, label)
            loss = - (y_pred - torch.logsumexp(pred, dim=-1, keepdim=True)).mean()

        return loss

    @torch.no_grad()
    def calc_acc(self, input, label, method='vote'):
        """ 정확도 계산

        Args:
            input: 입력 텐서 (B, T, D)
            label: 라벨 텐서 (B, 1)
            method: 'last_time' or 'vote',
                last_time 이면 마지막 time_step 의 출력을, vote 이면 최빈값을 기준으로 정확도를 계산

        Returns: 정확도 float

        """

        self.eval()
        pred = self.forward(input)

        if method == 'last_time':
            p_label = torch.argmax(pred[:, -1, :], dim=-1)
        elif method == 'vote':
            p_label, _ = torch.mode(torch.argmax(pred, -1), -1)
        else:
            raise ValueError

        acc = torch.sum(p_label == label.flatten()) / pred.size(0)

        return acc.item()

    def train_model(self, epoch, loader, evaluation=False, verbose=False):
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

    @abc.abstractmethod
    def forward(self, inp, *args, **kwargs):
        """ Gait condition detection 수행

        Args:
            inp: 입력 텐서 (B, T, input_dim)

        Returns:
            oup: detect 결과 텐서 (B, T, output_dim)

        """
        raise NotImplementedError

    @classmethod
    def load_from_config(cls, config, state_dict_file, map_location='cpu'):
        raise NotImplementedError


class LSTMDetector(Detector):
    def __init__(
            self,
            input_dim,
            output_dim,
            feature_dim,
            hidden_dim=256,
            n_lstm_layers=1,
            bidirectional=False,
            pre_layers=None,
            post_layers=None,
            act_fcn='relu',
            criterion='bce',
            lr=1e-3,
            p_drop=0.2,
            fc_norm='none',
            device='cuda',
    ):
        super().__init__(criterion)

        self.pre_module = construct_mlp(pre_layers, input_dim, feature_dim, fc_norm, act_fcn)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            dropout=p_drop,
            batch_first=True,
            bidirectional=bidirectional,
        )
        post_n_inp = 2 * hidden_dim if bidirectional else hidden_dim
        self.post_module = construct_mlp(post_layers, post_n_inp, output_dim, fc_norm, act_fcn)
        self.out_activation = nn.Sigmoid() if criterion == 'bce' else nn.Identity()

        self.hc_n = None

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, inp, hc_0=None):
        """ LSTM 기반 모델의 detect 수행

        Args:
            inp: 입력 텐서 (B, T, input_dim)
            hc_0: 초기 잠재 변수, 셀 상태 텐서 (n_layers, B, hidden_dim) 의 튜플
                None 이면 0 으로 초기화

        Returns:
            oup: detect 결과 텐서 (B, T, output_dim)

        """

        pre_processed = self.pre_module(inp)
        h, self.hc_n = self.lstm(pre_processed, hc_0)
        post_processed = self.post_module(h)
        oup = self.out_activation(post_processed)

        return oup

    @classmethod
    def load_from_config(cls, config, state_dict_file, map_location='cpu'):
        model = cls(input_dim=config['input_dim'],
                    output_dim=config['output_dim'],
                    feature_dim=config['feature_dim'],
                    hidden_dim=config['hidden_dim'],
                    n_lstm_layers=config['n_lstm_layers'],
                    pre_layers=[config['n_pre_nodes']] * config['n_pre_layers'],
                    post_layers=[config['n_post_nodes']] * config['n_post_layers'],
                    act_fcn=config['act_fcn'],
                    p_drop=config['p_drop'],
                    fc_norm=config['fc_norm'],
                    criterion=config['criterion'],
                    bidirectional=config['bidirectional'],
                    lr=config['lr'],
                    device=map_location)
        state_dict = torch.load(state_dict_file)
        model.load_state_dict(state_dict)

        return model


class CNNDetector(Detector):
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

            normalization_type='none',
            hidden_init=None,
            hidden_activation='relu',
            output_activation=identity,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,

            criterion='cce',
            lr=1e-3,
            device='cuda',

            fc_norm='none',
            fc_act_fcn='relu',
    ):
        super().__init__(criterion)
        self.cnn = Basic1DCNN(
            input_width=input_width,
            input_channels=input_channels,
            kernel_sizes=kernel_sizes,
            n_channels=n_channels,
            groups=groups,
            strides=strides,
            paddings=paddings,
            normalization_type=normalization_type,
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
        self.fc_out_act = nn.Sigmoid() if criterion == 'bce' else nn.Identity()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, inp, **kwargs):
        """ CNN 기반 모델의 detect 수행

        Args:
            inp: 입력 텐서 (B, T, input_dim)

        Returns:
            oup: detect 결과 텐서 (B, output_dim)

        """

        inp_for_cnn = torch.transpose(inp, 1, 2)
        h = self.cnn(inp_for_cnn)
        post_h = self.fc(torch.flatten(h, start_dim=1))
        oup = self.fc_out_act(post_h)

        return oup

    def calc_loss(self, input, label):
        """ 분류 손실 함수를 계산

        Args:
            input: 데이터 텐서 (B, T, D)
            label: 라벨 텐서 (B, 1)

        Returns:
             loss: 스칼라 텐서

        """

        pred = self.forward(input)  # (B, C)

        if self.criterion == 'bce':
            loss = - torch.sum(label * torch.log(pred) + (1 - label) * torch.log(1 - pred), -1).mean()
        else:
            y_pred = torch.gather(pred, -1, label)
            loss = - (y_pred - torch.logsumexp(pred, dim=-1, keepdim=True)).mean()

        return loss

    @torch.no_grad()
    def calc_acc(self, input, label, **kwargs):
        """ 정확도 계산

        Args:
            input: 입력 텐서 (B, T, D)
            label: 라벨 텐서 (B, 1)

        Returns:
            acc: 정확도 float

        """

        self.eval()
        pred = self.forward(input)
        p_label = torch.argmax(pred, dim=-1, keepdim=True)
        acc = torch.sum(p_label == label) / pred.size(0)

        return acc.item()

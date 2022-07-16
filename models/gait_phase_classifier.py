import abc
import math

import torch
from torch import nn, optim

from models import identity
from models.networks import LayerNormLSTM, construct_mlp, Basic1DCNN


class Classifier(nn.Module, abc.ABC):

    def calc_loss(self, input, label):
        """ 분류 손실 함수를 계산

        Args:
            input: 데이터 텐서 (B, T, D)
            label: 원-핫 벡터 텐서 (B, C)

        Returns:
             loss: 스칼라 텐서

        """

        pred = self.forward(input)  # (B, C)
        y_pred = torch.sum(pred * label, -1)
        loss = - (y_pred - torch.logsumexp(pred, dim=-1)).mean()

        return loss

    @torch.no_grad()
    def calc_acc(self, input, label, **kwargs):
        """ 정확도 계산

        Args:
            input: 입력 텐서 (B, T, D)
            label: 라벨 텐서 (B, C)

        Returns:
            acc: 정확도 float

        """

        self.eval()
        pred = self.forward(input)
        p_label = torch.argmax(pred, dim=-1)
        label_idx = torch.argmax(label, dim=-1)
        acc = torch.sum(p_label == label_idx) / pred.size(0)

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

            normalization_type='none',
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
        super().__init__()
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
        oup = self.fc(torch.flatten(h, start_dim=1))

        return oup

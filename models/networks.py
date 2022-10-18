"""
Credits for https://github.com/eambutu/snail-pytorch/blob/master/src/blocks.py
            https://github.com/pytorch/pytorch/blob/cbcb2b5ad767622cf5ec04263018609bde3c974a/benchmarks/fastrnns/custom_lstms.py#L60-L83
"""
import inspect
import math
import numbers
from typing import Tuple

import torch
import torch.jit as jit
from torch import nn, Tensor
from torch.nn import functional as F

from models import activation_from_string, identity, _str_to_activation


class CausalConv1d(nn.Module):
    def __init__(self, in_channel_dim, out_channel_dim, dilation, kernel, stride=1):
        super(CausalConv1d, self).__init__()
        self.dilation = dilation
        self.padding = dilation * (kernel - 1)
        self.causal_conv = nn.Conv1d(
            in_channel_dim,
            out_channel_dim,
            dilation=dilation,
            kernel_size=kernel,
            stride=stride,
            padding=self.padding,
        )

    def forward(self, input):
        """
        :param input: shape of B x T x Di
        :return: shape of B x T x Do
        """
        input_permuted = torch.permute(input, (0, 2, 1))                            # B x Di x T
        output_permuted = self.causal_conv(input_permuted)[..., :-self.padding]     # B x Do x T

        return torch.permute(output_permuted, (0, 2, 1))


class DenseBlock(nn.Module):
    def __init__(self, in_channel_dim, out_channel_dim, dilation, kernel=2):
        super(DenseBlock, self).__init__()
        self.causal_conv1 = CausalConv1d(
            in_channel_dim,
            out_channel_dim,
            dilation,
            kernel,
        )
        self.causal_conv2 = CausalConv1d(
            in_channel_dim,
            out_channel_dim,
            dilation,
            kernel,
        )

    def forward(self, input):
        """
        :param input: shape of B x T x Di
        :return: shape of B x T x (Di + n_filters)
        """
        tanh = torch.tanh(self.causal_conv1(input))
        sig = torch.sigmoid(self.causal_conv2(input))
        out = torch.cat([input, tanh * sig], dim=-1)
        return out


class TCBlock(nn.Module):
    def __init__(self, channel_dim, n_layers, n_filters):
        super(TCBlock, self).__init__()
        blocks = []
        channel_count = channel_dim
        for layer in range(n_layers):
            block = DenseBlock(channel_count, n_filters, dilation=2 ** layer)
            blocks.append(block)
            channel_count += n_filters
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        """
        :param input: shape of B x T x Di
        :return: shape of B x T x (Di + n_layers * n_filters)
        """
        return self.blocks(input)


class AttentionBlock(nn.Module):
    def __init__(self, channel_dim, key_dim, value_dim, ln=False):
        super(AttentionBlock, self).__init__()
        self.key_layer = nn.Linear(channel_dim, key_dim)
        self.query_layer = nn.Linear(channel_dim, key_dim)
        self.value_layer = nn.Linear(channel_dim, value_dim)
        self.sqrt_k = math.sqrt(key_dim)

        self.key_ln = nn.LayerNorm(key_dim) if ln else None
        self.query_ln = nn.LayerNorm(key_dim) if ln else None
        self.value_ln = nn.LayerNorm(value_dim) if ln else None
        self.out_ln = nn.LayerNorm(value_dim) if ln else None

    def forward(self, input):
        """
        :param input: shape of B x T x Di
        :return: shape of B x T x (Di + Dv)
        """
        keys = self.key_layer(input)                            # B x T x Dk
        queries = self.query_layer(input)                       # B x T x Dk
        values = self.value_layer(input)                        # B x T x Dv
        keys = self.key_ln(keys) if self.key_ln else keys
        queries = self.query_ln(queries) if self.query_ln else queries
        values = self.value_ln(values) if self.value_ln else values

        logits = torch.bmm(queries, keys.transpose(2, 1))       # B x T x T
        mask = logits.data.new(logits.size(1), logits.size(2)).fill_(1).byte()
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(dim=0).expand_as(logits)
        logits.data.masked_fill_(mask.bool(), float('-inf'))
        probs = F.softmax(logits / self.sqrt_k, dim=2)          # B x T x 1
        read = torch.bmm(probs, values)                         # B x T x Dv
        read = self.out_ln(read) if self.out_ln else read
        return torch.cat([input, read], dim=2)


class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))

        # The layernorms provide learnable biases
        self.layernorm_i = nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = nn.LayerNorm(hidden_size)

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, batch_first):
        super(LayerNormLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)

        prev_size, next_size = input_size, hidden_size
        for idx in range(num_layers):
            self.__setattr__(f'cell{idx}', LayerNormLSTMCell(prev_size, next_size))
            prev_size, next_size = hidden_size, hidden_size

    def forward(self, input, state=None):
        """
        :param input: shape of B x T x D (batch_first == True), or T x B x D
        :param state: (hx, cx), shape of B x D
        :return: output: shape of B x T x D (batch_first == True), or T x B x D
        """
        if self.batch_first:
            T = input.size(1)
            input = torch.transpose(input, 0, 1)        # T x B x D
        else:
            T = input.size(0)

        if state is None:
            B = input.size(1)
            hx = input.data.new(self.num_layers, B, self.hidden_size).fill_(0.)
            cx = input.data.new(self.num_layers, B, self.hidden_size).fill_(0.)
        else:
            hx, cx = state

        outputs, h_states, c_states, = [], [], []
        for t in range(T):
            h_out = input[t]
            hx_ns = []
            cx_ns = []
            for idx in range(self.num_layers):
                h_out, (hx_n, cx_n) = self.__getattr__(f'cell{idx}')(h_out, (hx[idx], cx[idx]))
                if idx < self.num_layers - 1:           # except the last layer
                    h_out = self.dropout(h_out)
                hx_ns.append(hx_n)
                cx_ns.append(cx_n)
            hx = torch.stack(hx_ns, 0)
            cx = torch.stack(cx_ns, 0)
            outputs.append(h_out)
        output = torch.stack(outputs, 0)

        if self.batch_first:
            output = torch.transpose(output, 0, 1)

        return output, (hx, cx)


def construct_mlp(layers, input_dim, output_dim, norm_type='none', act_fcn='relu'):
    """ mlp 모듈 생성, 마지막 output layer 는 activation 과 normalization layer 없도록 구성

    Args:
        layers: hidden units 수 리스트
        input_dim: 입력 텐서 (B, Ni) 의 Ni
        output_dim: 출력 텐서 (B, No) 의 No
        norm_type: 'none' or 'batch' or 'layer' 로 normalization layer 선택
        act_fcn: 각 layer 의 activation function

    Returns:
         module: 구축된 mlp module

    """
    assert norm_type in ['none', 'batch', 'layer']
    act_fcn = activation_from_string(act_fcn)

    module = nn.Sequential()
    if layers is None or layers == []:
        module.add_module('layer0', nn.Linear(input_dim, output_dim))
        if norm_type == 'batch':
            module.add_module('bn0', nn.BatchNorm1d(output_dim))
        elif norm_type == 'layer':
            module.add_module('ln0', nn.LayerNorm(output_dim))
    else:
        mlp_in = input_dim
        for idx, mlp_out in enumerate(layers):
            module.add_module(f'layer{idx}', nn.Linear(mlp_in, mlp_out))
            if norm_type == 'batch':
                module.add_module(f'bn{idx}', nn.BatchNorm1d(mlp_out))
            elif norm_type == 'layer':
                module.add_module(f'ln{idx}', nn.LayerNorm(mlp_out))
            module.add_module(f'act{idx}', act_fcn)
            mlp_in = mlp_out
        module.add_module('layer_out', nn.Linear(mlp_in, output_dim))

    return module


class Basic1DCNN(nn.Module):
    def __init__(
            self,
            input_width,
            input_channels,
            kernel_sizes,
            n_channels,
            groups,
            strides,
            paddings,
            normalization_type='none',
            hidden_init=None,
            hidden_activation='relu',
            output_activation=identity,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
    ):
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert all([n_c % groups == 0 for n_c in n_channels])
        assert normalization_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max'}
        if pool_type == 'max':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        super().__init__()

        self.input_width = input_width
        self.input_channels = input_channels
        self.output_activation = output_activation
        if isinstance(hidden_activation, str):
            hidden_activation = activation_from_string(hidden_activation)
        self.hidden_activation = hidden_activation
        self.normalization_type = normalization_type
        self.pool_type = pool_type

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(n_channels, kernel_sizes, strides, paddings)
        ):
            conv = nn.Conv1d(
                input_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )
            if hidden_init:
                hidden_init(conv.weight)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

            if pool_type == 'max':
                if pool_sizes[i]:
                    self.pool_layers.append(
                        nn.MaxPool1d(
                            kernel_size=pool_sizes[i],
                            stride=pool_strides[i],
                            padding=pool_paddings[i],
                        )
                    )
                else:
                    self.pool_layers.append(None)

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
        )
        # find output dim of conv_layers by trial and add norm conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            test_mat = conv_layer(test_mat)
            if self.normalization_type == 'batch':
                self.conv_norm_layers.append(nn.BatchNorm1d(test_mat.shape[1]))
            if self.normalization_type == 'layer':
                self.conv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
            if self.pool_type != 'none':
                if self.pool_layers[i]:
                    test_mat = self.pool_layers[i](test_mat)

        self.output_shape = test_mat.shape[1:]  # ignore batch dim

    def forward(self, conv_input):
        return self.apply_forward_conv(conv_input)

    def apply_forward_conv(self, h):
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.normalization_type != 'none':
                h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none':
                if self.pool_layers[i]:
                    h = self.pool_layers[i](h)
            h = self.hidden_activation(h)
        return h

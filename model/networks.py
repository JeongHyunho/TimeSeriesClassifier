"""
Credits for https://github.com/eambutu/snail-pytorch/blob/master/src/blocks.py
            https://github.com/pytorch/pytorch/blob/cbcb2b5ad767622cf5ec04263018609bde3c974a/benchmarks/fastrnns/custom_lstms.py#L60-L83
"""

import math
import numbers
from typing import Tuple

import torch
import torch.jit as jit
from torch import nn, Tensor
from torch.nn import functional as F


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


class LayerNormLSTMCell(jit.ScriptModule):
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

    @jit.script_method
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
            hx = input.data.new(B, self.hidden_size).fill_(0.)
            cx = input.data.new(B, self.hidden_size).fill_(0.)
        else:
            hx, cx = state

        outputs, h_states, c_states, = [], [], []
        for t in range(T):
            h_out = input[t]
            for idx in range(self.num_layers):
                h_out, (hx, cx) = self.__getattr__(f'cell{idx}')(h_out, (hx, cx))
                if idx < self.num_layers - 1:           # except the last layer
                    h_out = self.dropout(h_out)
            outputs.append(h_out)
            h_states.append(hx)
            c_states.append(cx)
        output = torch.stack(outputs, 0)
        h_states = torch.stack(h_states, 0)
        c_states = torch.stack(c_states, 0)

        if self.batch_first:
            output = torch.transpose(output, 0, 1)
            h_states = torch.transpose(h_states, 0, 1)
            c_states = torch.transpose(c_states, 0, 1)

        return output, (h_states, c_states)

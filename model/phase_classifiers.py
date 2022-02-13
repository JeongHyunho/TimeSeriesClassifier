import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

TRANS_TEST_WINDOW = 10


class LSTMClassifier(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 1024,
                 n_lstm_layers: int = 1,
                 n_out_layers: int = 1,
                 n_out_nodes: int = 256,
                 out_act_fcn=torch.nn.Tanh,
                 lr: float = 1e-3,
                 p_drop: float = 0.2,
                 double_dtype: bool = False,
                 device: str = 'cuda'):
        super(LSTMClassifier, self).__init__()
        self.double_dtype = double_dtype
        self.device = device

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(p_drop)

        # output module
        if n_out_layers == 1:
            self.output_module = torch.nn.Sequential()
            self.output_module.add_module('out_layer0', nn.Linear(hidden_dim, output_dim))
        else:
            self.output_module = torch.nn.Sequential()
            for out_idx in range(n_out_layers):
                if out_idx == 0:
                    self.output_module.add_module(f'out_layer{out_idx}', nn.Linear(hidden_dim, n_out_nodes))
                elif out_idx < n_out_layers - 1:
                    self.output_module.add_module(f'out_layer{out_idx}', nn.Linear(n_out_nodes, n_out_nodes))
                else:
                    self.output_module.add_module(f'out_layer{out_idx}', nn.Linear(n_out_nodes, output_dim))

                if out_idx < n_out_layers - 1:
                    self.output_module.add_module(f'out_layer{out_idx}_act', out_act_fcn())

        for module in self.output_module:
            if module.__class__.__name__.find('Linear') != -1:
                nn.init.xavier_normal_(module.weight.data)
                nn.init.uniform_(module.bias.data)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if self.double_dtype:
            self.double()
        else:
            self.float()
        self.to(self.device)

    def fit_data(self, data):
        if type(data) == np.ndarray:
            data = torch.tensor(data)

        if self.double_dtype:
            data = data.double()
        else:
            data = data.float()
        data = data.to(self.device)

        return data

    def forward(self, series):
        """
            series: B x T x N
            p_pred: B x T x C
        """
        h, _ = self.lstm(series)
        h = self.dropout(h)
        z = self.output_module(h)
        p_pred = torch.softmax(z, -1)

        return p_pred

    def calc_loss(self, y, series):
        """
            y: B x T x C
            series: B x T x N
        """
        p_pred = self.forward(series)
        target = torch.argmax(y, -1)
        loss = self.criterion(torch.transpose(p_pred, 1, 2), target)

        return loss

    def calc_correct_ratio(self, y, series) -> float:
        """
            y: B x T x C
            series: B x T x N
        """
        p_pred = self.forward(series)       # shape: B x T x C
        y_pred = torch.argmax(p_pred, -1)
        y_true = torch.argmax(y, -1)
        n_correct = (y_pred == y_true).sum().item()
        correct_ratio = n_correct / (y.shape[0] * y.shape[1])

        return correct_ratio

    def calc_correct_transient(self, y, series) -> (int, int):
        """
            y: B x T x C
            series: B x T x N
        """
        n_classes = y.shape[-1]
        p_pred = self.forward(series)
        y_pred = torch.argmax(p_pred, -1)
        y_true = torch.argmax(y, -1)

        n_trans, n_correct = 0, 0
        for t_pred, t_true in zip(y_pred, y_true):      # type: torch.Tensor
            trans_indices = (t_true[1:] == torch.fmod(t_true[:-1] + 1, n_classes)).nonzero()

            for idx in trans_indices:
                start = max(0, idx - TRANS_TEST_WINDOW)
                end = min(idx + TRANS_TEST_WINDOW + 1, t_true.shape[0] + 1)

                n_trans += (end - start).item()
                n_correct += (t_pred[start:end] == t_true[start:end]).sum().item()

        return n_trans, n_correct

    def train_classifier(self,
                         epoch: int,
                         loader: DataLoader):
        train_loss, total_acc = 0., 0.
        n_trans_samples, n_corret_trans = 0, 0
        for data, labels in loader:
            data = self.fit_data(data)
            labels = self.fit_data(labels)

            self.train()
            loss = self.calc_loss(labels, data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.eval()
            with torch.no_grad():
                total_acc += 100 * self.calc_correct_ratio(labels, data)
                tran_samples, correct_tran = self.calc_correct_transient(labels, data)
                n_trans_samples += tran_samples
                n_corret_trans += correct_tran

            train_loss += loss.item()

        train_loss = train_loss / len(loader)
        total_acc = total_acc / len(loader)
        trans_acc = 100 * n_corret_trans / n_trans_samples

        print(f'(train) Epoch: {epoch}, Loss: {train_loss:.4f}, Acc: {total_acc:.1f}, Trans: {trans_acc:.1f}')

        return train_loss, total_acc, trans_acc

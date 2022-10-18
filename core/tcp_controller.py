from __future__ import annotations

import abc
import json
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt

from core.tcp_base import BaseTcp


class BaseController(BaseTcp, abc.ABC):
    """ Base controller via tcp connection """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, session_type='control', **kwargs)
        self.train_dir = self.main_dir.parent / 'train'

    @property
    def control_len(self):
        """ Length of control signal """
        raise NotImplementedError

    def save(self) -> Path:
        """ Save processed data or predictions """
        raise NotImplementedError


class ProsthesisController(BaseController):
    """ Controller for prosthesis """

    data_len = 14
    control_len = 1

    END_SIGNAL = 0
    NOT_READY = -1

    def __init__(
            self,
            model_dir: str,
            *args,
            model_file: str = 'model.pt',
            variant_file: str = 'variant.json',
            device = 'cpu',
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_file = self.train_dir / model_dir / model_file
        self.variant_file = self.train_dir / model_dir / variant_file
        self.device = device
        self.predictions = []

        assert self.model_file.exists(), f"model file doesn't exist"
        assert self.variant_file.exists(), f"variant file doesn't exist"

        # from variant file
        variant = json.loads(self.variant_file.read_text())
        self.model_arch = variant['arch']
        self.input_dim = variant['input_dim']

        if self.model_arch == 'cnn':
            self.input_width = variant['cnn']['input_width']
            self.t_input = torch.zeros(self.input_width, self.input_dim).to(device)     # buffer tensor for cnn

        # load torch model
        self.model = torch.load(self.model_file, map_location=device)
        self.n_received = 0

        # load standardization parameters
        stand_file = self.train_dir.joinpath('stand.json')
        if stand_file.exists():
            s_dict = json.loads(stand_file.read_text())
            self.inp_mean = torch.FloatTensor(s_dict['mean']).to(device)
            self.inp_std = torch.FloatTensor(s_dict['std']).to(device)
        else:
            self.logger.warn(f"standardization parameters not exiest at {stand_file}")
            self.inp_mean = torch.zeros(self.input_dim).to(device)
            self.inp_std = torch.ones(self.input_dim).to(device)

    def is_terminal(self, data) -> bool:
        return abs(data[-1] - self.END_SIGNAL) < 1e-6

    @torch.no_grad()
    def receive(self, data) -> (bool, tuple | int):
        self.model.eval()
        d_tensor = (torch.FloatTensor(data[:self.input_dim]).to(self.device) - self.inp_mean) / (self.inp_std + 1e-6)

        if self.model_arch == 'cnn':
            self.t_input = torch.cat([self.t_input[1:, ...], d_tensor[None, ...]], dim=0)
            if self.n_received >= self.input_width - 1:
                res = self.model(self.t_input[None, ...])[0, ...]
                pred = torch.argmax(res).item()
            else:
                pred = self.NOT_READY
        else:   # lstm
            res = self.model(d_tensor[None, None, ...], hc0=self.model.hc_n)[0, 0, ...]
            pred = torch.argmax(res).item()

        self.predictions.append(pred)
        self.n_received += 1

        terminal = self.is_terminal(data)
        if terminal:
            self.logger.info(f'terminal signal received!')

        return terminal, pred

    def save(self) -> Path:
        assert self.output_fmt == 'csv', NotImplementedError

        df = pd.DataFrame(self.predictions)
        df.to_csv(self.out_filename)

        return self.out_filename


class ArmCurlController(ProsthesisController):
    """ Controller for arm curl """

    data_len = 5
    control_len = 0
    NOT_READY = [-1, -1]
    data = []

    @torch.no_grad()
    def receive(self, data) -> (bool, tuple | int):
        self.model.eval()
        self.data.append(data)
        d_tensor = (torch.FloatTensor(data[:self.input_dim]).to(self.device) - self.inp_mean) / (self.inp_std + 1e-6)

        if self.model_arch == 'cnn':
            self.t_input = torch.cat([self.t_input[1:, ...], d_tensor[None, ...]], dim=0)
            if self.n_received >= self.input_width - 1:
                res = self.model(self.t_input[None, ...])[0, ...]
                pred = [p.item() for p in res]
            else:
                pred = self.NOT_READY
        else:   # lstm
            res = self.model(d_tensor[None, None, ...], hc0=self.model.hc_n)[0, 0, ...]
            pred = [p.item() for p in res]

        self.predictions.append(pred)
        self.n_received += 1

        terminal = self.is_terminal(data)
        if terminal:
            self.logger.info(f'terminal signal received!')

        return terminal, None

    def save(self) -> Path:
        assert self.output_fmt == 'csv', NotImplementedError

        df = pd.DataFrame(self.predictions)
        df.to_csv(self.out_filename)

        in_array = np.stack(self.data, axis=0)
        pred_array = np.stack(self.predictions, axis=0)

        theta = in_array[:, 2]
        torque = in_array[:, 3]
        pred_theta = pred_array[:, 0]
        pred_torque = pred_array[:, 1]

        fh = plt.figure(figsize=(4, 6))
        plt.subplot(2, 1, 1)
        plt.title(self.session_name + f' #{self.trial_idx}')
        plt.plot(np.vstack([theta, pred_theta]).T, label=['data', 'pred'])
        plt.legend()
        plt.ylabel('Theta')
        plt.subplot(2, 1, 2)
        plt.plot(np.vstack([torque, pred_torque]).T)
        plt.ylabel('Torque')
        plt.xlabel('index')
        fh.tight_layout()

        img_filename = self.main_dir.joinpath(self.trial_prefix + f'{self.trial_idx}.png')
        fh.savefig(img_filename)
        self.logger.info(f'plot saved at {img_filename}')
        plt.close()

        return self.out_filename

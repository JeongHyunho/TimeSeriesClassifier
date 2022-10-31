from __future__ import annotations

import abc
import datetime
import json
import math
import time
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
        self.log_dir = self.main_dir.parent / 'log'
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

    data_len = 16
    control_len = 2

    END_SIGNAL = 0
    NOT_READY = -1

    def __init__(
            self,
            model_dir: str,
            *args,
            model_file: str = 'model.pt',
            variant_file: str = 'variant.json',
            device: str = 'cpu',
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
        self.signal_type = variant['signal_type']

        if self.signal_type == 'all':
            self.input_dim = 8
            self.signal_rng = slice(0, 8)
        elif self.signal_type == 'eim':
            self.input_dim = 4
            self.signal_rng = slice(0, 4)
        elif self.signal_type == 'emg':
            self.input_dim = 4
            self.signal_rng = slice(4, 8)
        else:
            raise ValueError

        if self.model_arch == 'cnn':
            self.input_width = variant['cnn']['input_width']
            self.t_input = torch.zeros(self.input_width, self.input_dim).to(device)     # buffer tensor for cnn

        # load torch model
        self.model = torch.load(self.model_file, map_location=device)
        self.n_received = 0

        # tracked mean, variance
        self.mu = torch.zeros(self.input_dim).to(self.device)
        self.var = torch.ones(self.input_dim).to(self.device)

        if self.model_arch == 'lstm':
            self.model.hc_n = None

        # load standardization parameters
        stand_file = self.log_dir.joinpath('stand.json')
        if stand_file.exists():
            s_dict = json.loads(stand_file.read_text())
            self.inp_mean = torch.FloatTensor(s_dict['mean']).to(device)
            self.inp_std = torch.FloatTensor(s_dict['std']).to(device)
        else:
            self.logger.warn(f"standardization parameters doesn't exist at {stand_file}")
            self.inp_mean = torch.zeros(self.input_dim).to(device)
            self.inp_std = torch.ones(self.input_dim).to(device)

        self.time0 = datetime.datetime.now()

    def is_terminal(self, data) -> bool:
        return abs(data[-3] - self.END_SIGNAL) < 1e-6

    @torch.no_grad()
    def receive(self, data) -> (bool, tuple | int):
        self.model.eval()
        inp = torch.FloatTensor(data[self.signal_rng]).to(self.device)
        self.mu = ((self.n_received + 1) * self.mu + inp) / (self.n_received + 2)
        d_tensor = (torch.FloatTensor(data[:self.input_dim]).to(self.device) - self.inp_mean) / (self.inp_std + 1e-6)
        # d_tensor = (inp - self.mu) / (self.inp_std + 1e-6)

        t0 = time.time()
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
        elapse = time.time() - t0

        if pred != self.NOT_READY:
            self.logger.debug(f'received: {d_tensor}')
            self.logger.debug(f'tracked mu: {self.mu}')
            self.logger.debug(f'log prob: {res.cpu().numpy().tolist()}')
            self.logger.debug(f'elapse: {elapse}')

        self.predictions.append(pred)
        self.n_received += 1

        terminal = self.is_terminal(data)
        if terminal:
            self.logger.info(f'terminal signal received!')

        # pred to (speed, phase)
        if pred in [0, 5]:
            speed, phase = 0, 0
        elif pred < 5:
            speed, phase = 0, pred % 5
        else:
            speed, phase = 1, pred % 6
        self.logger.debug(f'send control signal {pred}, ({speed}, {phase})!')

        return terminal, (speed, phase)

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

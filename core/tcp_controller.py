from __future__ import annotations

import abc
from datetime import datetime
import json
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


class TorchController(BaseController):
    """ Torch model based controller """

    data_len: int
    control_len: int

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

        assert self.model_file.exists(), f"model file {self.model_file} doesn't exist!"
        assert self.variant_file.exists(), f"variant file {self.variant_file} doesn't exist!"

        # load variant file
        self.variant = json.loads(self.variant_file.read_text())
        self.model_arch = self.variant['arch']
        self.signal_type = self.variant['signal_type']

        # set input dimension and signal range from variant
        self.input_dim = self.get_input_dim()
        self.signal_rng = self.get_signal_rng()

        # set input window width
        if self.model_arch in ['cnn', 'mlp']:
            if self.model_arch == 'cnn':
                self.input_width = self.variant['cnn']['input_width']
            else:
                self.input_width = self.variant['mlp']['input_width']
            self.t_input = torch.zeros(self.input_width, self.input_dim).to(device)

        # load torch model
        self.model = torch.load(self.model_file, map_location=device)

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

        # be updated
        self.recv_data = []
        self.srt_time = datetime.now()
        self.num_received = 0
        self.num_correct = 0

    def get_input_dim(self) -> int:
        """ get the input data dimension from variant file """
        raise NotImplementedError

    def get_signal_rng(self) -> slice:
        """ get the input range in received data from variant file"""
        raise NotImplementedError

    @torch.no_grad()
    def receive(self, data) -> (bool, torch.Tensor):
        self.recv_data.append(data)
        self.num_received += 1
        self.logger.debug(f"#{self.num_received} received: {data}")

        self.model.eval()
        inp = torch.FloatTensor(data[self.signal_rng]).to(self.device)
        d_tensor = (inp - self.inp_mean) / (self.inp_std + 1e-6)

        t0 = time.time_ns()
        if self.model_arch in ['cnn', 'mlp']:
            self.t_input = torch.cat([self.t_input[1:, ...], d_tensor[None, ...]], dim=0)
            if self.num_received >= self.input_width:
                ret = self.model(self.t_input[None, ...])[0, ...]
            else:
                ret = None
        else:   # lstm
            ret = self.model(d_tensor[None, None, ...], hc0=self.model.hc_n)[0, 0, ...]
        elapse = 1e-9 * (time.time_ns() - t0)
        self.logger.debug(f"elapse: {elapse}")

        # termination protocol
        terminal = self.is_terminal(data)
        if terminal:
            self.logger.info(f'terminal signal received!')

        return terminal, ret


class ProsthesisController(TorchController):
    """ Controller for prosthesis """

    data_len = 16
    control_len = 2

    END_SIGNAL = 0
    NOT_READY = -1

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.recv_data = []
        self.pred_speed = []
        self.pred_phase = []

    def get_input_dim(self) -> int:
        if self.signal_type == 'all':
            input_dim = 8
        elif self.signal_type == 'eim':
            input_dim = 4
        elif self.signal_type == 'emg':
            input_dim = 4
        else:
            raise ValueError

        return input_dim

    def get_signal_rng(self) -> slice:
        if self.signal_type == 'all':
            signal_rng = slice(0, 8)
        elif self.signal_type == 'eim':
            signal_rng = slice(0, 4)
        elif self.signal_type == 'emg':
            signal_rng = slice(4, 8)
        else:
            raise ValueError

        return signal_rng

    def is_terminal(self, data) -> bool:
        return abs(data[-3] - self.END_SIGNAL) < 1e-6

    @torch.no_grad()
    def receive(self, data) -> (bool, tuple | int):
        terminal, ret = super().receive(data)

        if ret is not None:
            pred = torch.argmax(ret).item()
            self.logger.debug(f"log prob: {ret.cpu().numpy().tolist()}")
        else:
            pred = self.NOT_READY

        # get (speed, phase)
        if pred != self.NOT_READY:
            if pred in [0, 4]:
                speed, phase = 0, 0
            elif pred < 4:
                speed, phase = 0, pred
            else:
                speed, phase = 1, pred % 4
            self.logger.debug(f'send control signal {pred}, ({speed}, {phase})!')

        else:
            speed = self.NOT_READY
            phase = self.NOT_READY

        self.pred_speed.append(speed)
        self.pred_phase.append(phase)

        return terminal, (speed, phase)

    def save(self) -> Path:
        assert self.output_fmt == 'csv', NotImplementedError

        df = self.post_process()
        df.to_csv(self.out_filename)
        self.logger.info(f'data saved in {self.out_filename}')

        return self.out_filename

    def post_process(self) -> pd.DataFrame:
        recv_array = np.stack(self.recv_data, axis=0)
        pred_speed = np.array(self.pred_speed)
        pred_phase = np.array(self.pred_phase)

        # backup received data and predictions
        bk_filename = self.main_dir.joinpath(self.trial_prefix + f'{self.trial_idx}_bk.npz')
        np.savez(bk_filename, recv_array=recv_array, pred_speed=pred_speed, pred_phase=pred_phase)

        n_receives = len(recv_array)
        self.logger.info(f'total {n_receives} data received.')
        signal = recv_array[:, :8]
        foot_switch = recv_array[:, 8:11]
        speed = recv_array[:, 14].astype('i')
        true_phase = recv_array[:, 15].astype('i')

        pd_data = pd.DataFrame(
            np.concatenate([signal, true_phase[..., None], pred_phase[..., None]], axis=-1),
            columns=[*[f'signal{i}' for i in range(8)], 'true_phase', 'pred_phase'],
        )

        fh = plt.figure(figsize=(4, 10))
        plt.subplot(4, 1, 1)
        plt.title(self.session_name + f' #{self.trial_idx}' + f' Speed: {speed[0]}')
        plt.plot(signal[:, :4])
        plt.ylabel('EMG')
        plt.subplot(4, 1, 2)
        plt.plot(signal[:, 4:8])
        plt.ylabel('EIM')
        plt.subplot(4, 1, 3)
        plt.plot(foot_switch)
        plt.ylabel('Foot Switch')
        plt.subplot(4, 1, 4)
        plt.plot(np.array([true_phase, pred_phase]).T)
        plt.ylabel('True/Pred Phase')
        fh.tight_layout()

        img_filename = self.main_dir.joinpath(self.trial_prefix + f'{self.trial_idx}.png')
        fh.savefig(img_filename)
        plt.close(fh)

        return pd_data


class DummyProsthesisController(ProsthesisController):

    def __init__(
            self,
            model_dir: str,
            *args,
            model_file: str = 'model.pt',
            variant_file: str = 'variant.json',
            device: str = 'cpu',
            **kwargs,
    ):
        super(ProsthesisController, self).__init__(*args, **kwargs)

    def receive(self, data) -> (bool, tuple | int):
        terminal = self.is_terminal(data)
        speed = data[-2]
        phase = data[-1]

        self.logger.debug(f'received: {data}')
        self.logger.debug(f'send msg: {phase}')

        return terminal, (speed, phase)

    def save(self):
        pass


class ArmCurlController(BaseController):
    """ Controller for arm curl """

    data_len = 5
    control_len = 1

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


        pass

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

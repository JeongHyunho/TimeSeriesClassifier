import abc
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from core.tcp_base import BaseTcp
from core.util import yes_or_no


class BaseBuffer(BaseTcp, abc.ABC):
    """ Base buffer for writing data during an experiment """

    _out_filename = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, session_type='log', **kwargs)
        self.data = []

    def receive(self, data) -> bool:
        """ Save a single data and return terminal signal """

        self.data.append(data)
        self.logger.debug(f"#{len(self.data)} received: {data}")

        terminal = self.is_terminal(data)
        if terminal:
            self.logger.info(f'terminal signal received! total {len(self.data)} samples were received')

        return terminal

    def save(self, overwrite=False) -> Path:
        """ Save the received data in session directory """

        assert self.output_fmt == 'csv', NotImplementedError

        # handle overwriting
        if Path(self.out_filename).exists() and not overwrite:
            yes = yes_or_no(f'confirm to overwrite on {self.out_filename}')
            if not yes:
                self._out_filename = self.tmp_dir.joinpath(f'{self.creation_time}.{self.output_fmt}')

        pd_data = self.post_process()
        pd_data.to_csv(self.out_filename)
        self.logger.info(f'data saved in {self.out_filename}')

        return self.out_filename

    def post_process(self) -> pd.DataFrame:
        """ Post-process received data """
        raise NotImplementedError


class ProsthesisBuffer(BaseBuffer):
    """ Buffer for prosthesis experiment """

    DATA_LEN = 16
    PHASE_INFO = {'standing': 0, 'flat': 1, 'rising': 2, 'swing': 3}
    END_SIGNAL = 0

    def __init__(self, *args, preprocess='preprocess.json', **kwargs):
        super().__init__(*args, **kwargs)

        self.preproc_file = self.main_dir / preprocess
        if not self.preproc_file.exists():
            # create default preprocess file
            preproc_init = {"0": []}
            self.preproc_file.write_text(json.dumps(preproc_init))
            self.logger.info(f"created preprocess file: {self.preproc_file}")

    @property
    def data_len(self):
        return self.DATA_LEN

    def is_terminal(self, data) -> bool:
        return abs(data[-3] - self.END_SIGNAL) < 1e-6

    def post_process(self) -> pd.DataFrame:
        array = np.stack(self.data, axis=0)

        # backup all logged data
        bk_filename = self.main_dir.joinpath(self.trial_prefix + f'{self.trial_idx}_bk.npz')
        np.savez(bk_filename, array)

        # unfolding
        n_steps = len(array)
        self.logger.info(f'total {n_steps} steps recorded.')
        signal = array[:, :8]
        foot_switch = array[:, 8:11]
        angle = array[:, 11:13]
        speed = array[:, 14].astype('i')
        phase = array[:, 15].astype('i')

        # (idx-1) at phase transition
        trs_idx = np.flatnonzero(np.diff(phase) != 0)
        trs_idx = np.hstack([trs_idx, n_steps - 1])

        label = phase
        pd_data = pd.DataFrame(
            np.concatenate([signal, label[..., None]], axis=-1),
            columns=[*[f'signal{i}' for i in range(8)], 'label'],
        )

        # debug plot
        def draw_step_box(ax: plt.Axes):
            ylim = ax.get_ylim()
            last_trs = 0

            for idx in trs_idx:
                c = f"C{phase[idx]:.0f}"
                box = Rectangle((last_trs, ylim[0]), idx - last_trs + 1, ylim[1] - ylim[0], color=c, alpha=0.3)
                ax.add_patch(box)
                last_trs = idx

        fh = plt.figure(figsize=(4, 10))
        plt.subplot(4, 1, 1)
        plt.title(self.session_name + f' #{self.trial_idx}' + f' Speed: {speed[0]}')
        plt.plot(signal[:, :4])
        plt.ylabel('EMG')
        draw_step_box(plt.gca())
        plt.subplot(4, 1, 2)
        plt.plot(signal[:, 4:8])
        plt.ylabel('EIM')
        draw_step_box(plt.gca())
        plt.subplot(4, 1, 3)
        plt.plot(foot_switch)
        plt.ylabel('Foot Switch')
        draw_step_box(plt.gca())
        plt.subplot(4, 1, 4)
        plt.plot(angle)
        plt.ylabel('Angle')
        draw_step_box(plt.gca())
        fh.tight_layout()

        img_filename = self.main_dir.joinpath(self.trial_prefix + f'{self.trial_idx}.png')
        fh.savefig(img_filename)
        plt.close(fh)

        return pd_data


class ArmCurlBuffer(BaseBuffer):
    """ Buffer for arm curl experiment """

    DATA_LEN = 6
    END_SIGNAL = 0

    @property
    def data_len(self):
        return self.DATA_LEN

    def is_terminal(self, data) -> bool:
        return abs(data[-1] - self.END_SIGNAL) < 1e-6

    def post_process(self) -> pd.DataFrame:
        array = np.stack(self.data, axis=0)
        emg_signal = array[:, :2]
        hall_signal = array[:, 2]
        theta = array[:, 3]
        torque = array[:, 4]

        pd_data = pd.DataFrame(
            np.hstack([emg_signal, hall_signal[..., None], theta[..., None], torque[..., None]]),
            columns=['emg0', 'emg1', 'hall', 'theta', 'torque'],
        )

        fh = plt.figure(figsize=(4, 8))
        plt.subplot(4, 1, 1)
        plt.title(self.session_name + f' #{self.trial_idx}')
        plt.plot(emg_signal)
        plt.ylabel('EMG')
        plt.subplot(4, 1, 2)
        plt.plot(hall_signal)
        plt.ylabel('Hall Sensor')
        plt.subplot(4, 1, 3)
        plt.plot(theta)
        plt.ylabel('Theta')
        plt.subplot(4, 1, 4)
        plt.plot(torque)
        plt.ylabel('Torque')
        plt.xlabel('index')
        fh.tight_layout()

        img_filename = self.main_dir.joinpath(self.trial_prefix + f'{self.trial_idx}.png')
        fh.savefig(img_filename)
        self.logger.info(f'plot saved at {img_filename}')
        plt.close()

        return pd_data

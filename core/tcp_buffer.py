import abc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    DATA_LEN = 14
    TERRAIN_SIGNAL = {'standing': 0, 'even': 1, 'stair_up': 2, 'stair_down': 3, 'ramp_up': 4, 'ramp_down': 5}
    END_SIGNAL = 0

    def __init__(self, config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        # check configuration
        matched = [key for key in config['step_orders'].keys() if re.match(key, f'{self.trial_prefix}{self.trial_idx}')]
        assert len(matched) == 1, f"empty or overlapped 'step orders' for {self.trial_idx} in {config}"
        step_orders = config['step_orders'][matched[0]]

        valid_orders = [v for v in step_orders.values() if v == v]
        assert len(valid_orders) == len(set(valid_orders)), f'some step order of {self.trial_idx}-th trial are overlapped!'

        self.step_orders = step_orders
        self.logger.info(f'{self.trial_idx}-th {self.trial_prefix} step orders: '
                         f'stair_up: {step_orders["stair_up"]}, stair_down: {step_orders["stair_down"]}, '
                         f'ramp_up: {step_orders["ramp_up"]}, ramp_down: {step_orders["ramp_down"]}')

    @property
    def data_len(self):
        return self.DATA_LEN

    def is_terminal(self, data) -> bool:
        return abs(data[-1] - self.END_SIGNAL) < 1e-6

    def post_process(self) -> pd.DataFrame:
        array = np.stack(self.data, axis=0)
        signal = array[:, :8]
        foot_switch = array[:, 8:11]
        angle = array[:, 11:13]

        on_land = np.any(foot_switch > 0.5, axis=-1).astype('i')
        to_list = np.flatnonzero(np.diff(on_land) == -1)
        hs_list = np.flatnonzero(np.diff(on_land) == 1)

        n_steps = len(to_list)
        self.logger.info(f'total {n_steps} steps recorded.')

        last_to = 0
        label = self.TERRAIN_SIGNAL['even'] * np.ones(len(array))
        for i_step, to in enumerate(to_list):

            if i_step == 0:
                label[last_to:to] = self.TERRAIN_SIGNAL['standing']
            elif i_step == self.step_orders['stair_up']:
                label[last_to:to] = self.TERRAIN_SIGNAL['stair_up']
            elif i_step == self.step_orders['stair_down']:
                label[last_to:to] = self.TERRAIN_SIGNAL['stair_down']
            elif i_step == self.step_orders['ramp_up']:
                label[last_to:to] = self.TERRAIN_SIGNAL['ramp_up']
            elif i_step == self.step_orders['ramp_down']:
                label[last_to:to] = self.TERRAIN_SIGNAL['ramp_down']

            last_to = to

        label[hs_list[-1]:] = self.TERRAIN_SIGNAL['standing']

        pd_data = pd.DataFrame(
            np.concatenate([array[:, :-1], label[..., None]], axis=-1),
            columns=[*[f'signal{i}' for i in range(8)], *[f'switch{j}' for j in range(3)], 'angle0', 'angle1', 'label'],
        )

        # debug plot
        def draw_step_box(ax: plt.Axes):
            ylim = ax.get_ylim()
            last_to = 0

            for i_step, to in enumerate(to_list):
                if i_step == 0:
                    c = f"C{self.TERRAIN_SIGNAL['standing']}"
                elif i_step == self.step_orders['stair_up']:
                    c = f"C{self.TERRAIN_SIGNAL['stair_up']}"
                elif i_step == self.step_orders['stair_down']:
                    c = f"C{self.TERRAIN_SIGNAL['stair_down']}"
                elif i_step == self.step_orders['ramp_up']:
                    c = f"C{self.TERRAIN_SIGNAL['ramp_up']}"
                elif i_step == self.step_orders['ramp_down']:
                    c = f"C{self.TERRAIN_SIGNAL['ramp_down']}"
                else:
                    last_to = to
                    continue

                box = Rectangle((last_to, ylim[0]), to - last_to + 1, ylim[1] - ylim[0], color=c, alpha=0.3)
                ax.add_patch(box)
                last_to = to

            srt, end = hs_list[-1], len(array)
            box = Rectangle((srt, ylim[0]), end - srt + 1, ylim[1] - ylim[0],
                            color=f"C{self.TERRAIN_SIGNAL['standing']}", alpha=0.3)
            ax.add_patch(box)

        fh = plt.figure(figsize=(4, 8))
        plt.subplot(3, 1, 1)
        plt.title(self.session_name + f' #{self.trial_idx}')
        plt.plot(signal)
        plt.ylabel('EMG/EIM')
        draw_step_box(plt.gca())
        plt.subplot(3, 1, 2)
        plt.plot(foot_switch)
        plt.ylabel('Foot Switch')
        draw_step_box(plt.gca())
        plt.subplot(3, 1, 3)
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

    DATA_LEN = 5
    END_SIGNAL = 0

    @property
    def data_len(self):
        return self.DATA_LEN

    def is_terminal(self, data) -> bool:
        return abs(data[-1] - self.END_SIGNAL) < 1e-6

    def post_process(self) -> pd.DataFrame:
        array = np.stack(self.data, axis=0)
        signal = array[:, :2]
        theta = array[:, 2]
        torque = array[:, 3]

        pd_data = pd.DataFrame(
            np.concatenate([signal, theta[..., None], torque[..., None]], axis=-1),
            columns=['eim0', 'eim1', 'theta', 'torque'],
        )

        fh = plt.figure(figsize=(4, 8))
        plt.subplot(3, 1, 1)
        plt.title(self.session_name + f' #{self.trial_idx}')
        plt.plot(signal)
        plt.ylabel('EIM')
        plt.subplot(3, 1, 2)
        plt.plot(theta)
        plt.ylabel('Theta')
        plt.subplot(3, 1, 3)
        plt.plot(torque)
        plt.ylabel('Torque')
        plt.xlabel('index')
        fh.tight_layout()

        img_filename = self.main_dir.joinpath(self.trial_prefix + f'{self.trial_idx}.png')
        fh.savefig(img_filename)
        self.logger.info(f'plot saved at {img_filename}')
        plt.close()

        return pd_data

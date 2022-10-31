import logging
import re
from datetime import datetime
from pathlib import Path

from core import conf


class BaseTcp:
    """ Tcp application base class

        Directory hierarchy is 'session -> main(type) -> trial(prefix)'

    """

    _out_filename = None

    def __init__(
            self,
            session_name: str,
            session_type: str,
            trial_prefix: str = 'trial',
            trial_idx: int = None,
            output_dir: str = conf.OUTPUT_DIR,
            output_fmt: str = 'csv',
    ):
        self.session_name = session_name
        self.session_type = session_type
        self.main_dir = Path(output_dir) / session_name / session_type
        self.tmp_dir = Path(output_dir) / session_name / 'tmp'
        self.trial_prefix = trial_prefix
        self.output_fmt = output_fmt

        self.logger = logging.getLogger('.'.join(['session', session_name, session_type]))
        self.creation_time = datetime.now().strftime('%y%m%d-%H%M%S')

        # directory setup
        assert Path(output_dir).exists(), "output directory doesn't exist"
        if not self.main_dir.exists():
            self.main_dir.mkdir(parents=True, exist_ok=True)
        if not self.tmp_dir.exists():
            self.tmp_dir.mkdir(parents=True)

        # check the number of trials
        if trial_idx is not None:
            self.trial_idx = trial_idx
        else:
            out_files = sorted(self.main_dir.rglob(f'{trial_prefix}*.{self.output_fmt}'))
            if len(out_files) == 0:
                self.trial_idx = 0
            else:
                self.trial_idx = len(out_files)

        self.logger.info(f"{str(self)} starts!")

    def __str__(self):
        return f"{self.session_name} for {self.session_type} of {self.trial_idx}-th {self.trial_prefix}"

    @property
    def data_len(self) -> int:
        """ Number of single received data """
        raise NotImplementedError

    def receive(self, data):
        """ Receive one time-step data via tcp """
        raise NotImplementedError

    def is_terminal(self, data) -> bool:
        """ Determine to terminate session depending on received data"""
        raise NotImplementedError

    def save(self):
        """ Save received data or predictions """
        pass

    @property
    def out_filename(self) -> Path:
        """ Return output file path """

        if self._out_filename is None:
            out_filename = self.main_dir.joinpath(self.trial_prefix + f'{self.trial_idx}.{self.output_fmt}')
            self._out_filename = out_filename

        return self._out_filename

    @out_filename.setter
    def out_filename(self, new_filename: Path):
        """ Set new output file path"""
        self._out_filename = new_filename

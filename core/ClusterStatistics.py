import logging
from textwrap import dedent
from datetime import datetime, timedelta
from pathlib import Path

from typing import List

import pandas as pd


class ClusterStatistics:
    """ Statistics of clustered training """

    def __init__(self, config: List[dict], logger_root=''):
        self.config_df = pd.DataFrame(config)
        self.creation_time = datetime.now()
        self.logger = logging.getLogger(".".join([logger_root, 'stat']))

        self.config_df.insert(len(self.config_df.columns), 'state', [float('nan')] * len(self.config_df))
        self.config_df.insert(len(self.config_df.columns), 'node', [float('nan')] * len(self.config_df))

    def update_state(self, job_id, kv_dict: dict):
        idx = self.config_df['job_id'] == job_id
        for k, v in kv_dict.items():
            columns = self.config_df.columns
            if k not in columns:
                self.config_df.insert(len(columns) - 1, k, [float('nan')] * len(self.config_df))
            self.config_df.loc[idx, k] = v

    def get_monitor_board(self):
        # basic statistics
        num_total = len(self.config_df)
        num_pending = sum(self.config_df['state'] == 'PENDING')
        num_running = sum(self.config_df['state'] == 'RUNNING')
        num_complete = sum(self.config_df['state'] == 'COMPLETED')
        num_failed = sum(self.config_df['state'] == 'FAILED')
        dur_sec = (datetime.now() - self.creation_time).seconds

        board = dedent(f"""
        Pendings: {num_pending}, Runnings: {num_running}, Completes: {num_complete}/{len(self.config_df)}, Failed: {num_failed}
        Duration: {int(dur_sec/60):02d}:{dur_sec%60:02d} from {self.creation_time.strftime('%y/%m/%d-%H:%M:%S')}
        """)

        # remaining time estimation
        if num_complete > 0:
            remain_sec = int((num_total - num_complete) * dur_sec / num_complete)
            end_time = self.creation_time + timedelta(seconds=remain_sec)
            remain_info = f"{int(remain_sec/60):02d}:{remain_sec%60:02d} at {end_time.strftime('%y/%m/%d-%H:%M:%S')}"
        else:
            remain_info = 'NaN'

        board += f"Estimated remaining: {remain_info}\n"

        # running jobs info
        running_idx = self.config_df['state'] == 'RUNNING'
        board += f"Running statistics: \n{str(self.config_df[running_idx])}"

        # add failed job info
        failed_idx = self.config_df['state'] == 'FAILED'
        failed_jobs = [f'{str(job)}({node})' for job, node
                       in zip(self.config_df[failed_idx]['job_id'], self.config_df[failed_idx]['node'])]
        board += dedent(f"""
        Failed job ids: {', '.join(failed_jobs)}
        """)

        return board

    def __str__(self):
        return str(self.config_df)

    def not_completed(self) -> pd.DataFrame:
        idx = self.config_df['state'] != "COMPLETED"
        return self.config_df[idx]

    def save(self, filepath: Path):
        self.config_df.to_csv(filepath)
        self.logger.info(f"summary saved at {filepath}")

    @staticmethod
    def load(summary_file):
        pass

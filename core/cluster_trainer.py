from __future__ import annotations

import json
import logging
import textwrap
import time
from datetime import datetime

import pyslurm
from pathlib import Path

from core.ClusterStatistics import ClusterStatistics
from core.util import dot_map_dict_to_nested_dict, sample_config, list_of_dicts__to__dict_of_lists, yes_or_no, \
    nested_dict_to_dot_map_dict
from core import conf


class ClusterTrainer:
    """ Cluster torch module trainer """

    stat = None     # type: ClusterStatistics
    exp_dir = None  # type: Path

    def __init__(
            self,
            cluster: dict,
            output_dir: str | Path = conf.OUTPUT_DIR,
            train_session: str = 'train'
    ):
        self.cluster = cluster
        self.output_dir = Path(output_dir)
        self.train_session = train_session

        self.job = pyslurm.job()
        self.job_ids = []
        self.logger = logging.getLogger('session.trainer')
        self.creation_time = datetime.now().strftime('%y%m%d-%H%M%S')

    def run(self, train_py, name, num_samples, config, py_args='', overwrite=False):
        self.logger.info(f"train session of {name} started (#samples: {num_samples})")

        # directory setup, handle overwriting
        exp_dir = self.output_dir / name / self.train_session
        if exp_dir.exists() and not overwrite:
            yes = yes_or_no(f'confirm to overwrite on {exp_dir}')
            if not yes:
                exp_dir = self.output_dir / name / self.creation_time
        self.exp_dir = exp_dir
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f'output dir is {self.exp_dir}')

        # sample configurations and submit jobs
        config = dot_map_dict_to_nested_dict(config)
        config_samples = []
        for idx in range(num_samples):
            _config = sample_config(config)
            job_id = self.submit_job(train_py, _config, py_args)
            self.job_ids.append(job_id)
            _config['job_id'] = job_id

            # save train configuration
            job_dir = self.exp_dir.joinpath(f'job{job_id}')
            job_dir.mkdir(parents=True, exist_ok=True)
            _config['job_dir'] = str(job_dir)
            varaint_json = job_dir.joinpath('variant.json')
            varaint_json.write_text(json.dumps(_config, indent=True))

            _config = nested_dict_to_dot_map_dict(_config)
            config_samples.append(_config)

        config_list = list_of_dicts__to__dict_of_lists(config_samples)
        self.stat = ClusterStatistics(config=config_list, logger_root='session.trainer')
        self.logger.info('cluster configuration sampled:')
        print(self.stat)

        # monitoring loop, break by end or ctrl-c
        try:
            while True:
                time.sleep(2.0)
                terminated = self.update_all_jobs()
                print(self.stat.get_monitor_board())

                if terminated:
                    break

        except KeyboardInterrupt:
            self.logger.info('interrupt! terminate jobs ...')
            self.terminate_jobs()
            time.sleep(2.0)

            self.update_all_jobs()
            print(self.stat.get_monitor_board())
            self.stat.save(self.exp_dir.joinpath('summary.csv'))

        self.logger.info('run results: ')
        print(self.stat)

    def submit_job(self, py_file, c_dict, py_args) -> int:

        sh_script = f"""
        set -e
        JOB_ID=$SLURM_JOB_ID
        . $HOME/.bashrc
        while [ ! -d "{self.exp_dir}" ]
        do
          echo "wait for creating experiment directory"
          sleep 5s
        done
        mkdir -p "{self.exp_dir}/job$JOB_ID"
        """

        conda = self.cluster.get('conda')
        if conda is not None:
            sh_script += f"""
        . $HOME/anaconda3/etc/profile.d/conda.sh
        conda activate {conda}
            """

        args = f"--job_dir {self.exp_dir}/job$JOB_ID --log_dir {self.exp_dir.parent}/log"
        # add report file
        if self.cluster.get('report'):
            args += f" --report {self.cluster['report']}"

        sh_script += f"""
        srun python3 {py_file} '{json.dumps(c_dict)}' {args} {py_args}
        mv job$JOB_ID.out "{self.exp_dir}/job$JOB_ID/job$JOB_ID.out"  
        """

        job_opt = {
            "wrap": textwrap.dedent(sh_script),
            "output": "job%j.out",
        }
        job_opt.update(self.cluster['sbatch_kwargs'])
        job_id = self.job.submit_batch_job(job_opt)

        return job_id

    def update_all_jobs(self) -> bool:
        terminated = True

        for job_id in self.job_ids:
            try:
                job_state = self.job.find_id(job_id)[0]['job_state']
                node = self.job.find_id(job_id)[0]['nodes']
            except ValueError:
                continue
            self.stat.update_state(job_id, {'state': job_state})
            self.stat.update_state(job_id, {'node': node})

            if job_state in ['PENDING', 'COMPLETED', 'FAILED']:
                continue

            if self.cluster.get('report'):
                report = self.exp_dir / f'job{job_id}' / self.cluster['report']
                if report.exists():
                    txt = (self.exp_dir / f'job{job_id}' / self.cluster['report']).read_text()
                    try:
                        self.stat.update_state(job_id, json.loads(txt))
                    except json.JSONDecodeError:
                        pass

            if job_state == 'RUNNING':
                terminated = False

        return terminated

    def terminate_jobs(self):
        df_killed = self.stat.not_completed()

        for job_id in df_killed['job_id']:
            try:
                pyslurm.slurm_kill_job(job_id, 9)
            except ValueError:
                pass

        self.logger.info(f'total {len(df_killed)} jobs are terminated')

    def save_summary(self, sort_by=None):
        df = self.stat.config_df.dropna()

        if sort_by:
            df = df.sort_values(by=[sort_by])

        if self.exp_dir:
            exp_dir = self.exp_dir
        else:
            exp_dir = Path(__file__).parent
            self.logger.warning(f'no experiment directory is not set, summary will be save to {exp_dir}')

        df.to_csv(exp_dir / 'summary.csv')

import shutil

import pytest
import os
import socket
from pathlib import Path

from core.cluster_trainer import ClusterTrainer


@pytest.mark.skipif(condition='lab6079' != socket.gethostname(), reason="only lab6079 is allowed")
def test_estimator_cluster_train(data_path, train_py, train_config, cluster_config, tmp_path):
    (tmp_path / 'test' / 'log').mkdir(parents=True, exist_ok=True)
    shutil.copytree(data_path, tmp_path / 'test' / 'log', dirs_exist_ok=True)

    trainer = ClusterTrainer(cluster_config, output_dir=tmp_path)
    trainer.run(
        train_py=train_py,
        name='test',
        num_samples=train_config['num_samples'],
        config=train_config,
        py_args="--use_neptune --tagging debug test",
    )

    config_df = trainer.stat.config_df.dropna().sort_values(by=['best_val_loss'])
    for job_dir in config_df['job_dir'][2:]:
        for pt_file in Path(job_dir).rglob('*.pt'):
            os.remove(pt_file)

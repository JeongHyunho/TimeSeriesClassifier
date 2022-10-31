import pytest
import os
import socket
from pathlib import Path

from core.cluster_trainer import ClusterTrainer


@pytest.mark.skipif(condition='lab6079' != socket.gethostname(), reason="only lab6079 is allowed")
def test_classifier_cluster_train(make_session_fcn, train_py, train_config, cluster_config, tmp_path):
    make_session_fcn(session_name='test', output_dir=tmp_path)

    trainer = ClusterTrainer(cluster_config, output_dir=tmp_path)
    trainer.run(
        train_py=train_py,
        name='test',
        num_samples=train_config['num_samples'],
        config=train_config,
    )

    sorted_dir = trainer.stat.config_df.dropna().sort_values(by=['best_val_loss'])['job_dir']
    for job_dir in sorted_dir[2:]:
        for pt_file in Path(job_dir).rglob('*.pt'):
            os.remove(pt_file)

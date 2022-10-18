import json

from core.cluster_trainer import ClusterTrainer


def test_cluster_basic(basic_train_py, basic_train_config_json, basic_cluster_config_json, tmp_path):
    train_config = json.loads(basic_train_config_json.read_text())
    cluster_config = json.loads(basic_cluster_config_json.read_text())

    trainer = ClusterTrainer(cluster_config, output_dir=tmp_path)
    trainer.run(
        train_py=basic_train_py,
        name='test',
        num_samples=train_config['num_samples'],
        config=train_config,
    )

import os
from datetime import datetime
from functools import partial

import click
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from data import load_data
from model.ankle_estimator import ExtendedLSTMEstimator


def try_train(config):
    train_set, val_set, _ = load_data(config['batch_size'])

    model = ExtendedLSTMEstimator(input_dim=config['input_dim'],
                                  output_dim=config['output_dim'],
                                  n_locals=config['n_locals'],
                                  feature_dim=config['feature_dim'],
                                  n_lstm_layers=config['n_lstm_layers'],
                                  pre_layers=[config['n_pre_nodes']] * config['n_pre_layers'],
                                  post_layers=[config['n_post_nodes']] * config['n_post_layers'],
                                  p_drop=config['p_drop'],
                                  layer_norm=config['layer_norm'],
                                  lr=config['lr'],
                                  device='cuda')

    for epoch in range(1000):
        """ Training """
        train_loss = model.train_model(epoch, train_set)

        """ Validation """
        val_loss = model.train_model(epoch, val_set, evaluation=True)

        """ Temporally save a model"""
        trial_dir = tune.get_trial_dir()
        path = os.path.join(trial_dir, "model.pt")
        torch.save(model.state_dict(), path)

        tune.report(train_loss=train_loss, val_loss=val_loss)


@click.command()
@click.option('--target', default='Ankle')
def main(target):
    target = target[0].upper() + target[1:]

    config = {
        'input_dim': 8,
        'output_dim': 1,
        'batch_size': tune.choice([32, 64, 128]),
        'n_locals': tune.choice([5, 10, 20, 30]),
        'feature_dim': tune.choice([5, 10, 15, 20, 25]),
        'n_lstm_layers': tune.choice([1, 2]),
        'n_pre_nodes': tune.choice([32, 64, 128]),
        'n_pre_layers': tune.choice([0, 1, 2]),
        'n_post_nodes': tune.choice([32, 64, 128]),
        'n_post_layers': tune.choice([0, 1, 2]),
        'p_drop': tune.choice([0.0, 0.1, 0.3, 0.5, 0.8]),
        'layer_norm': tune.choice([False, True]),
        'lr': tune.loguniform(1e-4, 1e-2)
    }

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=1000,
        grace_period=20,
        reduction_factor=2)
    reporter = CLIReporter(metric_columns=["train_loss", "val_loss", "training_iteration"])
    result = tune.run(
        partial(try_train),
        name='R' + target + 'FromEmgExtendedLSTM' + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        resources_per_trial={"cpu": os.cpu_count() // 2, "gpu": 0.50},
        config=config,
        num_samples=2000,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial("val_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']:.4f}")

    best_config = best_trial.config
    _, _, test_set = load_data(best_config['batch_size'], device='cpu')
    best_logdir = result.get_best_logdir(metric='val_loss', mode='min')

    model = ExtendedLSTMEstimator.load_from_config(best_config,
                                                   os.path.join(best_logdir, "model.pt"),
                                                   map_location='cpu')
    model.eval()
    test_loss = model.train_model(-1, test_set)
    print(f"Best trial test set loss: {test_loss:.4f}")


if __name__ == '__main__':
    main()

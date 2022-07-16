import os
from datetime import datetime
from functools import partial

import click
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from data import load_estimator_data
from models.ankle_estimator import LSTMEstimator


def try_train(config, target):
    train_set, val_set, _ = load_estimator_data(config['batch_size'], target=target)

    model = LSTMEstimator(input_dim=config['input_dim'],
                          output_dim=config['output_dim'],
                          feature_dim=config['feature_dim'],
                          hidden_dim=config['hidden_dim'],
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
        'feature_dim': tune.choice([5, 10, 15, 20, 25]),
        'hidden_dim': tune.choice([16, 32, 64, 128, 256]),
        'n_lstm_layers': tune.choice([1, 2, 3]),
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
        partial(try_train, target=target),
        name='R' + target + 'FromEmgLSTM' + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        resources_per_trial={"cpu": os.cpu_count() // 8, "gpu": 0.11},
        config=config,
        num_samples=500,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True,
    )

    best_trial = result.get_best_trial("val_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']:.4f}")

    best_config = best_trial.config
    _, _, test_set = load_estimator_data(best_config['batch_size'], device='cpu', target=target)
    best_logdir = result.get_best_logdir(metric='val_loss', mode='min')

    model = LSTMEstimator.load_from_config(best_config,
                                           os.path.join(best_logdir, "model.pt"),
                                           map_location='cpu')
    model.eval()
    test_loss = model.train_model(-1, test_set)
    print(f"Best trial test set loss: {test_loss:.4f}")


if __name__ == '__main__':
    main()

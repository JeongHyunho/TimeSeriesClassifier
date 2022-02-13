import os
from datetime import datetime
from functools import partial

import click
import torch
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from data import load_data, get_ordered_combinations
from model.ankle_estimator import SnailEstimator


def try_train(config, target):
    train_set, val_set, _ = load_data(config['batch_size'], target=target)

    model = SnailEstimator(input_dim=config['input_dim'],
                           output_dim=config['output_dim'],
                           key_dims=config['key_value_dims'][0],
                           value_dims=config['key_value_dims'][1],
                           filter_dims=[config['filter_dim']] * (len(config['key_value_dims'][0]) - 1),
                           target_length=config['target_length'],
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

        if val_loss > 1e5:
            raise Exception(f'Too large val_loss: {val_loss}')


@click.command()
@click.option('--target', default='Ankle')
def main(target):
    target = target[0].upper() + target[1:]

    key_value_dims = []
    for n_interleaves in [2, 3, 4]:
        key_dims_list = get_ordered_combinations([50, 100, 200, 300], n_interleaves)
        value_dims_list = get_ordered_combinations([50, 100, 200, 300], n_interleaves)
        key_value_dims += [[key_dims, value_dims] for key_dims, value_dims in zip(key_dims_list, value_dims_list)]
    config = {
        'input_dim': 8,
        'output_dim': 1,
        'batch_size': tune.choice([16, 32, 64, 128]),
        'key_value_dims': tune.sample_from(lambda _: key_value_dims[np.random.randint(len(key_value_dims))]),
        'filter_dim': tune.choice([16, 32, 64, 128, 256]),
        'target_length': tune.choice([8, 16, 32, 64, 128]),
        'layer_norm': tune.choice([False, True]),
        'lr': tune.loguniform(1e-5, 1e-2)
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
        name='R' + target + 'FromEmgSnail' + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        resources_per_trial={"cpu": os.cpu_count() // 2, "gpu": 0.5},
        config=config,
        num_samples=300,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial("val_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']:.4f}")

    best_config = best_trial.config
    _, _, test_set = load_data(best_config['batch_size'], device='cpu')
    best_logdir = result.get_best_logdir(metric='val_loss', mode='min')

    model = SnailEstimator.load_from_config(best_config,
                                            os.path.join(best_logdir, "model.pt"),
                                            map_location='cpu')
    model.eval()
    test_loss = model.train_model(-1, test_set)
    print(f"Best trial test set loss: {test_loss:.4f}")


if __name__ == '__main__':
    main()

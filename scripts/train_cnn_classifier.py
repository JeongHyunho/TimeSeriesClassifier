import os
from datetime import datetime

import torch
from ray import tune
from ray.tune import CLIReporter

from data import load_classifier_data
from models.gait_detector import CNNDetector
from models.gait_phase_classifier import CNNClassifier
from scripts import ValStopper


def try_train(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set, val_set, test_set = load_classifier_data(config['batch_size'], device=device)

    kernel_sizes = [config['kernel_size0']] * config['n_conv_layer0'] \
                   + [config['kernel_size1']] * config['n_conv_layer1']
    n_channels = [72 * config['k_channel0']] * config['n_conv_layer0'] \
                 + [72 * config['k_channel1']] * config['n_conv_layer1']
    pool_sizes = [0] * (config['n_conv_layer0'] - 1) + [5] \
                 + [0] * (config['n_conv_layer1'] - 1) + [5]
    pool_strides = [None] * (config['n_conv_layer0'] - 1) + [5] \
                 + [None] * (config['n_conv_layer1'] - 1) + [5]
    pool_padding = [None] * (config['n_conv_layer0'] - 1) + [0] \
                 + [None] * (config['n_conv_layer1'] - 1) + [0]

    model = CNNClassifier(
        input_width=80,
        input_channels=72,
        kernel_sizes=kernel_sizes,
        n_channels=n_channels,
        groups=72,
        strides=[1] * (config['n_conv_layer0'] + config['n_conv_layer1']),
        paddings=['same'] * (config['n_conv_layer0'] + config['n_conv_layer1']),
        fc_layers=[config['n_fc_units']] * config['n_fc_layers'],
        output_dim=2,
        normalization_type=config['cnn_norm'],
        pool_type='max',
        pool_sizes=pool_sizes,
        pool_strides=pool_strides,
        pool_paddings=pool_padding,
        fc_norm=config['fc_norm'],
        device=device,
    )

    # dir setups
    trial_dir = tune.get_trial_dir()
    model_file = os.path.join(trial_dir, "model.pt")
    best_val_loss = float('inf')
    best_model_file = os.path.join(trial_dir, "best_model.pt")

    for epoch in range(500):
        """ Training """
        train_loss = model.train_model(epoch, train_set)

        """ Validation """
        val_loss = model.train_model(epoch, val_set, evaluation=True)

        """ Accuracy for monitoring """
        acc = model.calc_acc(test_set.dataset.samples, test_set.dataset.labels, method='vote')

        """ Save the best model """
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, best_model_file)

        """ Temporally save a model"""
        torch.save(model, model_file)

        tune.report(
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            test_acc=acc,
        )


def main():

    config = {
        'batch_size': tune.choice([8, 32, 64, 128]),
        'n_conv_layer0': tune.choice([1, 2, 3]),
        'n_conv_layer1': tune.choice([1, 2, 3]),
        'kernel_size0': tune.choice([3, 5, 7, 9, 11]),
        'kernel_size1': tune.choice([3, 5, 7, 9, 11]),
        'k_channel0': tune.choice([1, 2, 3]),
        'k_channel1': tune.choice([1, 2, 3]),
        'n_fc_units': tune.choice([8, 16, 32]),
        'n_fc_layers': tune.choice([1, 2, 3]),
        'cnn_norm': tune.choice(['none', 'batch', 'layer']),
        'fc_norm': tune.choice(['none', 'batch', 'layer']),
        'lr': tune.loguniform(1e-5, 1e-3),
    }

    reporter = CLIReporter(metric_columns=["train_loss", "val_loss", "best_val_loss", "test_acc", "training_iteration"])
    result = tune.run(
        try_train,
        name='CNNClassifier' + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        resources_per_trial={"cpu": os.cpu_count() // 8, "gpu": 0.12},
        config=config,
        num_samples=2000,
        stop=ValStopper(n_graces=20, max_itr=500),
        progress_reporter=reporter,
        checkpoint_at_end=True,
    )

    best_trial = result.get_best_trial("val_loss", "min", "all")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial best validation loss: {best_trial.last_result['best_val_loss']:.4f}")


if __name__ == '__main__':
    main()

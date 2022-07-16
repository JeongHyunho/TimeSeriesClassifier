import os
from datetime import datetime

import click
import torch
from ray import tune
from ray.tune import CLIReporter

from data import load_detector_data
from models.gait_detector import LSTMDetector
from scripts import ValStopper


def try_train(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set, val_set, test_set = load_detector_data(config['batch_size'], device=device)

    model = LSTMDetector(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        feature_dim=config['feature_dim'],
        hidden_dim=config['hidden_dim'],
        n_lstm_layers=config['n_lstm_layers'],
        pre_layers=[config['n_pre_nodes']] * config['n_pre_layers'],
        post_layers=[config['n_post_nodes']] * config['n_post_layers'],
        act_fcn=config['act_fcn'],
        p_drop=config['p_drop'],
        fc_norm=config['fc_norm'],
        criterion=config['criterion'],
        bidirectional=config['bidirectional'],
        lr=config['lr'],
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
            torch.save(model.state_dict(), best_model_file)

        """ Temporally save a model"""
        torch.save(model.state_dict(), model_file)

        tune.report(
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            test_acc=acc,
        )


@click.command()
def main():
    config = {
        'input_dim': 72,
        'output_dim': 6,
        'batch_size': tune.choice([32, 64, 128]),
        'feature_dim': tune.choice([5, 10, 15, 20, 25]),
        'hidden_dim': tune.choice([16, 32, 64, 128, 256]),
        'n_lstm_layers': tune.choice([1, 2, 3]),
        'n_pre_nodes': tune.choice([32, 64, 128]),
        'n_pre_layers': tune.choice([0, 1, 2]),
        'n_post_nodes': tune.choice([32, 64, 128]),
        'n_post_layers': tune.choice([0, 1, 2]),
        'act_fcn': tune.choice(['relu', 'tanh']),
        'p_drop': tune.choice([0.0, 0.3, 0.8]),
        'fc_norm': tune.choice(['none', 'layer']),
        'lr': tune.loguniform(1e-5, 1e-3),
        'criterion': 'cce',
        'bidirectional': tune.choice([False, True]),
    }

    reporter = CLIReporter(metric_columns=["train_loss", "val_loss", "best_val_loss", "test_acc", "training_iteration"])
    result = tune.run(
        try_train,
        name='LSTMDetector' + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        resources_per_trial={"cpu": os.cpu_count() // 8, "gpu": 0.11},
        config=config,
        num_samples=500,
        stop=ValStopper(n_graces=20),
        progress_reporter=reporter,
        checkpoint_at_end=True,
    )

    best_trial = result.get_best_trial("val_loss", "min", "all")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial best validation loss: {best_trial.last_result['best_val_loss']:.4f}")


if __name__ == '__main__':
    main()

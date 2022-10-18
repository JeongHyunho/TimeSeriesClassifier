import os
import sys

os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import torch
from pathlib import Path
from argparse import ArgumentParser

from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.joinpath('..')))
from models.gait_phase_classifier import CNNClassifier, LSTMClassifier, Classifier
from data import load_prosthesis_loaders


def load_model_cnn(train_dir, config) -> (Classifier, DataLoader, DataLoader, DataLoader):
    c_cnn = config['cnn']

    train_dl, val_dl, test_dl = load_prosthesis_loaders(
        log_dir=train_dir,
        batch_size=c_cnn['batch_size'],
        window_size=c_cnn['input_width'],
        overlap_ratio=config['overlap_ratio'],
        num_classes=config['output_dim'],
        device=config['device'],
    )

    cnn_kwargs = CNNClassifier.kwargs_from_config(config)
    model = CNNClassifier(**cnn_kwargs)

    return model, train_dl, val_dl, test_dl


def load_model_lstm(train_dir, config) -> (Classifier, DataLoader, DataLoader, DataLoader):
    c_lstm = config['lstm']

    train_dl, val_dl, test_dl = load_prosthesis_loaders(
        log_dir=train_dir,
        batch_size=c_lstm['batch_size'],
        window_size=c_lstm['window_size'],
        overlap_ratio=config['overlap_ratio'],
        num_classes=config['output_dim'],
        device=config['device'],
    )

    lstm_kwargs = LSTMClassifier.kwargs_from_config(config)
    model = LSTMClassifier(**lstm_kwargs)

    return model, train_dl, val_dl, test_dl


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='json string for sampled train configuration')
    parser.add_argument('--job_dir', type=str, help='slurm job directory')
    parser.add_argument('--log_dir', type=str, help='train data directory')
    parser.add_argument('--report', type=str, default=None, help='report file name')
    args = parser.parse_args()

    config = json.loads(args.config)
    job_dir = Path(args.job_dir)
    log_dir = Path(args.log_dir)

    if config['arch'] == 'cnn':
        model, train_dl, val_dl, test_dl = load_model_cnn(log_dir, config)
    elif config['arch'] == 'lstm':
        model, train_dl, val_dl, test_dl = load_model_lstm(log_dir, config)
    else:
        raise ValueError(f"unexpected 'arch' value: {config['arch']}")

    model_file = job_dir.joinpath('model.pt')
    best_model_file = job_dir.joinpath('best_model.pt')

    n_wait = 0
    best_val_loss = float('inf')
    for epoch in range(config['epoch']):
        train_loss = model.train_model(epoch, train_dl)
        val_loss = model.train_model(epoch, val_dl, evaluation=True)
        acc = model.calc_acc(test_dl.dataset.samples, test_dl.dataset.labels)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, best_model_file)
            n_wait = 0
        else:
            n_wait += 1

        if args.report:
            if not job_dir.exists():
                job_dir.mkdir(parents=True, exist_ok=True)
            submitted = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'test_acc': acc,
            }
            job_dir.joinpath(args.report).write_text(json.dumps(submitted))

        # early stop protocol
        if n_wait >= config['n_grace']:
            break

    torch.save(model, model_file)
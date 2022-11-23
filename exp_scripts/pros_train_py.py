import math
import os
import sys
from datetime import datetime, timedelta

from core.util import nested_dict_to_dot_map_dict
from exp_scripts.neptune_config import neptune_config

os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import torch
from pathlib import Path
from argparse import ArgumentParser

from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.joinpath('..')))
from models.gait_phase_classifier import CNNClassifier, LSTMClassifier, Classifier, MLPClassifier
from data import load_prosthesis_loaders

import neptune.new as neptune


def load_model_cnn(train_dir, config) -> (Classifier, DataLoader, DataLoader, DataLoader):
    c_cnn = config['cnn']

    train_dl, val_dl, test_dl = load_prosthesis_loaders(
        log_dir=train_dir,
        batch_size=c_cnn['batch_size'],
        window_size=c_cnn['input_width'],
        overlap_ratio=config['overlap_ratio'],
        num_classes=config['output_dim'],
        signal_type=config['signal_type'],
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
        signal_type=config['signal_type'],
        device=config['device'],
    )

    lstm_kwargs = LSTMClassifier.kwargs_from_config(config)
    model = LSTMClassifier(**lstm_kwargs)

    return model, train_dl, val_dl, test_dl


def load_model_mlp(train_dir, config) -> (Classifier, DataLoader, DataLoader, DataLoader):
    c_mlp = config['mlp']

    train_dl, val_dl, test_dl = load_prosthesis_loaders(
        log_dir=train_dir,
        batch_size=c_mlp['batch_size'],
        window_size=c_mlp['input_width'],
        overlap_ratio=config['overlap_ratio'],
        num_classes=config['output_dim'],
        signal_type=config['signal_type'],
        device=config['device'],
    )

    lstm_kwargs = MLPClassifier.kwargs_from_config(config)
    model = MLPClassifier(**lstm_kwargs)

    return model, train_dl, val_dl, test_dl


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='json string for sampled train configuration')
    parser.add_argument('--job_dir', type=str, help='slurm job directory')
    parser.add_argument('--log_dir', type=str, help='train data directory')
    parser.add_argument('--report', type=str, default=None, help='report file name')
    parser.add_argument('--time_limit', type=int, default=None, help='bound on train time')
    parser.add_argument('--use_neptune', action='store_true', default=False, help='use neptune logger')
    parser.add_argument('--creation_time', type=str, default=None, help='used as neptune tag')
    args = parser.parse_args()

    config = json.loads(args.config)
    job_dir = Path(args.job_dir)
    log_dir = Path(args.log_dir)

    if config['arch'] == 'cnn':
        model, train_dl, val_dl, test_dl = load_model_cnn(log_dir, config)
    elif config['arch'] == 'lstm':
        model, train_dl, val_dl, test_dl = load_model_lstm(log_dir, config)
    elif config['arch'] == 'mlp':
        model, train_dl, val_dl, test_dl = load_model_mlp(log_dir, config)
    else:
        raise ValueError(f"unexpected 'arch' value: {config['arch']}")

    model_file = job_dir.joinpath('model.pt')
    best_model_file = job_dir.joinpath('best_model.pt')

    if args.use_neptune:
        neptune_client = neptune.init(**neptune_config(config['target']), tags=[log_dir.parent.name, args.creation_time])
        neptune_client['job_dir'] = job_dir.name
        for k, v in nested_dict_to_dot_map_dict(config).items():
            neptune_client[k] = v

    start_time = datetime.now()
    n_wait = 0
    best_val_loss = float('inf')
    test_acc_at_best = float('inf')
    exit_method = "normal"
    for epoch in range(config['epoch']):
        train_loss = model.train_model(epoch, train_dl)
        val_loss = model.train_model(epoch, val_dl, evaluation=True)
        val_acc = model.calc_acc(val_dl.dataset.samples, val_dl.dataset.labels)
        test_acc = model.calc_acc(test_dl.dataset.samples, test_dl.dataset.labels)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_acc_at_best = test_acc
            torch.save(model, best_model_file)
            n_wait = 0
        else:
            n_wait += 1

        logged = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'test_acc_act_best': test_acc_at_best,
            'val_acc': val_acc,
            'test_acc': test_acc,
        }

        if args.report:
            job_dir.joinpath(args.report).write_text(json.dumps(logged))
        if args.use_neptune:
            for k, v in logged.items():
                neptune_client['train/' + k].log(v)

        # early stop protocol
        if n_wait >= config['n_grace']:
            exit_method = "early_stop"
            break
        elif math.isnan(train_loss):
            exit_method = "nan"
            break
        elif abs(train_loss) > 1e5 or abs(val_loss) > 1e5:
            exit_method = "explode"
            break
        elif args.time_limit and datetime.now() - start_time > timedelta(minutes=args.time_limit):
            exit_method = "time_out"
            break

    torch.save(model, model_file)

    # save confusion matrix
    train_conf = model.confusion_matrix_figure(train_dl.dataset.samples, train_dl.dataset.labels)
    train_conf.savefig(job_dir.joinpath('train_confmax.png'))
    val_conf = model.confusion_matrix_figure(val_dl.dataset.samples, val_dl.dataset.labels)
    val_conf.savefig(job_dir.joinpath('val_confmax.png'))
    test_conf = model.confusion_matrix_figure(test_dl.dataset.samples, test_dl.dataset.labels)
    test_conf.savefig(job_dir.joinpath('test_confmax.png'))

    if args.use_neptune:
        neptune_client["train/conf_matrix"].upload(str(job_dir.joinpath('train_confmax.png')))
        neptune_client["val/conf_matrix"].upload(str(job_dir.joinpath('val_confmax.png')))
        neptune_client["test/conf_matrix"].upload(str(job_dir.joinpath('test_confmax.png')))
        neptune_client["exit_method"] = exit_method
        neptune_client["total_params"] = sum(p.numel() for p in model.parameters())
        neptune_client.stop()

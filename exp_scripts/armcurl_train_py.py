import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import sys
from datetime import datetime, timedelta

from core.util import nested_dict_to_dot_map_dict
from exp_scripts.neptune_config import neptune_config

import json
import torch
from pathlib import Path
from argparse import ArgumentParser

import neptune.new as neptune

from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.joinpath('..')))
from models.armcurl_estimator import CNNEstimator, LSTMEstimator, MLPEstimator

from data import load_armcurl_loaders


def load_model_cnn(train_dir, config) -> (CNNEstimator, DataLoader, DataLoader, DataLoader):
    c_cnn = config['cnn']

    train_dl, val_dl, test_dl = load_armcurl_loaders(
        log_dir=train_dir,
        batch_size=c_cnn['batch_size'],
        window_size=c_cnn['input_width'],
        overlap_ratio=config['overlap_ratio'],
        signal_type=config['signal_type'],
        device=config['device'],
    )

    cnn_kwargs = CNNEstimator.kwargs_from_config(config)
    model = CNNEstimator(**cnn_kwargs)

    return model, train_dl, val_dl, test_dl


def load_model_lstm(train_dir, config) -> (LSTMEstimator, DataLoader, DataLoader, DataLoader):
    c_lstm = config['lstm']

    train_dl, val_dl, test_dl = load_armcurl_loaders(
        log_dir=train_dir,
        batch_size=c_lstm['batch_size'],
        window_size=c_lstm['window_size'],
        overlap_ratio=config['overlap_ratio'],
        signal_type=config['signal_type'],
        device=config['device'],
    )

    lstm_kwargs = LSTMEstimator.kwargs_from_config(config)
    model = LSTMEstimator(**lstm_kwargs)

    return model, train_dl, val_dl, test_dl


def load_model_mlp(train_dir, config) -> (MLPEstimator, DataLoader, DataLoader, DataLoader):
    c_mlp = config['mlp']

    train_dl, val_dl, test_dl = load_armcurl_loaders(
        log_dir=train_dir,
        batch_size=c_mlp['batch_size'],
        window_size=c_mlp['input_width'],
        overlap_ratio=config['overlap_ratio'],
        signal_type=config['signal_type'],
        device=config['device'],
    )

    mlp_kwargs = MLPEstimator.kwargs_from_config(config)
    model = MLPEstimator(**mlp_kwargs)

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

    if not job_dir.exists():
        job_dir.mkdir(parents=True, exist_ok=True)

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
    test_acc_at_best = [None, None]
    exit_method = "normal"
    for epoch in range(config['epoch']):
        train_loss, train_info = model.train_model(epoch, train_dl)
        val_loss, val_info = model.train_model(epoch, val_dl, evaluation=True)
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
            'train_theta_mse': train_info['out0'],
            'train_torque_mse': train_info['out1'],
            'val_loss': val_loss,
            'val_theta_mse': val_info['out0'],
            'val_torque_mse': val_info['out1'],
            'best_val_loss': best_val_loss,
            'test_theta_mse': test_acc[0],
            'test_torque_mse': test_acc[1],
            'test_theta_mse_best': test_acc_at_best[0],
            'test_torque_mse_best': test_acc_at_best[1],
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
        elif abs(train_loss) > 1e5 or abs(val_loss) > 1e5:
            exit_method = "explode"
            break
        elif args.time_limit and datetime.now() - start_time > timedelta(minutes=args.time_limit):
            exit_method = "time_out"
            break

    torch.save(model, model_file)
    del model

    if args.use_neptune:
        # postprocessing
        model = torch.load(best_model_file, map_location='cpu')
        if type(model) in [CNNEstimator, MLPEstimator]:
            w_size = model.input_width
        else:
            w_size = None
        img_files = model.post_process(
            test_dl.dataset, post_dir=job_dir, w_size=w_size, y_labels=['Theta', 'Torque'], device='cpu',
        )

        for idx, file in enumerate(img_files):
            neptune_client[f"train/post_process{idx}"].upload(str(file))

        neptune_client["exit_method"] = exit_method
        neptune_client.stop()

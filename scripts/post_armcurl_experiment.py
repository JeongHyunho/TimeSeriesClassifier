import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from core import conf
from data import ArmCurlDataset


def batch_by_window(inp_tensor: torch.Tensor, window_size: int) -> torch.Tensor:
    # inp_tensor: (T, D)
    time_length = inp_tensor.size(0)
    assert time_length > window_size, 'too short time length'
    net_in = torch.stack([inp_tensor[i:i+window_size, :] for i in range(time_length - window_size + 1)], dim=0)

    return net_in


@torch.no_grad()
def post_process(session_dir: Path, log_dir: Path, model_dir: Path):
    # post process directory setup
    post_dir = session_dir / 'post' / model_dir.name
    post_dir.mkdir(parents=True, exist_ok=True)

    # load train result
    variant_str = (model_dir / 'variant.json').read_text()
    varint = json.loads(variant_str)
    (post_dir / 'variant.json').write_text(variant_str)
    if varint['arch'] == 'mlp':
        w_size = varint['mlp']['input_width']
    elif varint['arch'] == 'cnn':
        w_size = varint['cnn']['input_width']
    else:
        w_size = varint['lstm']['window_size']
    overlap_ratio = varint['overlap_ratio']
    dataset = ArmCurlDataset(log_dir=log_dir, window_size=w_size, overlap_ratio=overlap_ratio, device='cpu')
    test_x, test_y = dataset.get_test_stream(device='cpu')

    # prediction
    model = torch.load(model_dir / 'best_model.pt', map_location='cpu')
    model.eval()
    for idx, (x, y) in enumerate(zip(test_x, test_y)):
        if varint['arch'] in ['mlp', 'cnn']:
            cnn_in = batch_by_window(x, w_size)
            pred = model(cnn_in)
            pred = torch.cat([torch.nan * torch.ones(w_size - 1, pred.size(dim=-1)), pred], dim=0)
        else:
            pred = model(x[None, ...])[0, ...]

        theta, torque = y.numpy().T
        theta_pred, torque_pred = pred.numpy().T

        theta_rmse = np.sqrt(np.nansum((theta - theta_pred) ** 2) / (1. - np.isnan(theta_pred)).sum())
        torque_rmse = np.sqrt(np.nansum((torque - torque_pred) ** 2) / (1. - np.isnan(torque_pred)).sum())

        # save as figure
        fh = plt.figure(figsize=(4, 6))
        plt.subplot(2, 1, 1)
        plt.title(f'test sample #{idx}\nTheta RMSE: {theta_rmse:.2f}\nTorque RMSE: {torque_rmse:.2f}')
        plt.plot(np.vstack([theta, theta_pred]).T, label=['data', 'pred'])
        plt.legend()
        plt.ylabel('Theta')
        plt.subplot(2, 1, 2)
        plt.plot(np.vstack([torque, torque_pred]).T)
        plt.ylabel('Torque')
        plt.xlabel('index')
        fh.tight_layout()

        img_filename = post_dir.joinpath(f'test_{idx}.png')
        fh.savefig(img_filename)
        print(f'#{idx} image saved to {img_filename}')
        plt.close()

        # save as csv
        df = pd.DataFrame(np.vstack([theta, theta_pred, torque, torque_pred]).T,
                          columns=['theta', 'theta_pred', 'torque', 'torque_pred'])
        csv_filename = post_dir.joinpath(f'test_{idx}.csv')
        df.to_csv(csv_filename)


if __name__ == '__main__':
    output_dir = Path(conf.OUTPUT_DIR)
    session_name = 'armcurl_1123'
    job_dir = 'job6653'

    session_dir = output_dir / session_name
    log_dir = session_dir / 'log'
    model_dir = session_dir / 'train' / job_dir
    assert session_dir.exists() and log_dir.exists() and model_dir.exists()

    post_process(session_dir, log_dir, model_dir)

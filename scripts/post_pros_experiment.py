import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from core import conf
from data import ArmCurlDataset, load_prosthesis_loaders, ProsDataset


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
    window_size = varint['mlp']['input_width']
    c_mlp = varint['mlp']
    dataset = ProsDataset(
        log_dir=log_dir,
        window_size=c_mlp['input_width'],
        overlap_ratio=varint['overlap_ratio'],
        num_classes=varint['output_dim'],
        signal_type=varint['signal_type'],
        device='cpu',
        test=True,
    )
    test_x, test_y = dataset.get_test_stream(device='cpu')

    # prediction
    model = torch.load(model_dir / 'best_model.pt', map_location='cpu')
    model.eval()
    for idx, (x, y) in enumerate(zip(test_x, test_y)):
        mlp_in = batch_by_window(x, window_size)
        _, pred = torch.max(model(mlp_in), dim=-1)
        label = y[window_size-1:].flatten()

        # save as figure
        fh = plt.figure()
        t_range = slice(30, 510)
        plt.plot(pred.numpy()[t_range], '-k')
        plt.plot(label.numpy()[t_range], '--b')
        # plt.plot(pred.numpy(), '-k')
        # plt.plot(label.numpy(), '--b')
        fh.tight_layout()
        # plt.axis('off')
        filename = post_dir / f'pred_{idx}.png'
        fh.savefig(filename)
        plt.close()


if __name__ == '__main__':
    output_dir = Path(conf.OUTPUT_DIR)
    session_name = 'log_1102_fast'
    job_dir = 'job6325'

    session_dir = output_dir / session_name
    log_dir = session_dir / 'log'
    model_dir = session_dir / 'train' / job_dir
    assert session_dir.exists() and log_dir.exists() and model_dir.exists()

    post_process(session_dir, log_dir, model_dir)

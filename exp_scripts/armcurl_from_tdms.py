import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

log_root = Path('/home/user/Dropbox/MATLAB_dropbox/TimeSeriesClassifier/output/')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help='log session name')
    args = parser.parse_args()

    log_dir = log_root / args.name / 'log'
    assert log_dir.exists(), f"no log directory: {log_dir}"
    tdms_dir = log_root / args.name / 'from_tdms'
    assert tdms_dir.exists(), f"no tdms directory: {tdms_dir}"

    files = list(tdms_dir.glob('*.csv'))
    print(f"{len(files)} trials are found in tdms directory: {tdms_dir}:")
    for file in files:
        print(file)

    if len(files) == 0:
        print('no files found')
        sys.exit()

    pd_list = []
    for idx, file in enumerate(files):
        array = np.loadtxt(str(file), delimiter=',')
        data_len = len(array)

        emg = array[:, 0:2]
        enc = array[:, 2]
        ang = array[:, 3:4]
        tor = array[:, 4]

        fh = plt.figure(figsize=(4, 10))
        plt.subplot(4, 1, 1)
        plt.title(f'{args.name} {file.name}')
        plt.plot(emg)
        plt.ylabel('EMG')
        plt.subplot(4, 1, 2)
        plt.plot(enc)
        plt.ylabel('Encoder')
        plt.subplot(4, 1, 3)
        plt.plot(ang)
        plt.ylabel('Angle')
        plt.subplot(4, 1, 4)
        plt.plot(tor)
        plt.ylabel('Torque')
        fh.tight_layout()

        img_filename = log_dir / f"trial{idx:02d}.png"
        fh.savefig(img_filename)
        plt.close()

        pd_data = pd.DataFrame(
            np.concatenate([emg, enc[..., None], ang, tor[..., None]], axis=-1),
            columns=['emg0', 'emg1', 'enc', 'angle', 'torque'],
        )
        filename = log_dir / f"trial{idx:02d}.csv"
        pd_data.to_csv(filename)

    print(f"total {len(files)} trials are saved")

import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

log_root = Path('/home/user/Dropbox/MATLAB_dropbox/TimeSeriesClassifier/output/')
assert log_root.exists(), f"root directory, {log_root} doesn't exist"


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help='log session name')
    parser.add_argument('--win_size', type=int, default=2000, help='split window length, default is 2000')
    args = parser.parse_args()

    log_dir = log_root / args.name / 'log'
    files = list(log_dir.glob(f'trial[0-9]*.csv'))
    print(f"{len(files)} trials are found:")
    for file in files:
        print(file)

    if len(files) == 0:
        print('no files found')
        sys.exit()

    # backup
    bk_dir = log_root / 'bk'
    if not bk_dir.exists():
        bk_dir.mkdir(exist_ok=True)
    for file in files:
        shutil.copy(str(file), str(bk_dir / file.name))

    # split
    pd_list = []
    for file in files:
        array = np.loadtxt(str(file), delimiter=',', skiprows=1)
        data_len = np.size(array, axis=0)

        for str_idx in range(0, data_len, args.win_size):
            end_idx = str_idx + args.win_size
            if end_idx > data_len:
                end_idx = data_len

            signal = array[str_idx:end_idx, 1:9]
            angle = array[str_idx:end_idx, 9:11]
            label = array[str_idx:end_idx, 11:12]

            pd_data = pd.DataFrame(np.hstack([signal, angle, label]),
                                   columns=[*[f'signal{i}' for i in range(8)], 'angle0', 'angle1', 'label'])
            pd_list.append(pd_data)

    ans = input(f"{len(pd_list)} trial will be created. proceed? y/n: ")
    if ans != 'y':
        sys.exit()

    # remove originals
    for file in files:
        os.remove(str(file))

    # save
    for idx, pd_data in enumerate(pd_list):
        file = log_dir / f'trial{idx:02d}.csv'
        pd_data.to_csv(file)

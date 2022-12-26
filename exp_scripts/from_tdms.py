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
    parser.add_argument('--fix_label', action='store_true', help='fix mis-label: (1,2)->1, 2->1, 3->2, 4->3')
    args = parser.parse_args()

    log_dir = log_root / args.name / 'log'
    assert log_dir.exists(), f"no log directory: {log_dir}"
    tdms_dir = log_root / args.name / 'from_tdms'
    assert tdms_dir.exists(), f"no tdms directory: {tdms_dir}"

    files = list(tdms_dir.glob('*_real.csv'))
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

        signal = array[:, 0:8]
        foot_switch = array[:, 8:11]
        angle = array[:, 11:13]
        speed = array[:, 14].astype('i')
        phase = array[:, 15].astype('i')

        # (idx-1) at phase transition
        trs_idx = np.flatnonzero(np.diff(phase) != 0)
        trs_idx = np.hstack([trs_idx, data_len - 1])

        label = phase
        if args.fix_label:
            label[label == 2] = 1
            label[label == 3] = 2
            label[label == 4] = 3

        pd_data = pd.DataFrame(
            np.concatenate([signal, angle, label[..., None]], axis=-1),
            columns=[*[f'signal{i}' for i in range(8)], 'angle0', 'angle1', 'label'],
        )

        # debug plot
        def draw_step_box(ax: plt.Axes):
            ylim = ax.get_ylim()
            last_trs = 0

            for idx in trs_idx:
                c = f"C{phase[idx]:.0f}"
                box = Rectangle((last_trs, ylim[0]), idx - last_trs + 1, ylim[1] - ylim[0], color=c, alpha=0.3)
                ax.add_patch(box)
                last_trs = idx


        fh = plt.figure(figsize=(4, 10))
        plt.subplot(4, 1, 1)
        plt.title(args.name + f' #{idx}' + f' Speed: {speed[0]}')
        plt.plot(signal[:, :4])
        plt.ylabel('EMG')
        draw_step_box(plt.gca())
        plt.subplot(4, 1, 2)
        plt.plot(signal[:, 4:8])
        plt.ylabel('EIM')
        draw_step_box(plt.gca())
        plt.subplot(4, 1, 3)
        plt.plot(foot_switch)
        plt.ylabel('Foot Switch')
        draw_step_box(plt.gca())
        plt.subplot(4, 1, 4)
        plt.plot(angle)
        plt.ylabel('Angle')
        draw_step_box(plt.gca())
        fh.tight_layout()

        img_filename = log_dir / f"trial{idx:02d}.png"
        fh.savefig(img_filename)
        plt.close()

        filename = log_dir / f"trial{idx:02d}.csv"
        pd_data.to_csv(filename)

    print(f"total {len(files)} trials are saved")

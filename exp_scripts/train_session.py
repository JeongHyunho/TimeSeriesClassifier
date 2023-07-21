import json
import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from core.cluster_trainer import ClusterTrainer
from core import conf

logging.basicConfig(
    format='[%(name)s] %(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    datefmt='%y/%m/%d %H:%M:%S',
)

cur_dir = Path(__file__).parent
creation_time = datetime.now().strftime("%y/%m/%d-%H:%M:%S")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help='session folder containing train data')
    parser.add_argument('target', type=str, choices=['pros', 'armcurl', 'golf'], help='target experiment type')
    parser.add_argument('--output_dir', type=str, default=conf.OUTPUT_DIR, help='logged session data directory')
    parser.add_argument('--cluster_config', type=str, default='cluster_config.json', help='slurm configuration')
    parser.add_argument('--signal_type', type=str, default=None, choices=['emg', 'eim', 'imu', 'bio', 'enc', 'all', None],
                        help='signal type to be use for training')
    parser.add_argument('--train_session', type=str, default='train', help='train session name')
    parser.add_argument('--clear_models', action='store_true', default=False, help='remove lower performance models')
    parser.add_argument('--use_neptune', action='store_true', default=False, help='use neptune logger')
    parser.add_argument('--tagging', nargs='+', type=str, default=None, help='used as neptune tag')
    args = parser.parse_args()

    logger = logging.getLogger('session')
    logger.info(f'train starts!, session name is {args.name}, output_dir is {args.output_dir}')

    # load configurations
    target = args.target
    train_py = cur_dir / f'{target}_train_py.py'
    train_config = json.loads((cur_dir / f'{target}_train_config.json').read_text())
    cluster_config = json.loads(Path(args.cluster_config).read_text())

    # directory setup
    output_dir = Path(args.output_dir)
    session_dir = output_dir / args.name
    assert session_dir.exists(), f"session doesn't exist {session_dir}"

    # create and run trainer
    py_args = ""
    if args.use_neptune:
        py_args += f" --use_neptune --creation_time {creation_time}"
    if train_config.get("time_limit"):
        py_args += f" --time_limit {train_config['time_limit']}"

    # change signal_type in train configuration
    if args.signal_type is not None:
        logger.info(f"'signal_type'({train_config['signal_type']}) is changed to {args.signal_type}")
        train_config['signal_type'] = args.signal_type
    py_args += f" --tagging {train_config['signal_type']}"

    # add tagging
    py_args += " " + " ".join(args.tagging)

    # do not save pt files
    if args.clear_models:
        py_args += f" --clear_models"

    trainer = ClusterTrainer(cluster_config, output_dir=output_dir, train_session=args.train_session)
    trainer.run(
        train_py=train_py,
        name=args.name,
        num_samples=train_config['num_samples'],
        config=train_config,
        py_args=py_args,
    )

    # show top 5 results and remove others
    try:
        config_df = trainer.stat.config_df.dropna().sort_values(by=['test_acc_at_best'])
    except KeyError:
        logger.info('no success train result exists')
        sys.exit()
    trainer.save_summary(sort_by='test_acc_at_best')
    sorted_dir = config_df['job_dir']
    logger.info(f'top-5 results: ')
    print(config_df[:5])

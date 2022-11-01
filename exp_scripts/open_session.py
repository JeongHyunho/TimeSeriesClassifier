import json
import logging
import socket
from argparse import ArgumentParser
from datetime import datetime

from _thread import start_new_thread
from pathlib import Path

from core import conf
from core.session import run_log_session, run_control_session
from core.tcp_buffer import ProsthesisBuffer, ArmCurlBuffer
from core.tcp_controller import ProsthesisController, ArmCurlController, DummyProsthesisController

logging.basicConfig(
    format='[%(name)s] %(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    datefmt='%y/%m/%d %H:%M:%S',
)

srt_time = datetime.now().strftime("%y/%m/%d-%H:%M%S")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('type', type=str, choices=['log', 'control'], help='operation mode')
    parser.add_argument('target', type=str, choices=['pros', 'armcurl'], help='experiment target')
    parser.add_argument('--name', type=str, default=f'session_{srt_time}', help='session name')
    parser.add_argument('--job_dir', type=str, default=None, help='trained model directory, required for control')
    parser.add_argument('--trial_idx', type=str, default=None, help='trial idx for both log and control')
    parser.add_argument('--debug', action='store_true', default=False, help='set logger debug')
    parser.add_argument('--dummy', action='store_true', default=False, help='flag for dummy controller')
    args = parser.parse_args()

    target = args.target
    if target == 'pros':
        buffer_cls = ProsthesisBuffer
        if args.dummy:
            controller_cls = DummyProsthesisController
        else:
            controller_cls = ProsthesisController
        buffer_kwargs = {}
    elif target == 'armcurl':
        buffer_cls = ArmCurlBuffer
        controller_cls = ArmCurlController
        buffer_kwargs = {}
    else:
        raise ValueError(f"unexpected experiment target, got {target}")

    se_name = args.name
    if args.type == 'control' and args.job_dir is None and not args.dummy:
        raise ValueError("for control session 'job_dir' is required")

    logger = logging.getLogger('session')
    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info(f'open tcp server on {conf.ADDR}:{conf.PORT}')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.bind((conf.ADDR, conf.PORT))
    sock.listen(5)

    while True:
        # wait client
        c_conn, c_addr = sock.accept()
        logger.info(f'accept connection from {c_addr}')

        logger.info(f'start {args.type} session ({se_name})!')
        log_config = json.loads(Path('pros_log_config.json').read_text())
        if args.type == 'log':
            buffer = buffer_cls(
                session_name=se_name,
                trial_idx=args.trial_idx,
                **buffer_kwargs,
            )
            # collect data via tcp connection
            start_new_thread(run_log_session, (c_conn, buffer))

        elif args.type == 'control':
            # receive data and return feedback control signal
            controller = controller_cls(
                session_name=se_name,
                model_dir=f'{args.job_dir}',
                model_file='best_model.pt',
                device='cuda',
            )
            start_new_thread(run_control_session, (c_conn, controller))

        else:
            raise ValueError

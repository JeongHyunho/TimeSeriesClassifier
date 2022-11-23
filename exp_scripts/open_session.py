import json
import logging
import socket
from argparse import ArgumentParser
from datetime import datetime

from _thread import start_new_thread
from multiprocessing.pool import ThreadPool
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
    parser.add_argument('--addr', type=str, default=conf.ADDR, help=f'manual tcp address, default is {conf.ADDR}')
    parser.add_argument('--recv_port', type=int, default=conf.RECV_PORT, help=f'tcp port for receiving, default is {conf.RECV_PORT}')
    parser.add_argument('--send_port', type=int, default=conf.SEND_PORT, help=f'tcp port for sending, default is {conf.SEND_PORT}')
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

    # setup for sockets for receiving/sending
    logger.info(f'open tcp server for receiving on {args.addr}:{args.recv_port}')
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_recv.bind((args.addr, args.recv_port))
    sock_recv.listen(5)

    logger.info(f'open tcp server for sending on {args.addr}:{args.send_port}')
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_send.bind((args.addr, args.send_port))
    sock_send.listen(5)

    def sock_accept(sock):
        conn, addr = sock.accept()
        print(f'{sock} accept!')
        return conn, addr

    while True:
        # wait client
        pool = ThreadPool(processes=1)
        ret = pool.map(sock_accept, [sock_recv, sock_send])
        pool.close()
        pool.join()
        print('pool ends')
        conn_recv, addr_recv = ret[0]
        conn_send, addr_send = ret[1]
        logger.info(f'accept connection for receiving data from {addr_recv}')
        logger.info(f'accept connection for sending signal from {addr_send}')

        logger.info(f'start {args.type} session ({se_name})!')
        log_config = json.loads(Path('pros_log_config.json').read_text())
        if args.type == 'log':
            buffer = buffer_cls(
                session_name=se_name,
                trial_idx=args.trial_idx,
                **buffer_kwargs,
            )
            # collect data via tcp connection
            start_new_thread(run_log_session, (conn_recv, conn_send, buffer))

        elif args.type == 'control':
            # receive data and return feedback control signal
            controller = controller_cls(
                session_name=se_name,
                model_dir=f'{args.job_dir}',
                model_file='best_model.pt',
                device='cpu',
            )
            start_new_thread(run_control_session, (conn_recv, conn_send, controller))

        else:
            raise ValueError

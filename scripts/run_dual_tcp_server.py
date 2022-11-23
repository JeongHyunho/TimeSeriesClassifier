import socket
import struct
from _thread import start_new_thread

import logging
from multiprocessing.pool import ThreadPool
from threading import Thread
from time import time_ns

SOCK_TIMEOUT_SEC = 10
SOCK_DATA_LEN = 1

logging.basicConfig(
    format='[%(name)s] %(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    datefmt='%y/%m/%d %H:%M:%S',
)


def on_single_client(c_recv: socket.socket, c_send: socket.socket):
    c_recv.settimeout(SOCK_TIMEOUT_SEC)
    n_receives = 0
    last_time = time_ns()

    while True:
        received = c_recv.recv(4 * SOCK_DATA_LEN)

        if received == b'':
            logger.info('socket connection broken')
            break

        while 0 < len(received) < 4 * SOCK_DATA_LEN:
            received += c_recv.recv(4 * SOCK_DATA_LEN - len(received))

        data = struct.unpack(f'>{SOCK_DATA_LEN}f', received)
        logger.info(f'({n_receives}) received {len(received)} bytes: {data}')
        cur_time = time_ns()
        logger.info(f'receive loop freq: {1e9 / (cur_time - last_time + 1e-6)}')
        last_time = cur_time
        n_receives += 1

        c_send.send(struct.pack(f'>{SOCK_DATA_LEN}f', *data))

    c_recv.close()
    c_send.close()


# open server socket
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_recv_info = ('192.168.0.5', 9105)
server_recv_info = ('127.0.0.1', 9105)
sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock_recv.bind(server_recv_info)
sock_recv.listen(100)

sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_send_info = ('192.168.0.5', 9106)
server_send_info = ('127.0.0.1', 9106)
sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock_send.bind(server_send_info)
sock_send.listen(100)

logger = logging.getLogger('dual_server')


while True:
    try:
        logger.info(f'wait connection ...')
        pool = ThreadPool(processes=1)
        ret = pool.map(lambda sock: sock.accept(), [sock_recv, sock_send])
        conn_recv, addr_recv = ret[0]
        conn_send, addr_send = ret[1]
        logger.info(f'connection for msg receiving accepted! {addr_recv}')
        logger.info(f'connection for msg sending accepted! {addr_send}')

    except TimeoutError:
        logger.error('Timeout during waiting connections! try again ...')
        continue

    except KeyboardInterrupt:
        logger.error('KeyboardInterrupt! close sockets ...')

        if 'conn_recv' in locals():
            conn_recv.close()
        if 'conn_send' in locals():
            conn_send.close()
        break

    # new connection starts
    start_new_thread(on_single_client, (conn_recv, conn_send))

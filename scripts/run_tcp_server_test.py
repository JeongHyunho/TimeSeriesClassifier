import os
import time
from datetime import datetime
import socket
import struct
from _thread import start_new_thread

import torch

IN_FEATURE_DIM = 14
LOG_FILE = './tcp_server.log'


def pretty_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def write_log(msg, reset=False):
    if reset:
        open(LOG_FILE, 'w').close()

    with open(LOG_FILE, 'a') as f:
        print(msg, file=f)


def on_single_client(c_socket, c_addr):
    receive_once = False
    n_receives = 0

    while True:
        time.sleep(0.01)
        received = c_socket.recv(4 * IN_FEATURE_DIM)

        # empty data received or not
        if not received == b'':
            last_received = datetime.now()
            data = struct.unpack(f'{IN_FEATURE_DIM}f', received)
            write_log(f"received #{n_receives} from {c_addr[0]}:{c_addr[1]} {pretty_now()}")

            try:
                # try to send a number to a client
                c_socket.send(struct.pack('i', 1))
            except BrokenPipeError:
                # close socket if the pipe is broken
                write_log(f'broken, close socket to {c_addr[0]}:{c_addr[1]} {pretty_now()}')
                c_socket.close()
                break

            receive_once = True
            n_receives += 1

        # time-out protocol
        if receive_once and (datetime.now() - last_received).seconds > 10:
            write_log(f'time-out, close socket to {c_addr[0]}:{c_addr[1]} {pretty_now()}')
            c_socket.close()
            break


# open server socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_info = ('143.248.66.79', 9105)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(server_info)
sock.listen(100)

os.close(os.open(LOG_FILE, os.O_CREAT))
write_log(f'start since {pretty_now()}', reset=True)

while True:
    # wait connection
    conn, addr = sock.accept()
    write_log(f'Client({addr[0]}:{addr[1]}) connected')

    # new connection starts
    start_new_thread(on_single_client, (conn, addr))

import os
import time
from datetime import datetime
import socket
import struct
from _thread import start_new_thread

import torch

WIN_SIZE = 80
IN_FEATURE_DIM = 72
LOG_FILE = './tcp_server.log'


def pretty_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def write_log(msg):
    with open(LOG_FILE, 'a') as f:
        print(msg, file=f)


def on_single_client(c_socket, c_addr):
    receive_once = False
    in_tensor = torch.zeros(WIN_SIZE, IN_FEATURE_DIM)
    n_receives = 0

    while True:
        time.sleep(0.01)
        received = c_socket.recv(4 * IN_FEATURE_DIM)

        # empty data received or not
        if not received == b'':
            last_received = datetime.now()
            data = struct.unpack('72f', received)
            write_log(f"received #{n_receives} from {c_addr[0]}:{c_addr[1]} {pretty_now()}")

            # insert received data into a window
            if n_receives < WIN_SIZE:
                in_tensor[n_receives] = torch.tensor(data)
            else:
                in_tensor = torch.cat([in_tensor[:WIN_SIZE - 1], torch.tensor([data])], dim=0)

            if n_receives >= WIN_SIZE:
                # model forward
                out_tensor = model(in_tensor.unsqueeze(0)).squeeze(0)
                pred = torch.argmax(out_tensor, -1).item()
            else:
                pred = -1

            try:
                # send result to a client
                c_socket.send(struct.pack('i', pred))
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


# load classifier model
model_filename = '/home/user/ray_results7124/CNNClassifier_2022-02-24_23-57-38/try_train_1cc2b_00190_190_batch_size=32,' \
                 'cnn_norm=layer,fc_norm=layer,k_channel0=3,k_channel1=3,kernel_size0=9,kernel_size1=11,' \
                 'lr=6_2022-02-25_00-35-44/best_model.pt'
model = torch.load(model_filename, map_location='cpu')

# open server socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_info = ('143.248.66.79', 9105)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(server_info)
sock.listen(100)


os.close(os.open(LOG_FILE, os.O_CREAT))
write_log(f'start since {pretty_now()}')

while True:
    # wait connection
    conn, addr = sock.accept()
    write_log(f'Client({addr[0]}:{addr[1]}) connected')

    # new connection starts
    start_new_thread(on_single_client, (conn, addr))

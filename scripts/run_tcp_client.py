import socket
import struct
import numpy as np

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('143.248.66.79', 9107))
print('Connected to the server')

time_length = 200
input_dim = 2

signal = np.random.randn(input_dim, time_length)
angle = np.random.randn(1, time_length)
torque = np.random.randn(1, time_length)
is_terminal = np.hstack([np.ones(time_length - 1), 0])
data = np.vstack([signal, angle, torque, is_terminal]).T

# send msg
for line in data:
    msg = struct.pack('>5f', *line)
    sock.send(msg)
    print(f'send a msg #{line}')

# connection close
sock.close()

# socket module import!
import pickle
import socket

# socket create and connection
import struct

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('143.248.66.79', 9105))
print('Connected to the server')

# send msg
data = struct.pack('72f', *([0.]*72))
for i in range(200):
    sock.send(data)
    print(f'send a msg #{i}')

    # if i > 80:
    received = sock.recv(4 * 1)
    pred = struct.unpack('i', received)
    print(f'received: {pred}')

# connection close
sock.close()

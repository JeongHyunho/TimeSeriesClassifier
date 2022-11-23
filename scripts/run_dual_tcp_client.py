import socket
import struct

sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_send.connect(('127.0.0.1', 9105))
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_recv.connect(('127.0.0.1', 9106))
print('Connected to the server')


for _ in range(100):
    msg = struct.pack('>f', 1.0)
    sock_send.send(msg)
    print(f'send->{msg}, data->1')
    received = sock_recv.recv(4)
    data = struct.unpack('>f', received)
    print(f'received->{received}, data->{data}')


# connection close
sock_send.close()
sock_recv.close()

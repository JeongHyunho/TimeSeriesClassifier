import logging
import struct
from socket import socket

from core.tcp_controller import BaseController
from core.tcp_buffer import BaseBuffer

TIMEOUT_SEC = 20


def run_log_session(c_recv: socket, c_send: socket, buffer: BaseBuffer):
    """Start log session on tcp server c_socket with client

    Args:
        c_recv (socket): socket for receiving data
        c_send (socket): socket for sending signal
        buffer (BaseBuffer): experiment data writing module

    """

    logger = logging.getLogger('session')
    c_recv.settimeout(TIMEOUT_SEC)
    c_send.settimeout(TIMEOUT_SEC)
    recv_len = 4 * buffer.data_len

    while True:
        received = c_recv.recv(recv_len)
        try:
            while 0 < len(received) < recv_len:
                logger.debug(f'original received: {len(received)}')
                received += c_recv.recv(recv_len - len(received))
            logger.debug(f'received: {len(received)}')
        except KeyboardInterrupt:
            logger.info('interrupt! escape logging loop ...')
            break

        if received == b'':
            logger.error('socket connection broken')
            raise RuntimeError('socket connection broken')

        else:
            # try to receive and pass msg of incorrect bytes
            try:
                data = struct.unpack(f'>{buffer.data_len}f', received)
            except struct.error as e:
                logger.warning(e)
                continue
            is_terminal = buffer.receive(data)

            # terminate log session
            if is_terminal:
                break

    # save received data
    buffer.save()
    c_recv.close()
    c_send.close()


def run_control_session(c_recv: socket, c_send: socket, controller: BaseController):
    """ Start control session on tcp server c_socket with client
    Receiving data stream and send control signal

    Args:
        c_recv (socket): socket for receiving data
        c_send (socket): socket for sending signal
        controller (BaseController): module that receives data and creates control signal

    """

    logger = logging.getLogger('session')
    c_recv.settimeout(TIMEOUT_SEC)
    c_send.settimeout(TIMEOUT_SEC)
    recv_len = 4 * controller.data_len

    while True:
        try:
            received = c_recv.recv(recv_len)
            while 0 < len(received) < recv_len:
                logger.debug(f'original received: {len(received)}')
                received += c_recv.recv(recv_len - len(received))

        except KeyboardInterrupt:
            logger.info('interrupt! escape control loop ...')
            break
        except TimeoutError:
            logger.info('timeout! close control session ...')
            break

        if received == b'':
            logger.error('socket connection broken')
            raise RuntimeError('socket connection broken')

        else:
            # try to receive and pass msg of incorrect bytes
            try:
                data = struct.unpack(f'>{controller.data_len}f', received)
            except struct.error as e:
                logger.warning(e)
                continue
            is_terminal, action = controller.receive(data)

            # send control signal to the client
            if action:
                if controller.control_len > 1:
                    c_send.sendall(struct.pack(f'>{controller.control_len}f', *action))
                else:
                    c_send.sendall(struct.pack(f'>f', action))

            # terminate control session
            if is_terminal:
                break

    # save control signals
    controller.save()
    c_recv.close()
    c_send.close()

import logging
import struct
from socket import socket

from core.tcp_controller import BaseController
from core.tcp_buffer import BaseBuffer


TIMEOUT_SEC = 20


def run_log_session(c_socket: socket, buffer: BaseBuffer):
    """Start log session on tcp server c_socket with client

    Args:
        c_socket (socket): socket object
        buffer (BaseBuffer): experiment data writing module

    """

    logger = logging.getLogger('session')
    c_socket.settimeout(TIMEOUT_SEC)

    while True:
        try:
            received = c_socket.recv(4 * buffer.data_len)
        except KeyboardInterrupt:
            logger.info('interrupt! escape control loop ...')
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
    c_socket.close()


def run_control_session(c_socket: socket, controller: BaseController):
    """ Start control session on tcp server c_socket with client
    Receiving data stream and send control signal

    Args:
        c_socket (socket): socket object
        controller (BaseController): module that receives data and creates control signal

    """

    logger = logging.getLogger('session')
    c_socket.settimeout(TIMEOUT_SEC)

    while True:
        try:
            received = c_socket.recv(4 * controller.data_len)
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
                    c_socket.send(struct.pack(f'>{controller.control_len}f', *action))
                else:
                    c_socket.send(struct.pack(f'>f', action))

            # terminate control session
            if is_terminal:
                break

    # save control signals
    controller.save()
    c_socket.close()

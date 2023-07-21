import sys
import socket
from _thread import start_new_thread

import logging

import numpy as np
import pytest

from core.session import run_log_session
from core.tcp_buffer import ProsthesisBuffer


@pytest.mark.parametrize('trial_prefix', ['trial', 'test'])
def test_prosthesis_buffer(stream_data, trial_prefix, tmp_path):
    for _ in range(15):
        buffer = ProsthesisBuffer(session_name='test', trial_prefix=trial_prefix, output_dir=tmp_path)

        for data in stream_data:
            buffer.receive(data)

        out_filename = buffer.save()
        assert out_filename.exists()


@pytest.mark.skipif(condition='-s' not in sys.argv, reason="can't test stdin without -s")
@pytest.mark.parametrize('overwrite', [False, True])
def test_buffer_overwrite(stream_data, overwrite, tmp_path):
    for _ in range(2):
        buffer = ProsthesisBuffer(session_name='test', trial_idx=1, output_dir=tmp_path)

        for data in stream_data:
            buffer.receive(data)

        out_filename = buffer.save(overwrite=overwrite)
        assert out_filename.exists()


def test_log_connect(stream_data, create_client_fn, address, port, data_len, log_debug, tmp_path):
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_recv.bind((address, recv_port))
    sock_recv.listen(5)

    sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_send.bind((address, send_port))
    sock_send.listen(5)

    logger = logging.getLogger('test')
    if log_debug:
        logger.setLevel(logging.DEBUG)

    start_new_thread(create_client_fn, ('receive',))
    conn_recv, _ = sock_recv.accept()
    start_new_thread(create_client_fn, ('send',))
    conn_send, _ = sock_send.accept()

    buffer = ProsthesisBuffer(session_name='test', output_dir=tmp_path)

    run_log_session(conn, buffer)
    assert np.all(np.abs(stream_data - np.array(buffer.data)) < 1e-6)
    ser_sock.close()

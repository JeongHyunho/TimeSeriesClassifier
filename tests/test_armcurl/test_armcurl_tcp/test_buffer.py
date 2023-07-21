import pytest
import socket
import numpy as np
import logging
from _thread import start_new_thread

from core.session import run_log_session
from core.tcp_buffer import ArmCurlBuffer


@pytest.mark.parametrize('trial_prefix', ['trial', 'test'])
def test_armcurl_buffer(stream_data, trial_prefix, tmp_path):
    for _ in range(15):
        buffer = ArmCurlBuffer(session_name='test', trial_prefix=trial_prefix, output_dir=tmp_path)

        for data in stream_data:
            buffer.receive(data)

        out_filename = buffer.save()
        assert out_filename.exists()


def test_log_connect(stream_data, create_client_fn, address, recv_port, send_port, log_debug, tmp_path):
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_recv.bind((address, recv_port))
    sock_recv.listen(5)

    sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_send.bind((address, send_port))
    sock_send.listen(5)

    logger = logging.getLogger('session')
    if log_debug:
        logger.setLevel(logging.DEBUG)

    start_new_thread(create_client_fn, ('receive',))
    conn_recv, _ = sock_recv.accept()
    start_new_thread(create_client_fn, ('send',))
    conn_send, _ = sock_send.accept()

    buffer = ArmCurlBuffer(session_name='test', output_dir=tmp_path)

    run_log_session(conn_recv, conn_send, buffer)
    assert np.all(np.abs(stream_data - np.array(buffer.data)) < 1e-6)
    conn_recv.close()
    conn_send.close()

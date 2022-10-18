import sys
import socket
from _thread import start_new_thread

import logging

import numpy as np
import pytest

from core.session import run_log_session
from core.tcp_buffer import ProsthesisBuffer


@pytest.mark.parametrize('trial_prefix', ['trial', 'test'])
def test_prosthesis_buffer(buffer_config, stream_data, trial_prefix, tmp_path):
    for _ in range(2):
        buffer = ProsthesisBuffer(config=buffer_config, session_name='test', trial_prefix=trial_prefix, output_dir=tmp_path)

        for data in stream_data:
            buffer.receive(data)

        out_filename = buffer.save()
        assert out_filename.exists()


@pytest.mark.skipif(condition='-s' not in sys.argv, reason="can't test stdin without -s")
@pytest.mark.parametrize('overwrite', [False, True])
def test_buffer_overwrite(buffer_config, stream_data, overwrite, tmp_path):
    for _ in range(2):
        buffer = ProsthesisBuffer(buffer_config, session_name='test', trial_idx=1, output_dir=tmp_path)

        for data in stream_data:
            buffer.receive(data)

        out_filename = buffer.save(overwrite=overwrite)
        assert out_filename.exists()


def test_log_connect(stream_data, create_client_fn, address, port, data_len, buffer_config,
                     log_debug, tmp_path):
    ser_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ser_sock.bind((address, port))
    ser_sock.listen(5)

    logger = logging.getLogger('session')
    if log_debug:
        logger.setLevel(logging.DEBUG)

    start_new_thread(create_client_fn, ())
    conn, _ = ser_sock.accept()
    buffer = ProsthesisBuffer(config=buffer_config, session_name='test', output_dir=tmp_path)

    run_log_session(conn, buffer)
    assert np.all(np.abs(stream_data - np.array(buffer.data)) < 1e-6)
    ser_sock.close()

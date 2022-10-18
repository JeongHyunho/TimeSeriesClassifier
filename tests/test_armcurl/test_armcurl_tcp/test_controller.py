import socket
import logging

from _thread import start_new_thread

from core.tcp_controller import ArmCurlController
from core.session import run_control_session


def test_armcurl_controller(create_train_dir_fcn, stream_data, use_gpu, tmp_path):
    model_dirs = create_train_dir_fcn(tmp_path / 'test')

    for model_dir in model_dirs:
        controller = ArmCurlController(
            session_name='test',
            model_dir=model_dir,
            output_dir=tmp_path,
            device='cuda' if use_gpu else 'cpu',
        )

        for data in stream_data:
            controller.receive(data)

        out_filename = controller.save()
        assert out_filename.exists()


def test_armcurl_connect(create_train_dir_fcn, stream_data, create_client_fn, address, port, data_len,
                         use_gpu, log_debug, tmp_path):
    model_dir = create_train_dir_fcn(tmp_path / 'test', n_jobs=1)[0]

    ser_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ser_sock.bind((address, port))
    ser_sock.listen(5)

    logger = logging.getLogger('test')
    if log_debug:
        logger.setLevel(logging.DEBUG)

    start_new_thread(create_client_fn, ('control',))
    conn, _ = ser_sock.accept()
    controller = ArmCurlController(
        session_name='test',
        model_dir=model_dir,
        output_dir=tmp_path,
        device='cuda' if use_gpu else 'cpu',
    )

    run_control_session(conn, controller)
    ser_sock.close()

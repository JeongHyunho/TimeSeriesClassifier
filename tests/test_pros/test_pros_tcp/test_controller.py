import socket
import logging

from _thread import start_new_thread

from core.tcp_controller import ProsthesisController
from core.session import run_control_session


def test_prosthesis_controller(create_train_dir_fcn, stream_data, use_gpu, tmp_path):
    model_dirs = create_train_dir_fcn(tmp_path / 'test')

    for model_dir in model_dirs:
        controller = ProsthesisController(
            session_name='test',
            model_dir=model_dir,
            output_dir=tmp_path,
            device='cuda' if use_gpu else 'cpu',
        )

        for data in stream_data:
            controller.receive(data)

        out_filename = controller.save()
        assert out_filename.exists()


def test_control_connect(create_train_dir_fcn, create_client_fn, address, recv_port, send_port,
                         use_gpu, log_debug, tmp_path):
    model_dir = create_train_dir_fcn(tmp_path / 'test', n_jobs=1)[0]

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
    controller = ProsthesisController(
        session_name='test',
        model_dir=model_dir,
        output_dir=tmp_path,
        device='cuda' if use_gpu else 'cpu',
    )

    run_control_session(conn_recv, conn_send, controller)
    conn_recv.close()
    conn_send.close()

from data import load_armcurl_loaders
from data.armcurl_dataset import ArmCurlDataset


def test_armcurl_dataset(make_session_fcn, input_dim, output_dim, tmp_path):
    make_session_fcn(session_name='test', output_dir=tmp_path)
    ds = ArmCurlDataset(
        log_dir=tmp_path / 'test' / 'log',
        window_size=80,
        overlap_ratio=0.3,
    )

    x, y = ds[0]
    assert x.shape == (80, input_dim) and y.shape == (80, output_dim)


def test_armcurl_cnn_loaders(make_session_fcn, batch_size, cnn_basic_kwargs, overlap_ratio,
                             signal_type, tmp_path):
    make_session_fcn(session_name='test', output_dir=tmp_path)
    train_dir = tmp_path / 'test' / 'log'

    train_dl, val_dl, test_dl = load_armcurl_loaders(
        log_dir=train_dir,
        batch_size=batch_size,
        window_size=cnn_basic_kwargs['input_width'],
        overlap_ratio=overlap_ratio,
        signal_type=signal_type,
        device=cnn_basic_kwargs['device'],
    )

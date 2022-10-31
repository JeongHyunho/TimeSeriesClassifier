from data.pros_dataset import ProsDataset


def test_pros_dataset(make_session_fcn, input_dim, output_dim, signal_type, tmp_path):
    make_session_fcn(session_name='test', output_dir=tmp_path)
    ds = ProsDataset(
        log_dir=tmp_path / 'test' / 'log',
        window_size=80,
        overlap_ratio=0.3,
        num_classes=output_dim,
        signal_type=signal_type,
    )

    x, y = ds[0]
    assert x.shape == (80, input_dim) and y.shape == (80, output_dim)

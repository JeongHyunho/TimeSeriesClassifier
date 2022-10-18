from data.armcurl_dataset import ArmCurlDataset


def test_armcurl_dataset(make_session_fcn, input_dim, output_dim, tmp_path):
    make_session_fcn(session_name='test', output_dir=tmp_path)
    ds = ArmCurlDataset(
        log_dir=tmp_path / 'test' / 'log',
        window_size = 80,
        overlap_ratio = 0.3,
    )

    x, y = ds[0]
    assert x.shape == (80, input_dim) and y.shape == (80, output_dim)

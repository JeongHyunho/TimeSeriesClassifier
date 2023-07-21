from data.golf_dataset import GolfDataset


def test_golf_dataset(data_path, input_dim, output_dim):
    ds = GolfDataset(
        log_dir=data_path,
        window_size=10,
        overlap_ratio=0.3,
    )

    x, y = ds[0]
    assert x.shape == (10, input_dim) and y.shape == (10, output_dim)

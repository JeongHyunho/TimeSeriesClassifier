import pytest

from core.tcp_buffer import ArmCurlBuffer


@pytest.mark.parametrize('trial_prefix', ['trial', 'test'])
def test_armcurl_buffer(stream_data, trial_prefix, tmp_path):
    for _ in range(2):
        buffer = ArmCurlBuffer(session_name='test', trial_prefix=trial_prefix, output_dir=tmp_path)

        for data in stream_data:
            buffer.receive(data)

        out_filename = buffer.save()
        assert out_filename.exists()

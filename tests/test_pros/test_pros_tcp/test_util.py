import json

from core.util import sample_config, dot_map_dict_to_nested_dict


def test_sample_config(basic_train_config_json):
    config_dict = json.loads(basic_train_config_json.read_text())
    config_dict = dot_map_dict_to_nested_dict(config_dict)
    sampled = sample_config(config_dict)

    def check_nested_dict(d_in, d_ref):
        for k, v in d_in.items():
            if type(v) is list:
                assert v in d_ref[k]
            elif type(v) is dict:
                check_nested_dict(v, d_ref[k])

    check_nested_dict(sampled, config_dict)

import collections
import random
from functools import reduce

import torch
from colorama import Fore, Style


def batch_by_window(t: torch.Tensor, w_size: int) -> torch.Tensor:
    """ make batch tensor by shifting every window of size 'w_size' """

    time_length = t.size(0)
    assert t.ndim == 2, f"Tensor of (T, D) is expected, but got {t.shape}"
    assert time_length >= w_size, 'too short time length'

    net_in = torch.stack([t[i:i+w_size, :] for i in range(time_length - w_size + 1)], dim=0)

    return net_in


def yes_or_no(msg) -> bool:
    """ query input and return true if yes else false """

    while True:
        cont = input(Fore.GREEN + f"{msg}, yes/no > ")

        # loop for different answer
        while cont.lower() not in ("yes", "no"):
            cont = input(Fore.GREEN + f"{msg}, yes/no > ")

        print(Style.RESET_ALL, end='')

        if cont == "yes":
            return True
        else:
            return False


def sample_config(config: dict) -> dict:
    sampled = {}

    for k, v in config.items():
        if not hasattr(v, '__iter__') or type(v) is str:
            sampled[k] = v
        elif type(v) is list:
            sampled[k] = random.choice(v)
        elif type(v) is dict:
            if k in sampled.values():
                sub_sampled = sample_config(v)
                sampled[k] = sub_sampled
        else:
            raise ValueError(f"unexpected type of configuration domain: ({k}:{v})")

    return sampled


def dot_map_dict_to_nested_dict(dot_map_dict):
    """Convert something like
    ```
    {
        'one.two.three.four': 4,
        'one.six.seven.eight': None,
        'five.nine.ten': 10,
        'five.zero': 'foo',
    }
    ```
    into its corresponding nested dict.
    http://stackoverflow.com/questions/16547643/convert-a-list-of-delimited-strings-to-a-tree-nested-dict-using-python
    """
    tree = {}

    for key, item in dot_map_dict.items():
        split_keys = key.split('.')
        if len(split_keys) == 1:
            if key in tree:
                raise ValueError("Duplicate key: {}".format(key))
            tree[key] = item
        else:
            t = tree
            for sub_key in split_keys[:-1]:
                t = t.setdefault(sub_key, {})
            last_key = split_keys[-1]
            if not isinstance(t, dict):
                raise TypeError(
                    "Key inside dot map must point to dictionary: {}".format(
                        key
                    )
                )
            if last_key in t:
                raise ValueError("Duplicate key: {}".format(last_key))
            t[last_key] = item

    return tree


def nested_dict_to_dot_map_dict(d, parent_key=''):
    """
    Convert a recursive dictionary into a flat, dot-map dictionary.

    :param d: e.g. {'a': {'b': 2, 'c': 3}}
    :param parent_key: Used for recursion
    :return: e.g. {'a.b': 2, 'a.c': 3}
    """
    items = []

    for k, v in d.items():
        new_key = parent_key + "." + k if parent_key else k
        if isinstance(v, dict):
            items.extend(nested_dict_to_dot_map_dict(v, new_key).items())
        else:
            items.append((new_key, v))

    return dict(items)


def list_of_dicts__to__dict_of_lists(lst, default_value=''):
    """
    ```
    x = [
        {'foo': 3, 'bar': 1},
        {'foo': 4, 'bar': 2},
        {'foo': 5},
        {'foo': 6, 'bar': 3},
        {'foo': 7},
    ]
    ppp.list_of_dicts__to__dict_of_lists(x, {'bar': 0})
    # Output:
    # {'foo': [3, 4, 5, 6, 7], 'bar': [1, 2, '', 3, '']}
    ```
    """

    if len(lst) == 0:
        return {}
    keys = reduce(set.union, [set(d.keys()) for d in lst])
    output_dict = collections.defaultdict(list)

    for d in lst:
        for k in keys:
            try:
                output_dict[k].append(d[k] if k in d.keys() else default_value)
            except KeyError:
                raise ValueError(f'no specified missing key, {k}')

    return output_dict

#!/usr/bin/env python
# Copyright 2021  Bofeng Huang

import json
import re

import collections


def load_json(filepath: str):
    with open(filepath) as f:
        data = json.load(f)
    return data


def flatten_nested(dictionary, parent_key=False, separator="."):
    """
    Turn a nested dictionary into a flattened dictionary
    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary
    """

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, collections.MutableMapping):
            items.extend(flatten_nested(value, new_key, separator).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten_nested({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


class TextNormalizer:
    def __init__(self, normalizer_file: str) -> None:
        normalizer_data = load_json(normalizer_file)
        self.pattern_upper_normalizer = self.load_upper_normalizer(normalizer_data)

    def load_upper_normalizer(self, normalizer_data: dict):
        # todo: exception
        # words_to_upper = set(map(lambda x: x.lower(), flatten_nested(normalizer_data["case"]["upper"]).values()))
        words_to_upper = set(flatten_nested(normalizer_data["case"]["upper"]).values())
        pattern_text_to_upper = r"(" + r"|".join(rf"\b{word_}\b" for word_ in words_to_upper) + r")"
        # if capitalized by model, re upper it here
        return re.compile(pattern_text_to_upper, flags=re.IGNORECASE)

    def __call__(self, s: str) -> str:
        s = self.pattern_upper_normalizer.sub(lambda match: match.group(1).upper(), s)
        return s

"""Compatibility tests for cache APIs used by the collision attack."""

import torch

from attack.collision import _cache_key_values


class _NewDynamicCache:
    """Minimal representation of the current Transformers DynamicCache API."""

    def __init__(self):
        self.layers = [(torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))]

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index):
        return self.layers[index]


class _LegacyCache:
    def __init__(self):
        self.key_cache = [torch.ones(1, 1, 2, 2)]
        self.value_cache = [torch.zeros(1, 1, 2, 2)]


def test_collision_cache_helper_reads_current_dynamic_cache_api():
    keys, values = _cache_key_values(_NewDynamicCache())

    assert len(keys) == len(values) == 1
    assert torch.equal(keys[0], torch.ones(1, 1, 2, 2))
    assert torch.equal(values[0], torch.zeros(1, 1, 2, 2))


def test_collision_cache_helper_keeps_legacy_cache_api():
    keys, values = _cache_key_values(_LegacyCache())

    assert len(keys) == len(values) == 1
    assert torch.equal(keys[0], torch.ones(1, 1, 2, 2))
    assert torch.equal(values[0], torch.zeros(1, 1, 2, 2))

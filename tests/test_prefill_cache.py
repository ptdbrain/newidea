"""Tests for canonicalizing model prefill cache outputs."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "inference"))

from pdsplit import normalize_past_key_values  # noqa: E402


class ModernCache:
    def __init__(self, layers):
        self.layers = layers

    def to_legacy_cache(self):
        return self.layers


def test_normalize_modern_cache_to_cpu_tuple() -> None:
    key = torch.ones(1, 2, 3, 4)
    value = torch.zeros(1, 2, 3, 4)

    result = normalize_past_key_values(
        ModernCache(((key, value),)), dtype=torch.float16
    )

    assert isinstance(result, tuple)
    assert result[0][0].device.type == "cpu"
    assert result[0][0].dtype == torch.float16
    assert result[0][1].dtype == torch.float16


def test_normalize_legacy_tuple_cache() -> None:
    key = torch.ones(1, 1, 2, 4)
    value = torch.zeros(1, 1, 2, 4)

    result = normalize_past_key_values(((key, value),), dtype=torch.float32)

    assert torch.equal(result[0][0], key)
    assert torch.equal(result[0][1], value)


def test_normalize_rejects_malformed_layer() -> None:
    with pytest.raises(ValueError, match="exactly key and value"):
        normalize_past_key_values(((torch.ones(1),),), dtype=torch.float16)


def test_normalize_rejects_wrong_rank() -> None:
    key = torch.ones(1, 2, 3)
    value = torch.zeros_like(key)

    with pytest.raises(ValueError, match=r"\[B, H, T, D\]"):
        normalize_past_key_values(((key, value),), dtype=torch.float16)


def test_normalize_rejects_non_finite_cache() -> None:
    key = torch.full((1, 1, 1, 1), float("nan"))
    value = torch.zeros_like(key)

    with pytest.raises(FloatingPointError, match="non-finite"):
        normalize_past_key_values(((key, value),), dtype=torch.float16)

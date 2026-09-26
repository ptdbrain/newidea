"""Tests for the 3-bit extension of KIVI's group quantizer (CPU, no Triton)."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.int3_quant import (  # noqa: E402
    pack_3bit,
    quantize_and_pack_along_last_dim,
    unpack_3bit,
    unpack_and_dequant,
)


def _kivi_formula(data: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    """KIVI's quantize->dequantize at ``bits``, without its Triton packing."""
    groups = data.reshape(*data.shape[:-1], -1, group_size)
    mn = groups.amin(-1, keepdim=True)
    scale = (groups.amax(-1, keepdim=True) - mn) / (2**bits - 1)
    codes = ((groups - mn) / scale).clamp(0, 2**bits - 1).round()
    return (codes.to(torch.float16) * scale + mn).reshape(data.shape)


def test_pack_unpack_roundtrip_is_dense():
    codes = torch.randint(0, 8, (2, 3, 5, 64))

    packed = pack_3bit(codes)

    assert packed.dtype == torch.uint8
    assert packed.shape == (2, 3, 5, 64 * 3 // 8)
    assert torch.equal(unpack_3bit(packed).long(), codes)


def test_pack_rejects_lengths_not_divisible_by_eight():
    with pytest.raises(ValueError, match="divisible by 8"):
        pack_3bit(torch.zeros(1, 12, dtype=torch.int32))


def test_quantize_matches_kivi_layout_and_formula():
    data = torch.randn(1, 2, 64, 96, dtype=torch.float16)

    code, scale, mn = quantize_and_pack_along_last_dim(data, 32)
    restored = unpack_and_dequant(code, scale.unsqueeze(-1), mn.unsqueeze(-1), 32)

    assert code.shape == (1, 2, 64, 96 * 3 // 8)
    assert scale.shape == mn.shape == (1, 2, 64, 3)
    assert scale.dtype == mn.dtype == torch.float16
    assert restored.shape == data.shape
    assert torch.equal(restored, _kivi_formula(data, 32, 3))
    assert int(unpack_3bit(code).max()) <= 7


def test_values_on_the_three_bit_grid_are_recovered_exactly():
    grid = torch.arange(8, dtype=torch.float32).repeat(4)  # min 0, max 7, scale 1
    data = (grid * 0.5 - 1.0).reshape(1, 1, 1, 32).half()

    code, scale, mn = quantize_and_pack_along_last_dim(data, 32)
    restored = unpack_and_dequant(code, scale.unsqueeze(-1), mn.unsqueeze(-1), 32)

    assert torch.equal(restored, data)


def test_three_bit_error_lies_between_two_and_four_bit():
    data = torch.randn(1, 4, 64, 128)

    def mse(restored):
        return torch.mean((restored.float() - data) ** 2)

    code, scale, mn = quantize_and_pack_along_last_dim(data, 32)
    three_bit = mse(unpack_and_dequant(code, scale.unsqueeze(-1), mn.unsqueeze(-1), 32))

    assert mse(_kivi_formula(data, 32, 4)) < three_bit < mse(_kivi_formula(data, 32, 2))

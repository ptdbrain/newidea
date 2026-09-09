"""Tests for the KIVI-to-Shadow cache representation adapter."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from kivi_adapter import (  # noqa: E402
    KIVIConfig,
    dequantize_cache,
    quantize_cache,
    roundtrip_cache,
)


def _cache(sequence_length: int, *, device: str = "cpu", dtype=torch.float16):
    """Create a two-layer cache with a KIVI-compatible head dimension."""
    return tuple(
        (
            torch.randn(1, 2, sequence_length, 64, device=device, dtype=dtype),
            torch.randn(1, 2, sequence_length, 64, device=device, dtype=dtype),
        )
        for _ in range(2)
    )


def _install_fake_kivi(monkeypatch):
    """Use a shape-preserving fake only for CPU partition/provenance tests."""
    import kivi_adapter

    calls = {"quantize": [], "dequantize": []}

    def fake_quantize(data, group_size, bit):
        calls["quantize"].append((tuple(data.shape), group_size, bit))
        features_per_int = 32 // bit
        code_shape = (*data.shape[:-1], data.shape[-1] // features_per_int)
        groups = data.shape[-1] // group_size
        parameter_shape = (*data.shape[:-1], groups)
        code = torch.zeros(code_shape, dtype=torch.int32, device=data.device)
        scale = torch.ones(parameter_shape, dtype=data.dtype, device=data.device)
        minimum = torch.zeros_like(scale)
        return code, scale, minimum

    def fake_dequantize(code, scale, minimum, group_size, bit):
        calls["dequantize"].append((tuple(code.shape), group_size, bit))
        features_per_int = 32 // bit
        shape = (*code.shape[:-1], code.shape[-1] * features_per_int)
        return torch.zeros(shape, dtype=scale.dtype, device=code.device)

    monkeypatch.setattr(
        kivi_adapter,
        "triton_quantize_and_pack_along_last_dim",
        fake_quantize,
    )
    monkeypatch.setattr(
        kivi_adapter,
        "unpack_and_dequant_vcache",
        fake_dequantize,
    )
    return calls


@pytest.fixture
def config():
    return KIVIConfig(k_bits=2, v_bits=2, group_size=32, residual_length=32)


def test_kivi_quantized_cache_preserves_layer_count(config, monkeypatch):
    calls = _install_fake_kivi(monkeypatch)

    quantized = quantize_cache(_cache(130), config)

    assert quantized["format"] == "kivi-native-v1"
    assert len(quantized["layers"]) == 2
    assert len(calls["quantize"]) == 4


def test_key_residual_partition(config, monkeypatch):
    _install_fake_kivi(monkeypatch)

    layer = quantize_cache(_cache(130), config)["layers"][0]

    assert layer["key_quantized_length"] == 128
    assert layer["key_residual"].shape[2] == 2
    assert layer["key_code"].shape == (1, 2, 64, 8)


def test_value_residual_partition(config, monkeypatch):
    _install_fake_kivi(monkeypatch)

    layer = quantize_cache(_cache(130), config)["layers"][0]

    assert layer["value_quantized_length"] == 98
    assert layer["value_residual"].shape[2] == 32
    assert layer["value_code"].shape == (1, 2, 98, 4)


def test_sequence_shorter_than_residual(config, monkeypatch):
    calls = _install_fake_kivi(monkeypatch)

    source = _cache(16)
    quantized = quantize_cache(source, config)
    layer = quantized["layers"][0]
    restored = dequantize_cache(quantized, config)[0]

    assert layer["key_code"] is None
    assert layer["value_code"] is None
    assert layer["key_residual"].shape[2] == 16
    assert layer["value_residual"].shape[2] == 16
    assert calls["quantize"] == []
    assert torch.equal(restored[0], source[0][0])
    assert restored[0].shape == (1, 2, 16, 64)


def test_exact_residual_multiple_uses_kivi_key_and_value_residual(config, monkeypatch):
    calls = _install_fake_kivi(monkeypatch)

    quantized = quantize_cache((_cache(32)[0],), config)
    layer = quantized["layers"][0]

    assert layer["key_quantized_length"] == 32
    assert layer["key_residual"].shape[2] == 0
    assert layer["value_quantized_length"] == 0
    assert layer["value_residual"].shape[2] == 32
    assert len(calls["quantize"]) == 1


def test_full_prompt_quantized_flushes_all_prompt_residuals(config, monkeypatch):
    _install_fake_kivi(monkeypatch)
    fq_config = KIVIConfig(
        k_bits=2,
        v_bits=2,
        group_size=32,
        residual_length=32,
        mode="full_prompt_quantized",
    )
    source = _cache(7)

    quantized = quantize_cache(source, fq_config)
    layer = quantized["layers"][0]
    restored = dequantize_cache(quantized, fq_config)

    assert quantized["mode"] == "full_prompt_quantized"
    assert quantized["fp_residual_prompt_tokens"] == 0
    assert layer["key_quantized_length"] == 32  # padded outside the prompt
    assert layer["key_valid_length"] == 7
    assert layer["key_residual"].shape[2] == 0
    assert layer["value_quantized_length"] == 7
    assert layer["value_residual"].shape[2] == 0
    assert restored[0][0].shape == source[0][0].shape
    assert restored[0][1].shape == source[0][1].shape


def test_kivi_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        KIVIConfig(2, 2, 32, 32, mode="not-a-mode")


def test_adapter_calls_public_kivi_dequantizer(config, monkeypatch):
    calls = _install_fake_kivi(monkeypatch)
    quantized = quantize_cache(_cache(130), config)

    restored = dequantize_cache(quantized, config)

    assert len(restored) == 2
    assert len(calls["dequantize"]) == 4


def test_roundtrip_preserves_shape_and_finite_values(config, monkeypatch):
    _install_fake_kivi(monkeypatch)
    source = _cache(130)

    restored = roundtrip_cache(source, config)

    assert len(restored) == len(source)
    for (key, value), (source_key, source_value) in zip(restored, source):
        assert key.shape == source_key.shape
        assert value.shape == source_value.shape
        assert torch.isfinite(key).all()
        assert torch.isfinite(value).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="KIVI public kernels require CUDA")
def test_real_kivi_roundtrip_on_cuda(config):
    source = _cache(130, device="cuda")

    restored = roundtrip_cache(source, config)

    for (key, value), (source_key, source_value) in zip(restored, source):
        assert key.shape == source_key.shape
        assert value.shape == source_value.shape
        assert torch.isfinite(key).all()
        assert torch.isfinite(value).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="KIVI public kernels require CUDA")
def test_four_bit_roundtrip_has_no_more_mse_than_two_bit():
    source = _cache(96, device="cuda")[0]
    two_bit = roundtrip_cache(
        (source,), KIVIConfig(k_bits=2, v_bits=2, group_size=32, residual_length=32)
    )[0]
    four_bit = roundtrip_cache(
        (source,), KIVIConfig(k_bits=4, v_bits=4, group_size=32, residual_length=32)
    )[0]

    two_mse = sum(
        torch.mean((reconstructed - original).float().square())
        for reconstructed, original in zip(two_bit, source)
    )
    four_mse = sum(
        torch.mean((reconstructed - original).float().square())
        for reconstructed, original in zip(four_bit, source)
    )
    assert four_mse <= two_mse

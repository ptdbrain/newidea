"""Tests for cache pipeline composition."""

from pathlib import Path
import sys

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.cache_pipeline import (  # noqa: E402
    CachePipeline,
    IdentityPipeline,
    KIVIPipeline,
    KVCloakKIVIPipeline,
)
from src.kivi_adapter import KIVIConfig  # noqa: E402


def _cache():
    return ((torch.ones(1, 1, 8, 4), torch.zeros(1, 1, 8, 4)),)


def test_identity_pipeline_returns_original_cache():
    source = _cache()

    assert IdentityPipeline().roundtrip(source) is source


def test_cache_pipeline_is_an_interface():
    try:
        CachePipeline().roundtrip(_cache())
    except NotImplementedError:
        pass
    else:
        raise AssertionError("CachePipeline must require an implementation")


def test_kivi_pipeline_calls_adapter_and_preserves_legacy_container(monkeypatch):
    import src.cache_pipeline as cache_pipeline

    seen = []

    def fake_roundtrip(cache, config):
        seen.append((cache, config))
        return cache

    monkeypatch.setattr(cache_pipeline, "roundtrip_cache", fake_roundtrip)
    source = _cache()
    config = KIVIConfig(2, 2, 32, 32)

    result = KIVIPipeline(config).roundtrip(source)

    assert result is source
    assert seen == [(source, config)]


class _FakeKVCloak:
    def __init__(self):
        self.calls = []

    def obfuscate(self, cache):
        self.calls.append("obfuscate")
        return cache

    def deobfuscate(self, cache):
        self.calls.append("deobfuscate")
        return cache


def test_kvcloak_kivi_pipeline_orders_operations(monkeypatch):
    import src.cache_pipeline as cache_pipeline

    events = []

    def fake_quantize(cache, config):
        events.append("quantize")
        return {"format": "fake", "layers": cache}

    def fake_dequantize(native, config):
        events.append("dequantize")
        return native["layers"]

    monkeypatch.setattr(cache_pipeline, "quantize_cache", fake_quantize)
    monkeypatch.setattr(cache_pipeline, "dequantize_cache", fake_dequantize)
    cloak = _FakeKVCloak()
    source = _cache()

    # A fake cloak returns its input, so a tuple is enough to verify ordering.
    result = KVCloakKIVIPipeline(cloak, KIVIConfig(2, 2, 32, 32)).roundtrip(source)

    for (result_key, result_value), (source_key, source_value) in zip(result, source):
        assert torch.equal(result_key, source_key)
        assert torch.equal(result_value, source_value)
    assert cloak.calls == ["obfuscate", "deobfuscate"]
    assert events == ["quantize", "dequantize"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="KIVI public kernels require CUDA")
def test_real_kvcloak_kivi_pipeline_restores_original_sequence_shape():
    from defense.core.kvcloak import KVCloak, create_test_kv_config

    torch.manual_seed(42)
    source = (
        (
            torch.randn(1, 1, 130, 64, device="cuda", dtype=torch.float16),
            torch.randn(1, 1, 130, 64, device="cuda", dtype=torch.float16),
        ),
    )
    cloak_config = create_test_kv_config(
        1, 1, [130, 130], 1.0, 1.0, 16, 64, "cuda", torch.float16
    )
    cloak = KVCloak(cloak_config, torch.float16, True, False, False)

    result = KVCloakKIVIPipeline(
        cloak, KIVIConfig(2, 2, 32, 32)
    ).roundtrip(source)

    assert result[0][0].shape == source[0][0].shape
    assert result[0][1].shape == source[0][1].shape
    assert torch.isfinite(result[0][0]).all()
    assert torch.isfinite(result[0][1]).all()

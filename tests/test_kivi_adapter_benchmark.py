"""Tests for KIVI adapter latency reporting."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from defense.eval.kivi_adapter_benchmark import benchmark_cache  # noqa: E402
from src.kivi_adapter import KIVIConfig  # noqa: E402


def test_benchmark_reports_quant_dequant_and_roundtrip(monkeypatch):
    import defense.eval.kivi_adapter_benchmark as benchmark_module

    native = {"format": "fake", "layers": ()}
    cache = ((torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)),)
    monkeypatch.setattr(benchmark_module, "quantize_cache", lambda cache, config: native)
    monkeypatch.setattr(
        benchmark_module, "dequantize_cache", lambda native, config: cache
    )

    result = benchmark_cache(
        cache,
        KIVIConfig(2, 2, 32, 32),
        device="cpu",
        warmup=1,
        trials=3,
    )

    assert result["trials"] == 3
    assert result["device"] == "cpu"
    assert set(result["timings_ms"]) == {"quantize", "dequantize", "roundtrip"}
    for timing in result["timings_ms"].values():
        assert timing["mean"] >= 0
        assert timing["median"] >= 0

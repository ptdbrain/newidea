"""Tests for the AES-over-quantized-KV benchmark helpers (CPU only)."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from defense.baseline.aes_kvcache import KVCacheAESProtecter  # noqa: E402
from defense.eval.aes_quant_benchmark import (  # noqa: E402
    benchmark_aes,
    linear_fit,
    payload_nbytes,
    size_equivalent_payload,
    summarize,
)


def _native(seq_len=64, head_dim=64, heads=2, bits=4):
    """INT-``bits`` payload with KIVI's full-prompt-quantized shapes."""
    groups = seq_len // 32
    values_per_int = 32 // bits
    return {
        "format": "kivi-native-v1",
        "layers": (
            {
                "key_code": torch.zeros(1, heads, head_dim, seq_len // values_per_int, dtype=torch.int32),
                "key_scale": torch.ones(1, heads, head_dim, groups, dtype=torch.float16),
                "key_min": torch.zeros(1, heads, head_dim, groups, dtype=torch.float16),
                "key_residual": torch.empty(1, heads, 0, head_dim, dtype=torch.float16),
                "value_code": torch.zeros(1, heads, seq_len, head_dim // values_per_int, dtype=torch.int32),
                "value_scale": torch.ones(1, heads, seq_len, head_dim // 32, dtype=torch.float16),
                "value_min": torch.zeros(1, heads, seq_len, head_dim // 32, dtype=torch.float16),
                "value_residual": torch.empty(1, heads, 0, head_dim, dtype=torch.float16),
                "seq_len": seq_len,
            },
        ),
    }


def test_payload_nbytes_counts_fp_and_native_tensors():
    fp = ((torch.zeros(1, 2, 64, 64, dtype=torch.float16),) * 2,)
    assert payload_nbytes(fp) == 2 * 2 * 64 * 64 * 2

    native = _native()
    values = 2 * 64 * 64
    metadata = 4 * values // 32 * 2  # K/V scale+min in fp16, one per 32 values
    assert payload_nbytes(native) == values // 2 * 2 + metadata


def test_size_equivalent_payload_matches_real_bit_width_sizes():
    int4 = _native(bits=4)

    assert payload_nbytes(size_equivalent_payload(int4, 4, 2)) == payload_nbytes(_native(bits=2))
    int3 = size_equivalent_payload(int4, 4, 3)
    layer = int3["layers"][0]
    assert layer["key_code"].numel() == 2 * 64 * 64 * 3 // 8
    assert layer["key_code"].dtype == torch.uint8
    assert layer["key_scale"] is int4["layers"][0]["key_scale"]
    assert int3["size_equivalent_bits"] == 3


@pytest.mark.parametrize("payload", [_native(), ((torch.randn(1, 2, 64, 64).half(),) * 2,)])
def test_benchmark_aes_reports_split_timings(payload):
    protector = KVCacheAESProtecter(b"0123456789abcdef", device="cpu")

    result = benchmark_aes(protector, payload, device=torch.device("cpu"), warmup=0, trials=2)

    assert result["payload_bytes"] == payload_nbytes(payload)
    assert result["ciphertext_bytes"] == result["payload_bytes"] + 28 * result["num_aes_calls"]
    assert result["total_ms_mean"] == pytest.approx(
        result["encrypt_ms"]["mean"] + result["decrypt_ms"]["mean"]
    )
    assert result["roundtrip_GBps"] > 0


def test_linear_fit_recovers_line():
    fit = linear_fit([(1e6, 3.0), (2e6, 5.0), (4e6, 9.0)])

    assert fit["slope_ms_per_MB"] == pytest.approx(2.0)
    assert fit["intercept_ms"] == pytest.approx(1.0)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["implied_GBps"] == pytest.approx(0.5)
    assert linear_fit([(1.0, 1.0)]) is None


def test_summarize_reports_ratios_against_fp16():
    def record(condition, payload_bytes, total_ms):
        return {
            "batch_size": 1,
            "seq_len": 512,
            "condition": condition,
            "payload_bytes": payload_bytes,
            "encrypt_ms": {"mean": total_ms / 2},
            "decrypt_ms": {"mean": total_ms / 2},
            "total_ms_mean": total_ms,
        }

    summary = summarize([record("FP16", 1000, 10.0), record("INT4", 250, 3.0)])

    int4 = summary["rows"][1]
    assert int4["bytes_vs_fp16"] == pytest.approx(0.25)
    assert int4["time_vs_fp16"] == pytest.approx(0.3)

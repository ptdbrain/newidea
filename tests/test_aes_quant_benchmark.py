"""Tests for the AES-over-quantized-KV benchmark helpers (CPU only)."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from defense.baseline.aes_kvcache import KVCacheAESProtecter  # noqa: E402
from defense.eval.aes_quant_benchmark import (  # noqa: E402
    benchmark_aes,
    expected_payload_nbytes,
    fp_payload,
    linear_fit,
    measure_interleaved,
    payload_nbytes,
    summarize,
)


def _native(seq_len=64, head_dim=64, heads=2, bits=4):
    """INT-``bits`` payload with KIVI's full-prompt-quantized shapes."""
    groups = seq_len // 32
    if bits == 3:  # dense uint8 packing from src/int3_quant.py
        code_dtype, key_code_len, value_code_len = torch.uint8, seq_len * 3 // 8, head_dim * 3 // 8
    else:
        code_dtype, key_code_len, value_code_len = torch.int32, seq_len * bits // 32, head_dim * bits // 32
    return {
        "format": "kivi-native-v1",
        "layers": (
            {
                "key_code": torch.zeros(1, heads, head_dim, key_code_len, dtype=code_dtype),
                "key_scale": torch.ones(1, heads, head_dim, groups, dtype=torch.float16),
                "key_min": torch.zeros(1, heads, head_dim, groups, dtype=torch.float16),
                "key_residual": torch.empty(1, heads, 0, head_dim, dtype=torch.float16),
                "value_code": torch.zeros(1, heads, seq_len, value_code_len, dtype=code_dtype),
                "value_scale": torch.ones(1, heads, seq_len, head_dim // 32, dtype=torch.float16),
                "value_min": torch.zeros(1, heads, seq_len, head_dim // 32, dtype=torch.float16),
                "value_residual": torch.empty(1, heads, 0, head_dim, dtype=torch.float16),
                "seq_len": seq_len,
            },
        ),
    }


def test_payload_nbytes_counts_fp_and_native_tensors():
    fp = fp_payload(((torch.zeros(1, 2, 64, 64, dtype=torch.float16),) * 2,))
    assert payload_nbytes(fp) == 2 * 2 * 64 * 64 * 2

    native = _native()
    values = 2 * 64 * 64
    metadata = 4 * values // 32 * 2  # K/V scale+min in fp16, one per 32 values
    assert payload_nbytes(native) == values // 2 * 2 + metadata


@pytest.mark.parametrize("bits", [4, 3, 2])
def test_expected_payload_nbytes_matches_packed_layout(bits):
    fp = ((torch.zeros(1, 2, 64, 64, dtype=torch.float16),) * 2,)

    assert expected_payload_nbytes(fp, bits, 32) == payload_nbytes(_native(bits=bits))
    assert expected_payload_nbytes(fp, bits, 32) / payload_nbytes(fp_payload(fp)) == pytest.approx(
        (bits / 8 + 4 / 32) / 2
    )


def test_expected_payload_nbytes_counts_key_padding():
    fp = ((torch.zeros(1, 1, 40, 64, dtype=torch.float16),) * 2,)  # keys padded to 64 tokens

    per_value = 4 / 8 + 4 / 32
    assert expected_payload_nbytes(fp, 4, 32) == (64 + 40) * 64 * per_value


def test_benchmark_aes_equal_calls_and_split_timings():
    protector = KVCacheAESProtecter(b"0123456789abcdef", device="cpu")
    int4 = _native(bits=4)
    payloads = {
        "FP16": fp_payload(((torch.randn(1, 2, 64, 64).half(), torch.randn(1, 2, 64, 64).half()),)),
        "INT4": int4,
        "INT3": _native(bits=3),
    }

    results = benchmark_aes(protector, payloads, device=torch.device("cpu"), warmup=0, trials=3)

    assert {result["num_aes_calls"] for result in results.values()} == {1}
    for name, result in results.items():
        assert result["payload_bytes"] == payload_nbytes(payloads[name])
        assert result["ciphertext_bytes"] == result["payload_bytes"] + 16
        assert result["total_ms"] == pytest.approx(
            result["encrypt_ms"]["median"] + result["decrypt_ms"]["median"]
        )
        assert result["aes_only_ms"] == pytest.approx(
            result["encrypt_aes_ms"]["median"] + result["decrypt_aes_ms"]["median"]
        )
        assert result["encrypt_ms"]["n"] == 3
        assert result["roundtrip_GBps"] > 0


def test_benchmark_aes_rejects_unequal_call_counts():
    protector = KVCacheAESProtecter(b"0123456789abcdef", device="cpu")
    one_layer = _native()
    two_layers = {**one_layer, "layers": one_layer["layers"] * 2}

    with pytest.raises(RuntimeError, match="call counts"):
        benchmark_aes(protector, {"a": one_layer, "b": two_layers}, device=torch.device("cpu"), warmup=0, trials=1)


def test_measure_interleaved_runs_every_task_and_collects_phases():
    calls = []

    def task(name):
        def run(profile):
            calls.append(name)
            profile["phase"] = 0.001
        return run

    timings = measure_interleaved(
        {"a": task("a"), "b": task("b")}, device=torch.device("cpu"), warmup=1, trials=4, seed=0
    )

    assert calls.count("a") == calls.count("b") == 5
    assert timings["a"]["total"]["n"] == 4
    assert timings["b"]["phase"]["median"] == pytest.approx(1.0)


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
            "encrypt_ms": {"median": total_ms / 2},
            "decrypt_ms": {"median": total_ms / 2},
            "total_ms": total_ms,
            "aes_only_ms": total_ms / 2,
        }

    summary = summarize([record("FP16", 1000, 10.0), record("INT4", 250, 3.0)])

    int4 = summary["rows"][1]
    assert int4["bytes_vs_fp16"] == pytest.approx(0.25)
    assert int4["time_vs_fp16"] == pytest.approx(0.3)

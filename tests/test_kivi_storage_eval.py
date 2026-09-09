"""Tests for native KIVI storage accounting."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from defense.eval.kivi_storage_eval import (  # noqa: E402
    compression_ratio,
    measure_cache_pair,
    native_storage_bytes,
)


def test_native_storage_counts_only_tensor_payloads():
    native = {
        "format": "kivi-native-v1",
        "layers": ({
            "code": torch.zeros(8, dtype=torch.int32),
            "scale": torch.zeros(4, dtype=torch.float16),
            "metadata": "ignored",
        },),
    }

    assert native_storage_bytes(native) == 8 * 4 + 4 * 2


def test_storage_pair_reports_compression_ratio():
    origin = ((torch.zeros(100, dtype=torch.float16), torch.zeros(100, dtype=torch.float16)),)
    native = {"layers": ({"code": torch.zeros(25, dtype=torch.int32)},)}

    report = measure_cache_pair(origin, native)

    assert report["origin_bytes"] == 400
    assert report["native_bytes"] == 100
    assert report["compression_ratio"] == compression_ratio(400, 100)
    assert report["compression_ratio"] == 4.0

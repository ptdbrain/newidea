"""Tests for batch KIVI cache materialization."""

import json
from pathlib import Path
import sys

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.kivi_adapter import KIVIConfig  # noqa: E402
from defense.baseline import kivi_kvcache  # noqa: E402


def test_batch_processor_writes_native_attack_view_and_provenance(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache" / "float16" / "dataset" / "model"
    sample_dir = cache_root / "sample-a" / "origin"
    sample_dir.mkdir(parents=True)
    source = ((torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)),)
    torch.save(source, sample_dir / "past_key_values.pt")

    native = {
        "format": "kivi-native-v1",
        "layers": ({"tensor": torch.tensor([1], dtype=torch.int32)},),
    }
    attack_view = ((torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)),)
    monkeypatch.setattr(kivi_kvcache, "quantize_cache", lambda cache, config: native)
    monkeypatch.setattr(
        kivi_kvcache, "dequantize_cache", lambda cache, config: attack_view
    )

    output_type = kivi_kvcache.build_protect_type(KIVIConfig(2, 2, 32, 32), "origin")
    written = kivi_kvcache.process_cache_directory(
        cache_root,
        KIVIConfig(2, 2, 32, 32),
        source_protect_type="origin",
        output_protect_type=output_type,
        device="cpu",
        dtype=torch.float16,
    )

    assert written == [cache_root / "sample-a" / output_type]
    output_dir = written[0]
    assert torch.load(output_dir / "native.pt", weights_only=True)["format"] == "kivi-native-v1"
    restored = torch.load(output_dir / "past_key_values.pt", weights_only=True)
    for (restored_key, restored_value), (expected_key, expected_value) in zip(
        restored, attack_view
    ):
        assert torch.equal(restored_key, expected_key)
        assert torch.equal(restored_value, expected_value)
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["method"] == "kivi"
    assert metadata["source"] == "origin"
    assert metadata["attack_view"] == "dequantized_from_native"
    assert metadata["kivi_commit"]
    assert metadata["kvcloak_commit"]


def test_batch_processor_marks_full_prompt_quantized_condition(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache" / "float16" / "dataset" / "model"
    sample_dir = cache_root / "sample-a" / "origin"
    sample_dir.mkdir(parents=True)
    source = ((torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)),)
    torch.save(source, sample_dir / "past_key_values.pt")

    native = {
        "format": "kivi-native-v1",
        "mode": "full_prompt_quantized",
        "fp_residual_prompt_tokens": 0,
        "layers": (),
    }
    monkeypatch.setattr(kivi_kvcache, "quantize_cache", lambda cache, config: native)
    monkeypatch.setattr(
        kivi_kvcache,
        "dequantize_cache",
        lambda cache, config: source,
    )
    config = KIVIConfig(
        2, 2, 32, 32, mode="full_prompt_quantized"
    )

    output_type = kivi_kvcache.build_protect_type(config, "origin")
    assert output_type == "kivi_k2_v2_g32_fq"
    written = kivi_kvcache.process_cache_directory(
        cache_root,
        config,
        source_protect_type="origin",
        output_protect_type=output_type,
        device="cpu",
        dtype=torch.float16,
    )

    metadata = json.loads((written[0] / "metadata.json").read_text())
    assert metadata["mode"] == "full_prompt_quantized"
    assert metadata["fp_residual_prompt_tokens"] == 0


def test_batch_processor_can_limit_smoke_slice(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache" / "float16" / "dataset" / "model"
    for sample_name in ("sample-a", "sample-b"):
        sample_dir = cache_root / sample_name / "origin"
        sample_dir.mkdir(parents=True)
        torch.save(
            ((torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)),),
            sample_dir / "past_key_values.pt",
        )
    native = {"format": "kivi-native-v1", "layers": ()}
    monkeypatch.setattr(kivi_kvcache, "quantize_cache", lambda cache, config: native)
    monkeypatch.setattr(
        kivi_kvcache,
        "dequantize_cache",
        lambda cache, config: ((torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)),),
    )

    written = kivi_kvcache.process_cache_directory(
        cache_root,
        KIVIConfig(2, 2, 32, 32),
        device="cpu",
        dtype=torch.float16,
        start_index=1,
        end_index=2,
    )

    assert written == [cache_root / "sample-b" / "kivi_k2_v2_g32_r32"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="KIVI public kernels require CUDA")
def test_batch_processor_roundtrips_real_kivi_payload(tmp_path):
    cache_root = tmp_path / "cache" / "float16" / "dataset" / "model"
    sample_dir = cache_root / "sample-a" / "origin"
    sample_dir.mkdir(parents=True)
    source = (
        (
            torch.randn(1, 1, 130, 64, dtype=torch.float16),
            torch.randn(1, 1, 130, 64, dtype=torch.float16),
        ),
    )
    torch.save(source, sample_dir / "past_key_values.pt")

    config = KIVIConfig(2, 2, 32, 32)
    output_type = kivi_kvcache.build_protect_type(config, "origin")
    written = kivi_kvcache.process_cache_directory(
        cache_root,
        config,
        source_protect_type="origin",
        output_protect_type=output_type,
        device="cuda",
        dtype=torch.float16,
    )

    native = torch.load(written[0] / "native.pt", weights_only=True)
    restored = torch.load(written[0] / "past_key_values.pt", weights_only=True)
    assert native["layers"][0]["key_code"].device.type == "cpu"
    assert restored[0][0].shape == source[0][0].shape
    assert restored[0][1].shape == source[0][1].shape
    assert torch.isfinite(restored[0][0]).all()
    assert torch.isfinite(restored[0][1]).all()

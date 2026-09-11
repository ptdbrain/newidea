"""Dependency-light tests for strict KIVI source-cache validation."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types

import pytest


def _load_kivi_batch_module_without_kernels(monkeypatch: pytest.MonkeyPatch):
    adapter = types.ModuleType("src.kivi_adapter")
    adapter.KIVIConfig = object
    adapter.cache_to_device = lambda cache, device, dtype=None: cache
    adapter.quantize_cache = lambda cache, config: cache
    adapter.dequantize_cache = lambda cache, config: cache
    monkeypatch.setitem(sys.modules, "src.kivi_adapter", adapter)
    sys.modules.pop("defense.baseline.kivi_kvcache", None)
    return importlib.import_module("defense.baseline.kivi_kvcache")


def test_batch_processor_strict_mode_rejects_missing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kivi_kvcache = _load_kivi_batch_module_without_kernels(monkeypatch)
    try:
        cache_root = tmp_path / "cache"
        (cache_root / "sample-a").mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match=r"sample-a.*past_key_values\.pt"):
            kivi_kvcache.process_cache_directory(
                cache_root,
                object(),
                output_protect_type="kivi-test",
                device="cpu",
                strict=True,
            )
    finally:
        sys.modules.pop("defense.baseline.kivi_kvcache", None)

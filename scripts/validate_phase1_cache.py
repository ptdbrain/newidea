"""Validate Phase 1 cache completeness and measure distortion/storage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from defense.eval.kivi_storage_eval import (
    compression_ratio,
    legacy_storage_bytes,
    native_storage_bytes,
)


def _load(path: Path) -> Any:
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.mean((left.float() - right.float()) ** 2).item())


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left.float()).item()
    if denominator == 0:
        return 0.0
    return float(torch.linalg.vector_norm((left - right).float()).item() / denominator)


def validate_cache(
    dataset_path: Path,
    cache_root: Path,
    conditions: dict[str, str],
) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = [json.loads(line) for line in handle if line.strip()]

    report: list[dict[str, Any]] = []
    for sample in dataset:
        prompt = sample["prompt"]
        sample_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        sample_dir = cache_root / sample_hash
        origin_path = sample_dir / "origin" / "past_key_values.pt"
        if not origin_path.is_file():
            raise FileNotFoundError(f"missing origin cache for {sample['sample_id']}: {origin_path}")
        origin = _load(origin_path)
        origin_bytes = legacy_storage_bytes(origin)
        for condition, directory_name in conditions.items():
            condition_dir = sample_dir / directory_name
            view_path = condition_dir / "past_key_values.pt"
            if not view_path.is_file():
                raise FileNotFoundError(f"missing attack view for {sample['sample_id']}: {view_path}")
            view = _load(view_path)
            if len(view) != len(origin):
                raise ValueError(f"layer count mismatch for {sample['sample_id']} / {condition}")
            k_mse = []
            v_mse = []
            k_rel = []
            v_rel = []
            finite = True
            for (origin_k, origin_v), (view_k, view_v) in zip(origin, view):
                if origin_k.shape != view_k.shape or origin_v.shape != view_v.shape:
                    raise ValueError(f"shape mismatch for {sample['sample_id']} / {condition}")
                finite = finite and bool(torch.isfinite(view_k).all() and torch.isfinite(view_v).all())
                k_mse.append(_mse(origin_k, view_k))
                v_mse.append(_mse(origin_v, view_v))
                k_rel.append(_relative_l2(origin_k, view_k))
                v_rel.append(_relative_l2(origin_v, view_v))

            metadata_path = condition_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            native_path = condition_dir / "native.pt"
            native_bytes = native_storage_bytes(_load(native_path)) if native_path.is_file() else origin_bytes
            report.append(
                {
                    "sample_id": sample["sample_id"],
                    "source_row_id": sample.get("source_row_id"),
                    "input_hash": sample_hash,
                    "input_token_length": sample.get("prompt_token_length"),
                    "length_stratum": sample.get("length_stratum"),
                    "condition": condition,
                    "mode": metadata.get("mode", "identity" if condition == "FP16" else None),
                    "fp_residual_prompt_tokens": metadata.get("fp_residual_prompt_tokens"),
                    "layer_count": len(view),
                    "finite": finite,
                    "k_mse_mean": sum(k_mse) / len(k_mse),
                    "v_mse_mean": sum(v_mse) / len(v_mse),
                    "k_relative_l2_mean": sum(k_rel) / len(k_rel),
                    "v_relative_l2_mean": sum(v_rel) / len(v_rel),
                    "origin_bytes": origin_bytes,
                    "native_bytes": native_bytes,
                    "compression_ratio": compression_ratio(origin_bytes, native_bytes),
                    "pass": finite
                    and (condition == "FP16" or metadata.get("mode") is not None),
                }
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conditions = {
        "FP16": "origin",
        "KIVI4-STD": "kivi_k4_v4_g32_r32",
        "KIVI4-FQ": "kivi_k4_v4_g32_fq",
        "KIVI2-STD": "kivi_k2_v2_g32_r32",
        "KIVI2-FQ": "kivi_k2_v2_g32_fq",
    }
    report = validate_cache(args.dataset, args.cache_root, conditions)
    if not all(item["pass"] for item in report):
        raise RuntimeError("cache validation failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(report)} sample-condition cache pairs")


if __name__ == "__main__":
    main()

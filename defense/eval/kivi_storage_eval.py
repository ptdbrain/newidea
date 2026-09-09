"""Measure raw tensor storage for native KIVI representations."""

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.provenance import KIVI_COMMIT, KVCLOAK_COMMIT


def _iter_tensors(value: Any, seen: set[int] | None = None) -> Iterator[torch.Tensor]:
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity not in seen:
            seen.add(identity)
            yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item, seen)


def tensor_storage_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def native_storage_bytes(native_cache: Any) -> int:
    """Count bytes of packed codes, parameters, and residual tensors."""
    return sum(tensor_storage_bytes(tensor) for tensor in _iter_tensors(native_cache))


def legacy_storage_bytes(past_key_values: Any) -> int:
    return sum(tensor_storage_bytes(tensor) for tensor in _iter_tensors(past_key_values))


def compression_ratio(origin_bytes: int, compressed_bytes: int) -> float:
    if origin_bytes < 0 or compressed_bytes <= 0:
        raise ValueError("storage byte counts must be non-negative and compressed_bytes > 0")
    return origin_bytes / compressed_bytes


def measure_cache_pair(origin_cache: Any, native_cache: Any) -> dict[str, float | int]:
    origin_bytes = legacy_storage_bytes(origin_cache)
    native_bytes = native_storage_bytes(native_cache)
    return {
        "kivi_commit": KIVI_COMMIT,
        "kvcloak_commit": KVCLOAK_COMMIT,
        "origin_bytes": origin_bytes,
        "native_bytes": native_bytes,
        "compression_ratio": compression_ratio(origin_bytes, native_bytes),
    }


def _load(path: Path) -> Any:
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def aggregate_cache_directory(
    cache_root: Path, protect_type: str
) -> dict[str, Any]:
    cache_root = Path(cache_root).expanduser()
    records = []
    for sample_dir in sorted(path for path in cache_root.iterdir() if path.is_dir()):
        origin_path = sample_dir / "origin" / "past_key_values.pt"
        native_path = sample_dir / protect_type / "native.pt"
        if not origin_path.is_file() or not native_path.is_file():
            continue
        record = {"sample": sample_dir.name, **measure_cache_pair(_load(origin_path), _load(native_path))}
        records.append(record)

    origin_total = sum(record["origin_bytes"] for record in records)
    native_total = sum(record["native_bytes"] for record in records)
    return {
        "kivi_commit": KIVI_COMMIT,
        "kvcloak_commit": KVCLOAK_COMMIT,
        "protect_type": protect_type,
        "sample_count": len(records),
        "origin_bytes": origin_total,
        "native_bytes": native_total,
        "compression_ratio": compression_ratio(origin_total, native_total)
        if native_total
        else 0.0,
        "samples": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-path", type=Path)
    parser.add_argument("--native-path", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--protect-type")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.origin_path and args.native_path:
        report = measure_cache_pair(_load(args.origin_path), _load(args.native_path))
    elif args.cache_root and args.protect_type:
        report = aggregate_cache_directory(args.cache_root, args.protect_type)
    else:
        parser.error("provide --origin-path/--native-path or --cache-root/--protect-type")

    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

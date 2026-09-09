"""Benchmark KIVI adapter quantize/dequantize/roundtrip latency."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kivi_adapter import (  # noqa: E402
    KIVIConfig,
    cache_to_device,
    dequantize_cache,
    quantize_cache,
)
from src.provenance import KIVI_COMMIT, KVCLOAK_COMMIT  # noqa: E402


def _timing_stats(samples: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def _measure(
    function: Callable[[], Any], *, device: torch.device, warmup: int, trials: int
) -> dict[str, float]:
    if warmup < 0 or trials <= 0:
        raise ValueError("warmup must be >= 0 and trials must be > 0")

    for _ in range(warmup):
        function()

    is_cuda = device.type == "cuda"
    if is_cuda:
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)

    samples: list[float] = []
    if is_cuda:
        with torch.cuda.device(device):
            for _ in range(trials):
                torch.cuda.synchronize(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                function()
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
    else:
        for _ in range(trials):
            start_time = time.perf_counter()
            function()
            samples.append((time.perf_counter() - start_time) * 1000.0)
    return _timing_stats(samples)


def benchmark_cache(
    past_key_values: Any,
    config: KIVIConfig,
    *,
    device: torch.device | str,
    warmup: int = 2,
    trials: int = 10,
) -> dict[str, Any]:
    """Return mean/median timings without claiming end-to-end throughput."""
    benchmark_device = torch.device(device)
    native_cache = quantize_cache(past_key_values, config)

    timings = {
        "quantize": _measure(
            lambda: quantize_cache(past_key_values, config),
            device=benchmark_device,
            warmup=warmup,
            trials=trials,
        ),
        "dequantize": _measure(
            lambda: dequantize_cache(native_cache, config),
            device=benchmark_device,
            warmup=warmup,
            trials=trials,
        ),
        "roundtrip": _measure(
            lambda: dequantize_cache(quantize_cache(past_key_values, config), config),
            device=benchmark_device,
            warmup=warmup,
            trials=trials,
        ),
    }
    first_key, first_value = past_key_values[0]
    return {
        "config": asdict(config),
        "kivi_commit": KIVI_COMMIT,
        "kvcloak_commit": KVCLOAK_COMMIT,
        "device": str(benchmark_device),
        "shape": {"key": list(first_key.shape), "value": list(first_value.shape)},
        "warmup": warmup,
        "trials": trials,
        "timings_ms": timings,
        "note": "Offline adapter timing; not native end-to-end model throughput.",
    }


def _load_cache(path: Path) -> Any:
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=130)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--k-bits", type=int, default=2)
    parser.add_argument("--v-bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=["standard", "full_prompt_quantized"],
        default="standard",
        help="KIVI lifecycle to benchmark.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    if args.cache_path:
        cache = cache_to_device(_load_cache(args.cache_path), args.device, dtype=dtype)
    else:
        cache = (
            (
                torch.randn(args.batch_size, args.num_heads, args.seq_len, args.head_dim, device=args.device, dtype=dtype),
                torch.randn(args.batch_size, args.num_heads, args.seq_len, args.head_dim, device=args.device, dtype=dtype),
            ),
        )
    report = benchmark_cache(
        cache,
        KIVIConfig(
            args.k_bits,
            args.v_bits,
            args.group_size,
            args.residual_length,
            mode=args.mode,
        ),
        device=args.device,
        warmup=args.warmup,
        trials=args.trials,
    )
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

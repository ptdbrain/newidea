"""Benchmark the AES-GCM KV-cache baseline on FP16 and packed quantized caches.

AES-GCM runtime depends on the number of plaintext bytes, not on their
values. Every condition runs through the same
``KVCacheAESProtecter.encrypt_layers``/``decrypt_layers`` path, which makes
exactly one AES call per layer, and differs only in the payload:

* ``FP16``      - the canonical prefill cache, ``{"key", "value"}`` per layer.
* ``INT4``/``INT3``/``INT2`` - the packed ``kivi-native-v1`` payload from
  ``src.kivi_adapter.quantize_cache`` (full-prompt-quantized lifecycle,
  group 32). INT4/INT2 use the pinned KIVI kernels; KIVI has no 3-bit
  packing, so INT3 applies KIVI's formula at 3 bits with the dense packing in
  ``src/int3_quant.py`` (8 values in 3 bytes).

Measurement protocol:

* every payload is verified first (bit-exact roundtrip, ciphertext size,
  equal AES call count across conditions, and for quantized payloads
  ``dequantize(decrypt(encrypt(q))) == dequantize(q)``);
* encrypt and decrypt of all conditions are timed interleaved, in a shuffled
  order each round, with the garbage collector paused, so clock or thermal
  drift affects every condition equally;
* each call is split into its phases (device-to-host gather, AES,
  host-to-device scatter); medians are the reported estimate;
* the machine's raw AES-GCM throughput is measured as a calibration ceiling.

Prefill and decode are measured once per shape with the FP16 model and are
the shared denominator of every overhead ratio.
"""

import argparse
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defense.baseline.aes_kvcache import KVCacheAESProtecter  # noqa: E402
from src.provenance import KIVI_COMMIT, KVCLOAK_COMMIT  # noqa: E402


CONDITION_BITS = {"FP16": 16, "INT4": 4, "INT3": 3, "INT2": 2}
KIVI_GROUP_SIZE = 32
KIVI_RESIDUAL_LENGTH = 32
KIVI_MODE = "full_prompt_quantized"
GCM_TAG_BYTES = 16
PAYLOAD_NOTES = {
    "fp16_cache": "Canonical FP prefill cache.",
    "kivi_native": "Packed kivi-native-v1 payload from the pinned KIVI code.",
    "kivi_native_int3": (
        "Packed kivi-native-v1 payload; KIVI formula at 3 bits with dense "
        "3-bit packing from src/int3_quant.py (KIVI has no 3-bit packing)."
    ),
}


def fp_payload(fp_cache: tuple) -> dict:
    """Wrap a legacy ``((K, V), ...)`` cache in the layered payload format."""
    return {
        "format": "fp-legacy",
        "layers": tuple({"key": key, "value": value} for key, value in fp_cache),
    }


def payload_tensors(payload: dict) -> list[torch.Tensor]:
    """Return every tensor the AES baseline encrypts for ``payload``."""
    return [
        value
        for layer in payload["layers"]
        for value in layer.values()
        if torch.is_tensor(value)
    ]


def payload_nbytes(payload: dict) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in payload_tensors(payload))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stats(samples: list[float]) -> dict[str, float]:
    quartiles = statistics.quantiles(samples, n=4) if len(samples) > 1 else [samples[0]] * 3
    return {
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p25": quartiles[0],
        "p75": quartiles[2],
        "min": min(samples),
        "max": max(samples),
        "n": len(samples),
    }


def measure(
    function: Callable[[], Any], *, device: torch.device, warmup: int, trials: int
) -> dict[str, float]:
    """Wall-clock timing in ms with device synchronization around each call."""
    return measure_interleaved(
        {"task": lambda profile: function()}, device=device, warmup=warmup, trials=trials, seed=0
    )["task"]["total"]


def measure_interleaved(
    tasks: dict[str, Callable[[dict], Any]],
    *,
    device: torch.device,
    warmup: int,
    trials: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Time every task ``trials`` times in a freshly shuffled order per round.

    Each task receives a profile dict and may add per-phase seconds to it.
    Returns ``{task: {"total": stats_ms, phase: stats_ms, ...}}``.
    """
    if warmup < 0 or trials <= 0:
        raise ValueError("warmup must be >= 0 and trials must be > 0")
    names = list(tasks)
    for _ in range(warmup):
        for name in names:
            tasks[name]({})
    rng = random.Random(seed)
    totals: dict[str, list[float]] = {name: [] for name in names}
    phases: dict[str, dict[str, list[float]]] = {name: {} for name in names}
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for _ in range(trials):
            rng.shuffle(names)
            for name in names:
                profile: dict[str, float] = {}
                _synchronize(device)
                start = time.perf_counter()
                tasks[name](profile)
                _synchronize(device)
                totals[name].append((time.perf_counter() - start) * 1000.0)
                for phase, seconds in profile.items():
                    phases[name].setdefault(phase, []).append(seconds * 1000.0)
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        name: {"total": _stats(totals[name]), **{phase: _stats(values) for phase, values in phases[name].items()}}
        for name in tasks
    }


def _gbps(num_bytes: int, milliseconds: float) -> float:
    return num_bytes / (milliseconds / 1000.0) / 1e9 if milliseconds > 0 else float("inf")


def verify_aes(protector: KVCacheAESProtecter, payload: dict) -> dict[str, int]:
    """Check a bit-exact roundtrip and the ciphertext size; return byte counts."""
    tensors = payload_tensors(payload)
    encrypted = protector.encrypt_layers(payload)
    restored = protector.decrypt_layers(encrypted)
    for original_layer, restored_layer in zip(payload["layers"], restored["layers"]):
        for name, value in original_layer.items():
            if torch.is_tensor(value):
                if value.dtype != restored_layer[name].dtype or not torch.equal(
                    value.cpu(), restored_layer[name].cpu()
                ):
                    raise RuntimeError(f"AES roundtrip changed field {name!r}")
            elif restored_layer[name] != value:
                raise RuntimeError(f"AES roundtrip changed metadata {name!r}")
    payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    num_calls = len(encrypted["layers"])
    ciphertext_bytes = sum(len(layer["ciphertext"]) for layer in encrypted["layers"])
    if ciphertext_bytes != payload_bytes + GCM_TAG_BYTES * num_calls:
        raise RuntimeError(
            f"ciphertext {ciphertext_bytes} B != payload {payload_bytes} B + tags"
        )
    return {
        "payload_bytes": payload_bytes,
        "ciphertext_bytes": ciphertext_bytes,
        "num_aes_calls": num_calls,
    }


def benchmark_aes(
    protector: KVCacheAESProtecter,
    payloads: dict[str, dict],
    *,
    device: torch.device,
    warmup: int,
    trials: int,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Verify every payload, then time encrypt/decrypt of all interleaved."""
    sizes = {name: verify_aes(protector, payload) for name, payload in payloads.items()}
    calls = {size["num_aes_calls"] for size in sizes.values()}
    if len(calls) != 1:
        raise RuntimeError(f"conditions use different AES call counts: {sizes}")

    encrypted = {name: protector.encrypt_layers(payload) for name, payload in payloads.items()}
    tasks: dict[str, Callable[[dict], Any]] = {}
    for name in payloads:
        tasks[f"encrypt:{name}"] = (
            lambda profile, payload=payloads[name]: protector.encrypt_layers(payload, profile)
        )
        tasks[f"decrypt:{name}"] = (
            lambda profile, ciphertext=encrypted[name]: protector.decrypt_layers(ciphertext, profile)
        )
    timings = measure_interleaved(tasks, device=device, warmup=warmup, trials=trials, seed=seed)

    results = {}
    for name in payloads:
        enc = timings[f"encrypt:{name}"]
        dec = timings[f"decrypt:{name}"]
        payload_bytes = sizes[name]["payload_bytes"]
        encrypt_ms = enc["total"]["median"]
        decrypt_ms = dec["total"]["median"]
        aes_encrypt_ms = enc["aes"]["median"]
        aes_decrypt_ms = dec["aes"]["median"]
        results[name] = {
            **sizes[name],
            "encrypt_ms": enc["total"],
            "encrypt_gather_ms": enc["gather"],
            "encrypt_aes_ms": enc["aes"],
            "decrypt_ms": dec["total"],
            "decrypt_aes_ms": dec["aes"],
            "decrypt_scatter_ms": dec["scatter"],
            "total_ms": encrypt_ms + decrypt_ms,
            "aes_only_ms": aes_encrypt_ms + aes_decrypt_ms,
            "encrypt_GBps": _gbps(payload_bytes, encrypt_ms),
            "decrypt_GBps": _gbps(payload_bytes, decrypt_ms),
            "roundtrip_GBps": _gbps(payload_bytes, encrypt_ms + decrypt_ms),
            "aes_encrypt_GBps": _gbps(payload_bytes, aes_encrypt_ms),
            "aes_decrypt_GBps": _gbps(payload_bytes, aes_decrypt_ms),
        }
    return results


def raw_aes_calibration(num_bytes: int, trials: int) -> dict[str, float]:
    """AES-GCM throughput of this host on one host buffer: the timing ceiling."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    aesgcm = AESGCM(os.urandom(16))
    nonce = os.urandom(12)
    buffer = np.frombuffer(os.urandom(num_bytes), dtype=np.uint8)
    ciphertext = aesgcm.encrypt(nonce, buffer, None)
    timings = measure_interleaved(
        {
            "encrypt": lambda profile: aesgcm.encrypt(nonce, buffer, None),
            "decrypt": lambda profile: aesgcm.decrypt(nonce, ciphertext, None),
        },
        device=torch.device("cpu"),
        warmup=1,
        trials=trials,
        seed=0,
    )
    return {
        "bytes": num_bytes,
        "encrypt_GBps": _gbps(num_bytes, timings["encrypt"]["total"]["median"]),
        "decrypt_GBps": _gbps(num_bytes, timings["decrypt"]["total"]["median"]),
    }


def environment_info(device: torch.device) -> dict[str, Any]:
    import cryptography
    from cryptography.hazmat.backends.openssl.backend import backend

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cryptography": cryptography.__version__,
        "openssl": backend.openssl_version_text(),
        "kivi_commit": KIVI_COMMIT,
        "kvcloak_commit": KVCLOAK_COMMIT,
    }


def linear_fit(points: list[tuple[float, float]]) -> dict[str, float] | None:
    """Least-squares fit of time (ms) against plaintext size (bytes)."""
    if len({x for x, _ in points}) < 2:
        return None
    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.sum((y - (slope * x + intercept)) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope_ms_per_MB": float(slope) * 1e6,
        "intercept_ms": float(intercept),
        "r2": 1.0 - residual / total if total > 0 else 1.0,
        "implied_GBps": 1e-6 / float(slope) if slope > 0 else float("inf"),
        "num_points": len(points),
    }


def _legacy_layers(cache: Any) -> tuple:
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    if hasattr(cache, "layers"):
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    return tuple(cache)


def _dynamic_cache(legacy: tuple):
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for layer_idx, (key, value) in enumerate(legacy):
        cache.update(key, value, layer_idx)
    return cache


def build_input_ids(
    tokenizer, text: str, batch_size: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    """Repeat ``text`` to exactly ``seq_len`` tokens, BOS included."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    body_len = seq_len - (1 if bos is not None else 0)
    body = (ids * math.ceil(body_len / len(ids)))[:body_len]
    row = ([bos] if bos is not None else []) + body
    return torch.tensor([row] * batch_size, dtype=torch.long, device=device)


def measure_model(
    model, input_ids: torch.Tensor, *, decode_steps: int, device: torch.device, warmup: int, trials: int
) -> dict[str, Any]:
    """Time prefill and greedy decode; return the timings and the prefill cache.

    Prefill keeps only the last-position logits, as a serving stack does.
    """
    outputs = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    legacy = _legacy_layers(outputs.past_key_values)
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    del outputs

    def decode():
        past = _dynamic_cache(legacy)
        token = next_token
        for _ in range(decode_steps):
            step = model(input_ids=token, past_key_values=past, use_cache=True)
            past = step.past_key_values
            token = step.logits[:, -1:, :].argmax(dim=-1)

    prefill_ms = measure(
        lambda: model(input_ids=input_ids, use_cache=True, logits_to_keep=1),
        device=device,
        warmup=warmup,
        trials=trials,
    )
    decode_ms = measure(decode, device=device, warmup=warmup, trials=trials)
    return {
        "cache": legacy,
        "prefill_ms": prefill_ms,
        "decode_ms_per_token": {
            key: value / decode_steps if key != "n" else value for key, value in decode_ms.items()
        },
    }


def build_payloads(
    fp_cache: tuple,
    conditions: list[str],
    protector: KVCacheAESProtecter,
    *,
    device: torch.device,
    warmup: int,
    trials: int,
) -> dict[str, tuple[dict, str, dict | None]]:
    """Return ``{condition: (payload, payload_kind, kivi_timings)}``."""
    payloads: dict[str, tuple[dict, str, dict | None]] = {
        "FP16": (fp_payload(fp_cache), "fp16_cache", None)
    }
    quantized = [name for name in conditions if name != "FP16"]
    if quantized:
        # KIVI's Triton kernels need CUDA; FP16-only runs work without them.
        from src.kivi_adapter import KIVIConfig, dequantize_cache, quantize_cache
    for name in quantized:
        bits = CONDITION_BITS[name]
        config = KIVIConfig(bits, bits, KIVI_GROUP_SIZE, KIVI_RESIDUAL_LENGTH, mode=KIVI_MODE)
        native = quantize_cache(fp_cache, config)
        expected_bytes = expected_payload_nbytes(fp_cache, bits, KIVI_GROUP_SIZE)
        if payload_nbytes(native) != expected_bytes:
            raise RuntimeError(
                f"{name}: packed payload {payload_nbytes(native)} B != expected {expected_bytes} B"
            )
        # AES must be transparent to the quantizer: the decrypted payload
        # dequantizes to exactly the same K/V as the payload never encrypted.
        via_aes = dequantize_cache(protector.decrypt_layers(protector.encrypt_layers(native)), config)
        direct = dequantize_cache(native, config)
        if not all(
            torch.equal(left, right)
            for left_layer, right_layer in zip(via_aes, direct)
            for left, right in zip(left_layer, right_layer)
        ):
            raise RuntimeError(f"{name}: dequantize(decrypt(encrypt(q))) != dequantize(q)")
        kivi_timings = {
            "kivi_quantize_ms": measure(
                lambda: quantize_cache(fp_cache, config), device=device, warmup=warmup, trials=trials
            ),
            "kivi_dequantize_ms": measure(
                lambda: dequantize_cache(native, config), device=device, warmup=warmup, trials=trials
            ),
        }
        kind = "kivi_native_int3" if bits == 3 else "kivi_native"
        payloads[name] = (native, kind, kivi_timings)
    return {name: payloads[name] for name in conditions}


def expected_payload_nbytes(fp_cache: tuple, bits: int, group_size: int) -> int:
    """Packed bytes of a full-prompt-quantized cache.

    Codes take ``bits / 8`` bytes per value; each group of ``group_size``
    values carries one scale and one minimum in the cache dtype. Keys are
    padded along the sequence to a multiple of ``group_size`` (which KIVI's
    32-value groups make a multiple of every pack unit); values are grouped
    along the head dimension and are not padded.
    """
    total = 0
    for key, value in fp_cache:
        batch, heads, length, dim = key.shape
        padded_key_values = batch * heads * math.ceil(length / group_size) * group_size * dim
        for num_values, element_size in (
            (padded_key_values, key.element_size()),
            (value.numel(), value.element_size()),
        ):
            total += num_values * bits // 8 + num_values // group_size * 2 * element_size
    return total


def run(args: argparse.Namespace, output_path: Path) -> list[dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.config import BITTER_LESSON_TEXT

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    model_path = (
        Path(args.model_path).expanduser()
        if args.model_path
        else PROJECT_ROOT / ".models" / args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation=args.attn_implementation
    ).to(device, dtype)
    model.eval()
    protector = KVCacheAESProtecter(key=os.urandom(args.aes_key_bytes), device=device)

    records = []
    with output_path.open("w", encoding="utf-8") as handle:
        for batch_size in args.batch_sizes:
            for seq_len in args.seq_lens:
                print(f"--- batch={batch_size} seq_len={seq_len} ---")
                input_ids = build_input_ids(tokenizer, BITTER_LESSON_TEXT, batch_size, seq_len, device)
                model_timing = measure_model(
                    model,
                    input_ids,
                    decode_steps=args.decode_steps,
                    device=device,
                    warmup=args.warmup,
                    trials=args.trials,
                )
                fp_cache = model_timing.pop("cache")
                fp_bytes = payload_nbytes(fp_payload(fp_cache))
                prefill_ms = model_timing["prefill_ms"]["median"]
                decode_token_ms = model_timing["decode_ms_per_token"]["median"]
                request_ms = prefill_ms + args.decode_steps * decode_token_ms
                payloads = build_payloads(
                    fp_cache,
                    args.conditions,
                    protector,
                    device=device,
                    warmup=args.warmup,
                    trials=args.trials,
                )
                aes_results = benchmark_aes(
                    protector,
                    {name: payload for name, (payload, _, _) in payloads.items()},
                    device=device,
                    warmup=args.warmup,
                    trials=args.trials,
                    seed=args.seed,
                )
                for condition, (_, kind, kivi_timings) in payloads.items():
                    aes = aes_results[condition]
                    total_ms = aes["total_ms"]
                    record = {
                        "model": args.model_name,
                        "dtype": args.dtype,
                        "device": str(device),
                        "attn_implementation": args.attn_implementation,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "condition": condition,
                        "bits": CONDITION_BITS[condition],
                        "payload_kind": kind,
                        "kivi_mode": None if kind == "fp16_cache" else KIVI_MODE,
                        "int3_extension": kind == "kivi_native_int3",
                        "group_size": None if kind == "fp16_cache" else KIVI_GROUP_SIZE,
                        **aes,
                        "compression_vs_fp16": fp_bytes / aes["payload_bytes"],
                        **model_timing,
                        "decode_steps": args.decode_steps,
                        "overhead_vs_prefill": total_ms / prefill_ms,
                        "overhead_vs_decode_token": total_ms / decode_token_ms,
                        "overhead_vs_request": total_ms / request_ms,
                        "kivi_quantize_ms": (kivi_timings or {}).get("kivi_quantize_ms"),
                        "kivi_dequantize_ms": (kivi_timings or {}).get("kivi_dequantize_ms"),
                        "aes_key_bits": args.aes_key_bytes * 8,
                        "timing_statistic": "median",
                        "trials": args.trials,
                        "warmup": args.warmup,
                        "note": PAYLOAD_NOTES[kind],
                        "timestamp": datetime.now().isoformat(),
                    }
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    records.append(record)
                    print(
                        f"  {condition:<9} {aes['payload_bytes'] / 2**20:9.2f} MiB  "
                        f"enc {aes['encrypt_ms']['median']:8.2f} ms  "
                        f"dec {aes['decrypt_ms']['median']:8.2f} ms  "
                        f"AES-only {aes['aes_only_ms']:8.2f} ms  "
                        f"{aes['roundtrip_GBps']:.2f} GB/s  vs prefill {record['overhead_vs_prefill']:.2%}"
                    )
                del payloads, fp_cache
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Add per-shape ratios against FP16 and linear fits of time vs bytes."""
    fp16 = {
        (record["batch_size"], record["seq_len"]): record
        for record in records
        if record["condition"] == "FP16"
    }
    rows = []
    for record in records:
        baseline = fp16.get((record["batch_size"], record["seq_len"]))
        rows.append(
            {
                **record,
                "bytes_vs_fp16": record["payload_bytes"] / baseline["payload_bytes"] if baseline else None,
                "time_vs_fp16": record["total_ms"] / baseline["total_ms"] if baseline else None,
                "aes_only_vs_fp16": record["aes_only_ms"] / baseline["aes_only_ms"] if baseline else None,
            }
        )

    def value(record, metric):
        entry = record[metric]
        return entry["median"] if isinstance(entry, dict) else entry

    fits = {
        metric: linear_fit([(record["payload_bytes"], value(record, metric)) for record in records])
        for metric in ("encrypt_ms", "decrypt_ms", "total_ms", "aes_only_ms")
    }
    return {"rows": rows, "linear_fits": fits}


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    headers = [
        "B", "T", "condition", "payload MiB", "bytes vs FP16", "enc ms", "dec ms", "total ms",
        "AES-only ms", "time vs FP16", "AES-only vs FP16", "roundtrip GB/s", "vs prefill",
        "vs decode tok", "vs request",
    ]
    lines = [
        "All times are medians over interleaved trials.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]

    def ratio(value):
        return "-" if value is None else f"{value:.3f}"

    for row in summary["rows"]:
        cells = [
            str(row["batch_size"]),
            str(row["seq_len"]),
            row["condition"],
            f"{row['payload_bytes'] / 2**20:.2f}",
            ratio(row["bytes_vs_fp16"]),
            f"{row['encrypt_ms']['median']:.2f}",
            f"{row['decrypt_ms']['median']:.2f}",
            f"{row['total_ms']:.2f}",
            f"{row['aes_only_ms']:.2f}",
            ratio(row["time_vs_fp16"]),
            ratio(row["aes_only_vs_fp16"]),
            f"{row['roundtrip_GBps']:.2f}",
            f"{row['overhead_vs_prefill']:.2%}",
            f"{row['overhead_vs_decode_token']:.2f}x",
            f"{row['overhead_vs_request']:.2%}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Linear fit of time against plaintext bytes (all points):", ""]
    lines += ["| metric | ms/MB | intercept ms | R² | implied GB/s | n |", "|---|---|---|---|---|---|"]
    for metric, fit in summary["linear_fits"].items():
        if fit is None:
            lines.append(f"| {metric} | - | - | - | - | - |")
            continue
        lines.append(
            f"| {metric} | {fit['slope_ms_per_MB']:.4f} | {fit['intercept_ms']:.3f} | "
            f"{fit['r2']:.4f} | {fit['implied_GBps']:.2f} | {fit['num_points']} |"
        )
    calibration = summary.get("raw_aes_calibration")
    if calibration:
        lines += [
            "",
            f"Raw AES-GCM on this host ({calibration['bytes'] / 2**20:.0f} MiB buffer): "
            f"encrypt {calibration['encrypt_GBps']:.2f} GB/s, "
            f"decrypt {calibration['decrypt_GBps']:.2f} GB/s.",
        ]
    lines += [
        "",
        "Notes:",
        "- Every condition makes exactly one AES-GCM call per layer through the same code path.",
        "- enc = GPU tensors -> ciphertext in host memory (gather/D2H + AES-GCM);",
        "  dec = ciphertext -> GPU tensors (AES-GCM + H2D/scatter). AES-only = the two AES phases.",
        "  GB = 1e9 bytes; GB/s use median times.",
        "- INT4/INT3/INT2 are packed kivi-native-v1 payloads (full-prompt-quantized, group 32).",
        "- INT3 uses KIVI's formula at 3 bits with dense packing from src/int3_quant.py;",
        "  KIVI itself has no 3-bit packing.",
        "- Prefill/decode are FP16 model timings shared by all conditions;",
        "  KIVI quantize/dequantize time is reported separately in the JSONL.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _int_list(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="Llama-3.2-1B")
    parser.add_argument("--model-path", default=None, help="Defaults to .models/<model-name>.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--batch-sizes", type=_int_list, default=[1])
    parser.add_argument("--seq-lens", type=_int_list, default=[128, 512, 1024, 2048, 4096])
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument(
        "--conditions",
        type=lambda raw: [item.strip() for item in raw.split(",") if item.strip()],
        default=list(CONDITION_BITS),
        help="Comma-separated subset of FP16,INT4,INT3,INT2 (FP16 is always run).",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--aes-key-bytes", type=int, choices=[16, 32], default=16)
    parser.add_argument("--calibration-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="defense/result/aes_quant_benchmark")
    args = parser.parse_args()

    unknown = set(args.conditions) - set(CONDITION_BITS)
    if unknown:
        parser.error(f"unknown conditions: {sorted(unknown)}")
    if args.decode_steps <= 0:
        parser.error("--decode-steps must be positive")
    args.conditions = ["FP16"] + [name for name in args.conditions if name != "FP16"]

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model_name}_{args.dtype}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = output_dir / f"{stem}.jsonl"

    environment = environment_info(torch.device(args.device))
    calibration = raw_aes_calibration(args.calibration_mib << 20, trials=args.trials)
    print(
        f"Raw AES-GCM: encrypt {calibration['encrypt_GBps']:.2f} GB/s, "
        f"decrypt {calibration['decrypt_GBps']:.2f} GB/s"
    )
    with torch.no_grad():
        records = run(args, output_path)
    summary = {
        **summarize(records),
        "raw_aes_calibration": calibration,
        "environment": environment,
        "args": {key: value for key, value in vars(args).items()},
    }
    (output_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, output_dir / f"{stem}_summary.md")
    print(f"Raw records: {output_path}")
    print(f"Summary: {output_dir / f'{stem}_summary.md'}")


if __name__ == "__main__":
    main()

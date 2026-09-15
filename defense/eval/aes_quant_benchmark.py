"""Benchmark the AES-GCM KV-cache baseline on FP16 and packed quantized caches.

AES-GCM runtime depends on the number of plaintext bytes, not on their
values. Every condition therefore runs through the same
``KVCacheAESProtecter`` code path and differs only in the payload:

* ``FP16``      - the canonical prefill cache ``((K, V), ...)``.
* ``INT4``/``INT2`` - the packed ``kivi-native-v1`` payload produced by the
  pinned public KIVI code (full-prompt-quantized lifecycle).
* ``INT3-size`` - KIVI has no 3-bit kernel. This payload keeps the real INT4
  scale/min/residual tensors and replaces every packed code tensor by random
  bytes of the dense 3-bit size. It is valid for byte-count-driven AES timing
  only; it is not a quantizer and must not be dequantized.

Prefill and decode are measured once per shape with the FP16 model and are
the shared denominator of every overhead ratio.
"""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
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


CONDITION_BITS = {"FP16": 16, "INT4": 4, "INT3-size": 3, "INT2": 2}
CODE_FIELDS = ("key_code", "value_code")
KIVI_GROUP_SIZE = 32
KIVI_RESIDUAL_LENGTH = 32
KIVI_MODE = "full_prompt_quantized"
PAYLOAD_NOTES = {
    "fp16_cache": "Canonical FP prefill cache.",
    "kivi_native": "Packed kivi-native-v1 payload from the pinned KIVI code.",
    "size_equivalent": (
        "Random bytes of the dense 3-bit code size plus real INT4 scale/min/"
        "residual tensors; valid for AES timing only, not a quantizer."
    ),
}


def is_native(payload: Any) -> bool:
    return isinstance(payload, dict) and "layers" in payload


def payload_tensors(payload: Any) -> list[torch.Tensor]:
    """Return every tensor the AES baseline encrypts for ``payload``."""
    if is_native(payload):
        return [
            value
            for layer in payload["layers"]
            for value in layer.values()
            if torch.is_tensor(value)
        ]
    return [tensor for layer in payload for tensor in layer]


def payload_nbytes(payload: Any) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in payload_tensors(payload))


def size_equivalent_payload(
    native_cache: dict, source_bits: int, target_bits: int, *, seed: int = 0
) -> dict:
    """Replace packed codes by random bytes sized for ``target_bits``.

    Scale, minimum and residual tensors are kept: their shapes depend on the
    group size and lifecycle, not on the bit-width. Codes are sized for dense
    packing, ``ceil(values * target_bits / 8)`` bytes.
    """
    generator = torch.Generator().manual_seed(seed)
    layers = []
    for layer in native_cache["layers"]:
        layer = dict(layer)
        for name in CODE_FIELDS:
            code = layer.get(name)
            if code is None:
                continue
            num_values = code.numel() * code.element_size() * 8 // source_bits
            num_bytes = math.ceil(num_values * target_bits / 8)
            layer[name] = torch.randint(
                0, 256, (num_bytes,), dtype=torch.uint8, generator=generator
            ).to(code.device)
        layers.append(layer)
    return {**native_cache, "layers": tuple(layers), "size_equivalent_bits": target_bits}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    function: Callable[[], Any], *, device: torch.device, warmup: int, trials: int
) -> dict[str, float]:
    """Wall-clock timing in ms; AES runs on the host, so CUDA events do not apply."""
    if warmup < 0 or trials <= 0:
        raise ValueError("warmup must be >= 0 and trials must be > 0")
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(trials):
        _synchronize(device)
        start = time.perf_counter()
        function()
        _synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean": statistics.fmean(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def _gbps(num_bytes: int, milliseconds: float) -> float:
    return num_bytes / (milliseconds / 1000.0) / 1e9 if milliseconds > 0 else float("inf")


def benchmark_aes(
    protector: KVCacheAESProtecter,
    payload: Any,
    *,
    device: torch.device,
    warmup: int,
    trials: int,
) -> dict[str, Any]:
    """Time encrypt and decrypt separately after checking a bit-exact roundtrip."""
    native = is_native(payload)
    encrypt = protector.encrypt_native if native else protector.encrypt
    decrypt = protector.decrypt_native if native else protector.decrypt
    tensors = payload_tensors(payload)

    encrypted = encrypt(payload)
    restored = decrypt(encrypted)
    if native:
        restored_tensors = payload_tensors(restored)
        blobs = [blob for layer in encrypted["layers"] for blob in layer["encrypted"].values()]
    else:
        restored_tensors = [t for pair in protector._iter_cache_layers(restored) for t in pair]
        blobs = [blob for layer in encrypted for blob in layer]
    if len(restored_tensors) != len(tensors) or not all(
        torch.equal(original.cpu(), roundtrip.cpu())
        for original, roundtrip in zip(tensors, restored_tensors)
    ):
        raise RuntimeError("AES roundtrip did not reproduce the payload")
    del restored

    timings = {
        "encrypt_ms": measure(lambda: encrypt(payload), device=device, warmup=warmup, trials=trials),
        "decrypt_ms": measure(lambda: decrypt(encrypted), device=device, warmup=warmup, trials=trials),
        "serialize_ms": measure(
            lambda: [protector._tensor_to_bytes(tensor) for tensor in tensors],
            device=device,
            warmup=warmup,
            trials=trials,
        ),
    }
    payload_bytes = payload_nbytes(payload)
    encrypt_ms = timings["encrypt_ms"]["mean"]
    decrypt_ms = timings["decrypt_ms"]["mean"]
    total_ms = encrypt_ms + decrypt_ms
    return {
        "payload_bytes": payload_bytes,
        "ciphertext_bytes": sum(len(nonce) + len(ciphertext) for nonce, ciphertext, *_ in blobs),
        "num_aes_calls": len(blobs),
        **timings,
        "total_ms_mean": total_ms,
        "encrypt_GBps": _gbps(payload_bytes, encrypt_ms),
        "decrypt_GBps": _gbps(payload_bytes, decrypt_ms),
        "roundtrip_GBps": _gbps(payload_bytes, total_ms),
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
        "decode_ms_per_token": {key: value / decode_steps for key, value in decode_ms.items()},
    }


def build_payloads(
    fp_cache: tuple, conditions: list[str], *, device: torch.device, warmup: int, trials: int, seed: int
) -> dict[str, tuple[Any, str, dict | None]]:
    """Return ``{condition: (payload, payload_kind, kivi_timings)}``."""
    payloads: dict[str, tuple[Any, str, dict | None]] = {"FP16": (fp_cache, "fp16_cache", None)}
    natives = {}
    required_bits = {CONDITION_BITS[name] for name in conditions if name in ("INT4", "INT2")}
    if "INT3-size" in conditions:
        required_bits.add(4)
    if required_bits:
        # KIVI's Triton kernels need CUDA; FP16-only runs work without them.
        from src.kivi_adapter import KIVIConfig, dequantize_cache, quantize_cache
    for bits in sorted(required_bits, reverse=True):
        config = KIVIConfig(bits, bits, KIVI_GROUP_SIZE, KIVI_RESIDUAL_LENGTH, mode=KIVI_MODE)
        native = quantize_cache(fp_cache, config)
        natives[bits] = native
        if f"INT{bits}" in conditions:
            kivi_timings = {
                "kivi_quantize_ms": measure(
                    lambda: quantize_cache(fp_cache, config), device=device, warmup=warmup, trials=trials
                ),
                "kivi_dequantize_ms": measure(
                    lambda: dequantize_cache(native, config), device=device, warmup=warmup, trials=trials
                ),
            }
            payloads[f"INT{bits}"] = (native, "kivi_native", kivi_timings)

    if 4 in natives and 2 in natives:
        # The INT3 estimate uses the same size model; check it against KIVI.
        predicted = payload_nbytes(size_equivalent_payload(natives[4], 4, 2, seed=seed))
        actual = payload_nbytes(natives[2])
        if predicted != actual:
            raise RuntimeError(f"size model mismatch: INT4->INT2 {predicted} != KIVI INT2 {actual}")
    if "INT3-size" in conditions:
        payloads["INT3-size"] = (
            size_equivalent_payload(natives[4], 4, 3, seed=seed),
            "size_equivalent",
            None,
        )
    return {name: payloads[name] for name in conditions}


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
                fp_bytes = payload_nbytes(fp_cache)
                prefill_ms = model_timing["prefill_ms"]["mean"]
                decode_token_ms = model_timing["decode_ms_per_token"]["mean"]
                request_ms = prefill_ms + args.decode_steps * decode_token_ms
                payloads = build_payloads(
                    fp_cache,
                    args.conditions,
                    device=device,
                    warmup=args.warmup,
                    trials=args.trials,
                    seed=args.seed,
                )
                for condition, (payload, kind, kivi_timings) in payloads.items():
                    aes = benchmark_aes(
                        protector, payload, device=device, warmup=args.warmup, trials=args.trials
                    )
                    total_ms = aes["total_ms_mean"]
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
                        "kivi_commit": KIVI_COMMIT,
                        "kvcloak_commit": KVCLOAK_COMMIT,
                        "note": PAYLOAD_NOTES[kind],
                        "timestamp": datetime.now().isoformat(),
                    }
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    records.append(record)
                    print(
                        f"  {condition:<9} {aes['payload_bytes'] / 2**20:9.2f} MiB  "
                        f"enc {aes['encrypt_ms']['mean']:8.2f} ms  dec {aes['decrypt_ms']['mean']:8.2f} ms  "
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
                "time_vs_fp16": record["total_ms_mean"] / baseline["total_ms_mean"] if baseline else None,
            }
        )
    fits = {
        metric: linear_fit(
            [
                (record["payload_bytes"], record[metric]["mean"] if isinstance(record[metric], dict) else record[metric])
                for record in records
            ]
        )
        for metric in ("encrypt_ms", "decrypt_ms", "total_ms_mean")
    }
    return {"rows": rows, "linear_fits": fits}


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    headers = [
        "B", "T", "condition", "payload MiB", "bytes vs FP16", "enc ms", "dec ms",
        "total ms", "time vs FP16", "roundtrip GB/s", "vs prefill", "vs decode tok", "vs request",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in summary["rows"]:
        cells = [
            str(row["batch_size"]),
            str(row["seq_len"]),
            row["condition"],
            f"{row['payload_bytes'] / 2**20:.2f}",
            "-" if row["bytes_vs_fp16"] is None else f"{row['bytes_vs_fp16']:.3f}",
            f"{row['encrypt_ms']['mean']:.2f}",
            f"{row['decrypt_ms']['mean']:.2f}",
            f"{row['total_ms_mean']:.2f}",
            "-" if row["time_vs_fp16"] is None else f"{row['time_vs_fp16']:.3f}",
            f"{row['roundtrip_GBps']:.2f}",
            f"{row['overhead_vs_prefill']:.2%}",
            f"{row['overhead_vs_decode_token']:.2f}x",
            f"{row['overhead_vs_request']:.2%}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Linear fit of AES time against plaintext bytes (all points):", ""]
    lines += ["| metric | ms/MB | intercept ms | R² | implied GB/s | n |", "|---|---|---|---|---|---|"]
    for metric, fit in summary["linear_fits"].items():
        if fit is None:
            lines.append(f"| {metric} | - | - | - | - | - |")
            continue
        lines.append(
            f"| {metric} | {fit['slope_ms_per_MB']:.4f} | {fit['intercept_ms']:.3f} | "
            f"{fit['r2']:.4f} | {fit['implied_GBps']:.2f} | {fit['num_points']} |"
        )
    lines += [
        "",
        "Notes:",
        "- enc = GPU tensors -> ciphertext in host memory (D2H + serialize + AES-GCM);",
        "  dec = ciphertext -> GPU tensors (AES-GCM + deserialize + H2D). GB = 1e9 bytes.",
        "- INT4/INT2 are packed kivi-native-v1 payloads (full-prompt-quantized, group 32).",
        "- INT3-size is size-equivalent only: KIVI has no 3-bit kernel.",
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
    parser.add_argument("--seq-lens", type=_int_list, default=[512, 1024, 2048, 4096])
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument(
        "--conditions",
        type=lambda raw: [item.strip() for item in raw.split(",") if item.strip()],
        default=list(CONDITION_BITS),
        help="Comma-separated subset of FP16,INT4,INT3-size,INT2 (FP16 is always run).",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--aes-key-bytes", type=int, choices=[16, 32], default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="defense/result/aes_quant_benchmark")
    args = parser.parse_args()

    unknown = set(args.conditions) - set(CONDITION_BITS)
    if unknown:
        parser.error(f"unknown conditions: {sorted(unknown)}")
    if args.dtype == "bfloat16":
        parser.error("the AES baseline serializes through NumPy, which has no bfloat16")
    if args.decode_steps <= 0:
        parser.error("--decode-steps must be positive")
    args.conditions = ["FP16"] + [name for name in args.conditions if name != "FP16"]

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model_name}_{args.dtype}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = output_dir / f"{stem}.jsonl"

    with torch.no_grad():
        records = run(args, output_path)
    summary = summarize(records)
    (output_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, output_dir / f"{stem}_summary.md")
    print(f"Raw records: {output_path}")
    print(f"Summary: {output_dir / f'{stem}_summary.md'}")


if __name__ == "__main__":
    main()

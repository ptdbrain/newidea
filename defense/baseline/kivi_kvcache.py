"""Materialize native KIVI caches and Shadow-compatible attack views."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kivi_adapter import KIVIConfig, cache_to_device, dequantize_cache, quantize_cache  # noqa: E402
from src.provenance import KIVI_COMMIT, KVCLOAK_COMMIT  # noqa: E402


def build_protect_type(config: KIVIConfig, source_protect_type: str = "origin") -> str:
    """Build the stable cache directory name for a KIVI condition."""
    if config.mode == "full_prompt_quantized":
        kivi_type = (
            f"kivi_k{config.k_bits}_v{config.v_bits}_g{config.group_size}_fq"
        )
    else:
        kivi_type = (
            f"kivi_k{config.k_bits}_v{config.v_bits}_g{config.group_size}"
            f"_r{config.residual_length}"
        )
    if source_protect_type == "origin":
        return kivi_type
    if source_protect_type == "kvcloak":
        return f"kvcloak_{kivi_type}"
    raise ValueError(
        "source_protect_type must be 'origin' or 'kvcloak', "
        f"got {source_protect_type!r}"
    )


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _load_legacy_cache(path: Path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        # Compatibility with older Torch versions that lack weights_only.
        return torch.load(path)


def _metadata(
    config: KIVIConfig,
    source_protect_type: str,
    native_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "method": "kivi",
        "k_bits": config.k_bits,
        "v_bits": config.v_bits,
        "group_size": config.group_size,
        "residual_length": config.residual_length,
        "mode": config.mode,
        "fp_residual_prompt_tokens": int(
            (native_cache or {}).get("fp_residual_prompt_tokens", 0)
        ),
        "source": source_protect_type,
        "kivi_commit": KIVI_COMMIT,
        "kvcloak_commit": KVCLOAK_COMMIT,
        "attack_view": "dequantized_from_native",
        "native_format": "kivi-native-v1",
    }


def process_cache_directory(
    cache_root: Path,
    config: KIVIConfig,
    *,
    source_protect_type: str = "origin",
    output_protect_type: str | None = None,
    device: torch.device | str = "cuda:0",
    dtype: torch.dtype | None = torch.float16,
    start_index: int = 0,
    end_index: int | None = None,
    strict: bool = False,
) -> list[Path]:
    """Process every cached sample below ``cache_root``.

    The input cache is never overwritten.  Both the packed native payload and
    its dequantized Shadow attack view are saved under a new condition name.
    """
    cache_root = Path(cache_root).expanduser()
    if not cache_root.is_dir():
        raise FileNotFoundError(f"cache root does not exist: {cache_root}")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index is not None and end_index < start_index:
        raise ValueError("end_index must be greater than or equal to start_index")
    output_protect_type = output_protect_type or build_protect_type(
        config, source_protect_type
    )

    written: list[Path] = []
    sample_dirs = sorted(path for path in cache_root.iterdir() if path.is_dir())
    sample_dirs = sample_dirs[start_index:end_index]
    for sample_dir in tqdm(sample_dirs, desc=f"Processing {output_protect_type}"):
        source_path = sample_dir / source_protect_type / "past_key_values.pt"
        if not source_path.is_file():
            if strict:
                raise FileNotFoundError(
                    f"missing source cache for {sample_dir.name}: {source_path}"
                )
            print(f"Warning: source KV cache not found at {source_path}")
            continue

        source_cache = _load_legacy_cache(source_path)
        source_cache = cache_to_device(source_cache, device, dtype=dtype)
        native_cache = quantize_cache(source_cache, config)
        attack_view = dequantize_cache(native_cache, config)

        output_dir = sample_dir / output_protect_type
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(_to_cpu(native_cache), output_dir / "native.pt")
        torch.save(_to_cpu(attack_view), output_dir / "past_key_values.pt")
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(_metadata(config, source_protect_type, native_cache), file, indent=2)
            file.write("\n")
        written.append(output_dir)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Llama-3.2-1B")
    parser.add_argument("--dataset-path", default="./dataset/lmsys-chat-1m_1k.jsonl")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--source-protect-type", choices=["origin", "kvcloak"], default="origin")
    parser.add_argument("--k-bits", type=int, default=2)
    parser.add_argument("--v-bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=["standard", "full_prompt_quantized"],
        default="standard",
        help="KIVI lifecycle: retain residual tokens or quantize the full prompt.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately when a selected sample has no source cache.",
    )
    args = parser.parse_args()

    config = KIVIConfig(
        args.k_bits,
        args.v_bits,
        args.group_size,
        args.residual_length,
        mode=args.mode,
    )
    cache_root = (
        Path(args.cache_root).expanduser()
        if args.cache_root
        else Path("cache") / args.dtype / Path(args.dataset_path).stem / args.model_name
    )
    output_type = build_protect_type(config, args.source_protect_type)
    print(f"Input cache: {cache_root}")
    print(f"Output condition: {output_type}")
    process_cache_directory(
        cache_root,
        config,
        source_protect_type=args.source_protect_type,
        output_protect_type=output_type,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        start_index=args.start_index,
        end_index=args.end_index,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()

"""
Generate collision+ (Chosen-Plaintext Attack) threshold configuration.

This script combines statistic.py and analysis.py to generate the distance
distribution configuration for collision+ attack using the specific "bitter lesson"
input text.

Usage:
    python attack/get_collision_threshold.py \
        --model_path ~/model/Llama-3.2-1B \
        --target_data_path cache/float32/config/Llama-3.2-1B/<hash>/origin/past_key_values.pt \
        --device cuda:0 \
        --dtype float32
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

try:
    from .collision_threshold_stats import (
        finalize_statistics,
        load_checkpoint,
        new_layer_statistics,
        save_checkpoint,
        stable_cache_digest,
        update_position_statistics,
        validate_progress_counts,
        write_json_atomic,
    )
except ImportError:  # Support direct `python attack/get_collision_threshold.py`.
    from collision_threshold_stats import (
        finalize_statistics,
        load_checkpoint,
        new_layer_statistics,
        save_checkpoint,
        stable_cache_digest,
        update_position_statistics,
        validate_progress_counts,
        write_json_atomic,
    )


def _cache_key_values(cache: Any) -> tuple[list[Tensor], list[Tensor]]:
    """Return K/V layer lists for DynamicCache and legacy cache tuples."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(cache.key_cache), list(cache.value_cache)
    layers = list(cache)
    return [layer[0] for layer in layers], [layer[1] for layer in layers]


def _current_position_distances(
    current_cache: Any,
    target_datas: Any,
    *,
    seq_id: int,
) -> list[tuple[Tensor, Tensor]]:
    """Return one scalar K/V distance per candidate and layer."""
    current_keys, current_values = _cache_key_values(current_cache)
    target_layers = list(target_datas)
    if len(current_keys) != len(current_values):
        raise ValueError(
            f"current cache has {len(current_keys)} K layers and "
            f"{len(current_values)} V layers"
        )
    if len(current_keys) != len(target_layers):
        raise ValueError(
            f"current cache has {len(current_keys)} layers, target has "
            f"{len(target_layers)}"
        )

    result: list[tuple[Tensor, Tensor]] = []
    for layer_idx, (candidate_k, candidate_v, target_layer) in enumerate(
        zip(current_keys, current_values, target_layers)
    ):
        if len(target_layer) != 2:
            raise ValueError(f"target layer {layer_idx} must contain K and V")
        target_k, target_v = target_layer
        for component_name, candidate, target in (
            ("K", candidate_k, target_k),
            ("V", candidate_v, target_v),
        ):
            if candidate.ndim != 4 or target.ndim != 4:
                raise ValueError(
                    f"layer {layer_idx} {component_name} tensors must be rank 4"
                )
            if candidate.shape[2] < 1:
                raise ValueError(
                    f"layer {layer_idx} {component_name} candidate cache is empty"
                )
            if not 0 <= seq_id < target.shape[2]:
                raise ValueError(
                    f"sequence position {seq_id} is outside layer {layer_idx} "
                    f"{component_name} target length {target.shape[2]}"
                )
            candidate_shape = (candidate.shape[1], candidate.shape[3])
            target_shape = (target.shape[1], target.shape[3])
            if target.shape[0] != 1 or candidate_shape != target_shape:
                raise ValueError(
                    f"layer {layer_idx} {component_name} shape mismatch: "
                    f"candidate [B,{candidate_shape[0]},{candidate_shape[1]}], "
                    f"target {tuple(target.shape)}"
                )

        candidate_k_position = candidate_k[:, :, -1, :]
        candidate_v_position = candidate_v[:, :, -1, :]
        target_k_position = target_k[:, :, seq_id, :]
        target_v_position = target_v[:, :, seq_id, :]
        k_dist = torch.linalg.vector_norm(
            candidate_k_position - target_k_position, dim=(1, 2)
        )
        v_dist = torch.linalg.vector_norm(
            candidate_v_position - target_v_position, dim=(1, 2)
        )
        result.append((k_dist, v_dist))
    return result


def _expanded_prefix_cache(current_kvcache: Any, batch_size: int) -> Any:
    if current_kvcache is None:
        return None
    current_keys, current_values = _cache_key_values(current_kvcache)
    expanded_cache = DynamicCache()
    for layer_idx, (key, value) in enumerate(zip(current_keys, current_values)):
        expanded_cache.update(
            key.expand(batch_size, -1, -1, -1),
            value.expand(batch_size, -1, -1, -1),
            layer_idx,
        )
    return expanded_cache


def collect_position_distances(
    model: AutoModelForCausalLM,
    target_datas: Any,
    *,
    current_kvcache: Any,
    seq_id: int,
    batch_size: int,
) -> list[tuple[Tensor, Tensor]]:
    """Evaluate the complete vocabulary with bounded GPU distance memory."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    vocab_size = int(model.config.vocab_size)
    if vocab_size <= 0:
        raise ValueError("model vocabulary size must be positive")

    if current_kvcache is None:
        past_length = 0
    else:
        prefix_keys, _ = _cache_key_values(current_kvcache)
        past_length = int(prefix_keys[0].shape[2]) if prefix_keys else 0
    if past_length != seq_id:
        raise ValueError(
            f"known-prefix cache length {past_length} does not match sequence "
            f"position {seq_id}"
        )

    num_layers = len(target_datas)
    chunks: list[list[list[Tensor]]] = [
        [[], []] for _ in range(num_layers)
    ]
    for batch_start in range(0, vocab_size, batch_size):
        batch_end = min(batch_start + batch_size, vocab_size)
        candidate_count = batch_end - batch_start
        input_batch = torch.arange(
            batch_start,
            batch_end,
            dtype=torch.long,
            device=model.device,
        ).unsqueeze(1)
        attention_mask = torch.ones(
            (candidate_count, past_length + 1),
            dtype=torch.long,
            device=model.device,
        )
        expanded_cache = _expanded_prefix_cache(current_kvcache, candidate_count)
        try:
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_batch,
                    attention_mask=attention_mask,
                    past_key_values=expanded_cache,
                    use_cache=True,
                    output_hidden_states=False,
                )
            batch_distances = _current_position_distances(
                outputs.past_key_values,
                target_datas,
                seq_id=seq_id,
            )
            for layer_idx, (k_dist, v_dist) in enumerate(batch_distances):
                chunks[layer_idx][0].append(
                    k_dist.detach().to(device="cpu", dtype=torch.float32)
                )
                chunks[layer_idx][1].append(
                    v_dist.detach().to(device="cpu", dtype=torch.float32)
                )
        except torch.OutOfMemoryError as error:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during Collision+ calibration at sequence position "
                f"{seq_id}, candidates [{batch_start}:{batch_end}), batch size "
                f"{batch_size}. Retry this checkpoint with a smaller "
                "COLLISION_PLUS_BATCH_SIZE."
            ) from error
        finally:
            if "outputs" in locals():
                del outputs
            del expanded_cache, attention_mask, input_batch

    result: list[tuple[Tensor, Tensor]] = []
    for layer_idx, (k_chunks, v_chunks) in enumerate(chunks):
        if not k_chunks or not v_chunks:
            raise RuntimeError(f"layer {layer_idx} produced no distance chunks")
        k_distances = torch.cat(k_chunks)
        v_distances = torch.cat(v_chunks)
        if k_distances.numel() != vocab_size or v_distances.numel() != vocab_size:
            raise RuntimeError(
                f"layer {layer_idx} produced incomplete vocabulary distances: "
                f"K={k_distances.numel()}, V={v_distances.numel()}, "
                f"expected={vocab_size}"
            )
        if not bool(torch.isfinite(k_distances).all()) or not bool(
            torch.isfinite(v_distances).all()
        ):
            raise FloatingPointError(
                f"layer {layer_idx} produced non-finite candidate distances"
            )
        result.append((k_distances, v_distances))
    return result


def rebuild_prefix_cache(
    model: AutoModelForCausalLM,
    input_ids: Tensor,
    *,
    next_seq_id: int,
) -> Any:
    """Rebuild the true-token prefix represented by a streaming checkpoint."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("calibration input_ids must have shape [1, sequence_length]")
    if not 0 <= next_seq_id <= input_ids.shape[1]:
        raise ValueError(
            f"next_seq_id {next_seq_id} is outside sequence length {input_ids.shape[1]}"
        )
    if next_seq_id == 0:
        return None
    current_kvcache = None
    for seq_id in range(next_seq_id):
        current_kvcache = _append_true_token(
            model,
            input_ids,
            seq_id=seq_id,
            current_kvcache=current_kvcache,
        )
    return current_kvcache


def _append_true_token(
    model: AutoModelForCausalLM,
    input_ids: Tensor,
    *,
    seq_id: int,
    current_kvcache: Any,
) -> Any:
    token = input_ids[:, seq_id : seq_id + 1]
    attention_mask = torch.ones(
        (1, seq_id + 1), dtype=torch.long, device=model.device
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=token,
            attention_mask=attention_mask,
            past_key_values=current_kvcache,
            use_cache=True,
            output_hidden_states=False,
        )
    return outputs.past_key_values


def calibrate_streaming(
    model: AutoModelForCausalLM,
    target_datas: Any,
    *,
    input_ids: Tensor,
    batch_size: int,
    checkpoint_path: Path,
    metadata: dict[str, Any],
) -> list[dict[str, list[float]]]:
    """Calibrate exact full-vocabulary thresholds with resumable CPU stats."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("calibration input_ids must have shape [1, sequence_length]")
    sequence_length = int(input_ids.shape[1])
    vocab_size = int(model.config.vocab_size)
    num_layers = len(target_datas)

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        next_seq_id, layer_statistics = load_checkpoint(
            checkpoint_path,
            expected_metadata=metadata,
        )
        print(
            f"Resuming Collision+ calibration at sequence position "
            f"{next_seq_id}/{sequence_length}"
        )
    else:
        next_seq_id = 0
        layer_statistics = new_layer_statistics(num_layers)

    if len(layer_statistics) != num_layers:
        raise ValueError(
            f"checkpoint has {len(layer_statistics)} layers, target has {num_layers}"
        )
    if not 0 <= next_seq_id <= sequence_length:
        raise ValueError(
            f"checkpoint next_seq_id {next_seq_id} is outside sequence length "
            f"{sequence_length}"
        )
    validate_progress_counts(
        layer_statistics,
        completed_positions=next_seq_id,
        vocab_size=vocab_size,
    )
    if next_seq_id == sequence_length:
        return finalize_statistics(
            layer_statistics,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
        )

    current_kvcache = rebuild_prefix_cache(
        model,
        input_ids,
        next_seq_id=next_seq_id,
    )
    for seq_id in tqdm(
        range(next_seq_id, sequence_length),
        desc="Calculating exact Collision+ distances",
        initial=next_seq_id,
        total=sequence_length,
    ):
        position_distances = collect_position_distances(
            model,
            target_datas,
            current_kvcache=current_kvcache,
            seq_id=seq_id,
            batch_size=batch_size,
        )
        true_token_id = int(input_ids[0, seq_id].item())
        update_position_statistics(
            layer_statistics,
            position_distances,
            true_token_id=true_token_id,
        )
        save_checkpoint(
            checkpoint_path,
            next_seq_id=seq_id + 1,
            layer_statistics=layer_statistics,
            metadata=metadata,
            batch_size=batch_size,
        )
        current_kvcache = _append_true_token(
            model,
            input_ids,
            seq_id=seq_id,
            current_kvcache=current_kvcache,
        )

    return finalize_statistics(
        layer_statistics,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate collision+ (CPA) threshold configuration."
    )
    parser.add_argument(
        "--model_path",
        default="~/model/Llama-3.2-1B",
        help="Path to the model.",
    )
    parser.add_argument(
        "--target_data_path",
        required=True,
        help="Path to the target KV-cache (past_key_values.pt).",
    )
    parser.add_argument(
        "--input_text",
        default='"One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning. The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries."',
        help="Input text used to generate the KV-cache.",
    )
    parser.add_argument(
        "--target",
        default="past_key_values",
        choices=["past_key_values", "hidden_states"],
        help="Target data type.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Candidate micro-batch size for exact full-vocabulary calibration.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Data type.",
    )
    parser.add_argument(
        "--protect_type",
        default="origin",
        help="Protection type.",
    )
    parser.add_argument(
        "--target_model_name",
        default=None,
        help="Target model name (defaults to model_path name).",
    )
    return parser


def main():
    args = build_parser().parse_args()

    torch.manual_seed(42)
    torch.serialization.add_safe_globals([DynamicCache, set])

    # Parse paths
    model_path = Path(args.model_path).expanduser()
    target_data_path = Path(args.target_data_path).expanduser()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not target_data_path.exists():
        raise FileNotFoundError(f"Target data path not found: {target_data_path}")

    target_model_name = args.target_model_name or model_path.name
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    dtype_name = str(dtype).split(".")[-1]

    # Calculate input hash
    input_hash = hashlib.sha1(args.input_text.encode("utf-8")).hexdigest()

    print(f"Model: {model_path.name}")
    print(f"Target model: {target_model_name}")
    print(f"Target data: {target_data_path}")
    print(f"Input hash: {input_hash}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype_name}")
    print()

    if args.target != "past_key_values":
        raise ValueError(
            "exact streaming calibration currently supports target=past_key_values only"
        )

    # Load model and data
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, attn_implementation="eager"
    ).to(device, dtype)
    model.eval()

    print("Loading target data...")
    target_datas_cpu = torch.load(
        target_data_path, map_location="cpu", weights_only=True
    )
    target_digest = stable_cache_digest(target_datas_cpu)
    target_keys, target_values = _cache_key_values(target_datas_cpu)
    target_datas = tuple(
        (
            key.to(device=device, dtype=dtype),
            value.to(device=device, dtype=dtype),
        )
        for key, value in zip(target_keys, target_values)
    )
    del target_datas_cpu

    # Prepare inputs
    inputs = tokenizer(args.input_text, return_tensors="pt").to(device)
    bos_token_id = model.config.bos_token_id

    if bos_token_id is not None:
        if isinstance(bos_token_id, int):
            bos_token_id = [bos_token_id]

        bos_len = len(bos_token_id)
        if inputs["input_ids"][-1, :bos_len].tolist() != bos_token_id:
            inputs["input_ids"] = torch.cat(
                [torch.tensor([bos_token_id]).to(device), inputs["input_ids"]], dim=1
            )
            inputs["attention_mask"] = torch.cat(
                [torch.ones(1, bos_len).to(device), inputs["attention_mask"]], dim=1
            )

    input_ids = inputs["input_ids"]
    sequence_length = int(input_ids.shape[1])
    if len(target_datas) != int(model.config.num_hidden_layers):
        raise ValueError(
            f"target cache has {len(target_datas)} layers, model has "
            f"{model.config.num_hidden_layers}"
        )
    for layer_idx, (key, value) in enumerate(target_datas):
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise ValueError(
                f"target layer {layer_idx} must contain equal rank-4 K/V tensors"
            )
        if key.shape[0] != 1 or key.shape[2] != sequence_length:
            raise ValueError(
                f"target layer {layer_idx} shape {tuple(key.shape)} does not match "
                f"calibration sequence length {sequence_length}"
            )

    # Step 1: Generate exact streaming distance statistics
    print("\n" + "=" * 60)
    print("Step 1: Generating exact streaming distance statistics...")
    print("=" * 60)
    target_dist_dir = target_data_path.parent / f"{args.target}_dist"
    target_dist_dir.mkdir(parents=True, exist_ok=True)
    legacy_files = sorted(target_dist_dir.glob("seq=*.pt"))
    if legacy_files:
        print(
            f"Warning: ignoring {len(legacy_files)} legacy distance files in "
            f"{target_dist_dir}"
        )
    checkpoint_path = target_dist_dir / "streaming_stats_v2.pt"
    metadata = {
        "target_digest": target_digest,
        "input_hash": input_hash,
        "model_path": str(model_path.resolve()),
        "model_name": model_path.name,
        "target_model_name": target_model_name,
        "vocab_size": int(model.config.vocab_size),
        "num_layers": len(target_datas),
        "sequence_length": sequence_length,
        "dtype": dtype_name,
        "target": args.target,
    }
    start_time = time.time()
    statistics = calibrate_streaming(
        model,
        target_datas,
        input_ids=input_ids,
        batch_size=args.batch_size,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )
    print(f"\nStreaming checkpoint: {checkpoint_path}")
    print(f"Time: {time.time() - start_time:.2f}s")

    for layer_idx, layer_statistics in enumerate(statistics):
        layer_statistics.update(
            {
                "L0_model_name": model_path.name,
                "L1_model_name": target_model_name,
                "input_hash": input_hash,
                "seq_len": sequence_length,
                "target": args.target,
                "layer_idx": layer_idx,
            }
        )

    # Save configuration
    output_path = Path(
        f"attack/config/{args.protect_type}/{dtype_name}/{target_model_name}.json"
    )
    write_json_atomic(output_path, statistics)

    print(f"\nCollision+ configuration saved to: {output_path}")
    print("\nYou can now run collision+ attack with: --enhance")


if __name__ == "__main__":
    main()

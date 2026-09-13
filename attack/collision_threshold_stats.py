"""Bounded-memory statistics for exact Collision+ calibration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


COMPONENT_NAMES = ("K", "V")
CHECKPOINT_FORMAT = "collision-plus-streaming-v2"


@dataclass
class RunningStats:
    """Mergeable float64 statistics without retaining source observations."""

    count: int = 0
    mean_value: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: Tensor) -> None:
        values = values.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        if values.numel() == 0:
            return
        if not bool(torch.isfinite(values).all()):
            raise ValueError("distance statistics contain non-finite values")

        chunk_count = int(values.numel())
        chunk_mean = float(values.mean().item())
        centered = values - chunk_mean
        chunk_m2 = float(torch.sum(centered * centered).item())
        chunk_minimum = float(values.min().item())
        chunk_maximum = float(values.max().item())

        if self.count == 0:
            self.count = chunk_count
            self.mean_value = chunk_mean
            self.m2 = chunk_m2
            self.minimum = chunk_minimum
            self.maximum = chunk_maximum
            return

        combined_count = self.count + chunk_count
        delta = chunk_mean - self.mean_value
        self.mean_value += delta * chunk_count / combined_count
        self.m2 += (
            chunk_m2
            + delta * delta * self.count * chunk_count / combined_count
        )
        self.count = combined_count
        self.minimum = min(self.minimum, chunk_minimum)
        self.maximum = max(self.maximum, chunk_maximum)

    @property
    def mean(self) -> float:
        return self.mean_value if self.count else 0.0

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(max(self.m2, 0.0) / (self.count - 1))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_value": self.mean_value,
            "m2": self.m2,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunningStats":
        result = cls(
            count=int(payload["count"]),
            mean_value=float(payload["mean_value"]),
            m2=float(payload["m2"]),
            minimum=float(payload["minimum"]),
            maximum=float(payload["maximum"]),
        )
        if result.count < 0 or result.m2 < 0:
            raise ValueError("invalid serialized running statistics")
        if result.count and not all(
            math.isfinite(value)
            for value in (
                result.mean_value,
                result.m2,
                result.minimum,
                result.maximum,
            )
        ):
            raise ValueError("serialized running statistics are non-finite")
        return result


LayerStatistics = dict[str, list[RunningStats]]


def new_layer_statistics(num_layers: int) -> list[LayerStatistics]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    return [
        {
            "target": [RunningStats(), RunningStats()],
            "others": [RunningStats(), RunningStats()],
        }
        for _ in range(num_layers)
    ]


def update_position_statistics(
    layer_statistics: Sequence[LayerStatistics],
    position_distances: Sequence[tuple[Tensor, Tensor]],
    *,
    true_token_id: int,
) -> None:
    if len(position_distances) != len(layer_statistics):
        raise ValueError(
            f"expected {len(layer_statistics)} layers, got {len(position_distances)}"
        )

    vocab_size: int | None = None
    for layer_index, components in enumerate(position_distances):
        if len(components) != 2:
            raise ValueError(f"layer {layer_index} must contain K and V distances")
        k_distances, v_distances = components
        for component_name, values in zip(COMPONENT_NAMES, components):
            if not isinstance(values, Tensor) or values.ndim != 1:
                raise ValueError(
                    f"layer {layer_index} {component_name} distances must be one-dimensional"
                )
            if values.device.type != "cpu":
                raise ValueError(
                    f"layer {layer_index} {component_name} distances must be on CPU"
                )
        if k_distances.numel() != v_distances.numel():
            raise ValueError(
                f"layer {layer_index} K/V distances must have equal vocabulary length"
            )
        if vocab_size is None:
            vocab_size = int(k_distances.numel())
        elif k_distances.numel() != vocab_size:
            raise ValueError(
                f"layer {layer_index} distances have vocabulary length "
                f"{k_distances.numel()}, expected {vocab_size}"
            )

    if vocab_size is None or not 0 <= true_token_id < vocab_size:
        raise ValueError(
            f"true token ID {true_token_id} is outside vocabulary size {vocab_size or 0}"
        )

    for state, components in zip(layer_statistics, position_distances):
        for component_index, values in enumerate(components):
            state["target"][component_index].update(
                values[true_token_id : true_token_id + 1]
            )
            state["others"][component_index].update(values[:true_token_id])
            state["others"][component_index].update(values[true_token_id + 1 :])


def finalize_statistics(
    layer_statistics: Sequence[LayerStatistics],
    *,
    sequence_length: int,
    vocab_size: int,
) -> list[dict[str, list[float]]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than one")

    expected_target_count = sequence_length
    expected_others_count = sequence_length * (vocab_size - 1)
    result: list[dict[str, list[float]]] = []

    for layer_index, state in enumerate(layer_statistics):
        for component_index, component_name in enumerate(COMPONENT_NAMES):
            target = state["target"][component_index]
            others = state["others"][component_index]
            if target.count != expected_target_count:
                raise ValueError(
                    f"layer {layer_index} {component_name} target count: expected "
                    f"{expected_target_count}, got {target.count}"
                )
            if others.count != expected_others_count:
                raise ValueError(
                    f"layer {layer_index} {component_name} others count: expected "
                    f"{expected_others_count}, got {others.count}"
                )

        layer_result = {
            "target_mean": [item.mean for item in state["target"]],
            "target_std": [item.std for item in state["target"]],
            "target_max": [item.maximum for item in state["target"]],
            "others_mean": [item.mean for item in state["others"]],
            "others_std": [item.std for item in state["others"]],
            "others_min": [item.minimum for item in state["others"]],
        }
        if not all(
            math.isfinite(value)
            for values in layer_result.values()
            for value in values
        ):
            raise ValueError(f"layer {layer_index} final statistics are non-finite")
        result.append(layer_result)

    return result


def _legacy_layers(cache: Any) -> list[tuple[Tensor, Tensor]]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    layers = list(cache)
    result: list[tuple[Tensor, Tensor]] = []
    for layer_index, layer in enumerate(layers):
        if len(layer) != 2 or not all(isinstance(item, Tensor) for item in layer):
            raise TypeError(f"cache layer {layer_index} must contain K/V tensors")
        result.append((layer[0], layer[1]))
    if not result:
        raise ValueError("target cache must contain at least one layer")
    return result


def stable_cache_digest(cache: Any) -> str:
    """Hash cache tensor identity independently of torch.save container bytes."""
    digest = hashlib.sha256()
    for layer_index, (key, value) in enumerate(_legacy_layers(cache)):
        for component_name, tensor in zip(COMPONENT_NAMES, (key, value)):
            cpu_tensor = tensor.detach().to(device="cpu").contiguous()
            header = json.dumps(
                {
                    "layer": layer_index,
                    "component": component_name,
                    "dtype": str(cpu_tensor.dtype),
                    "shape": list(cpu_tensor.shape),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(header).to_bytes(8, byteorder="big"))
            digest.update(header)
            digest.update(cpu_tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _serialize_layer_statistics(
    layer_statistics: Sequence[LayerStatistics],
) -> list[dict[str, list[dict[str, float | int]]]]:
    return [
        {
            population: [stats.to_dict() for stats in layer[population]]
            for population in ("target", "others")
        }
        for layer in layer_statistics
    ]


def _deserialize_layer_statistics(payload: Any) -> list[LayerStatistics]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("checkpoint layer statistics must be a non-empty list")
    result: list[LayerStatistics] = []
    for layer_index, layer in enumerate(payload):
        if not isinstance(layer, Mapping):
            raise ValueError(f"checkpoint layer {layer_index} is malformed")
        restored: LayerStatistics = {}
        for population in ("target", "others"):
            components = layer.get(population)
            if not isinstance(components, list) or len(components) != 2:
                raise ValueError(
                    f"checkpoint layer {layer_index} {population} must contain K/V"
                )
            restored[population] = [
                RunningStats.from_dict(component) for component in components
            ]
        result.append(restored)
    return result


def save_checkpoint(
    checkpoint_path: Path,
    *,
    next_seq_id: int,
    layer_statistics: Sequence[LayerStatistics],
    metadata: Mapping[str, Any],
    batch_size: int,
) -> None:
    if next_seq_id < 0:
        raise ValueError("next_seq_id must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "next_seq_id": next_seq_id,
        "metadata": dict(metadata),
        "execution": {"batch_size": batch_size},
        "layer_statistics": _serialize_layer_statistics(layer_statistics),
    }
    torch.save(payload, temp_path)
    os.replace(temp_path, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    *,
    expected_metadata: Mapping[str, Any],
) -> tuple[int, list[LayerStatistics]]:
    checkpoint_path = Path(checkpoint_path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(
            f"could not load Collision+ checkpoint {checkpoint_path}: {error}"
        ) from error
    if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported Collision+ checkpoint format in {checkpoint_path}"
        )
    actual_metadata = payload.get("metadata")
    if not isinstance(actual_metadata, Mapping):
        raise ValueError("Collision+ checkpoint metadata is malformed")
    for field in sorted(set(expected_metadata) | set(actual_metadata)):
        expected = expected_metadata.get(field)
        actual = actual_metadata.get(field)
        if actual != expected:
            raise ValueError(
                f"checkpoint metadata mismatch for {field}: expected {expected}, "
                f"got {actual}"
            )
    next_seq_id = int(payload.get("next_seq_id", -1))
    if next_seq_id < 0:
        raise ValueError("Collision+ checkpoint next_seq_id is invalid")
    return next_seq_id, _deserialize_layer_statistics(
        payload.get("layer_statistics")
    )


def write_json_atomic(output_path: Path, payload: Any) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, output_path)

"""Adapter between Shadow's legacy KV cache and KIVI's packed cache.

The adapter deliberately owns only representation conversion and cache
partitioning.  Quantization and dequantization are delegated to the public
functions from the pinned KIVI checkout.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import torch

try:
    from .provenance import KIVI_COMMIT, KVCLOAK_COMMIT
except ImportError:  # Support legacy ``sys.path += ['src']; import kivi_adapter``.
    from provenance import KIVI_COMMIT, KVCLOAK_COMMIT


try:
    from quant.new_pack import (
        triton_quantize_and_pack_along_last_dim,
        unpack_and_dequant_vcache,
    )
except ModuleNotFoundError as import_error:
    # Keep normal command lines usable from a fresh checkout while retaining
    # the direct import of KIVI's public API above.
    _kivi_root = Path(__file__).resolve().parents[1] / "third_party" / "KIVI"
    if not (_kivi_root / "quant" / "new_pack.py").is_file():
        raise ModuleNotFoundError(
            "KIVI is unavailable. Initialize third_party/KIVI or set "
            "PYTHONPATH to the KIVI checkout."
        ) from import_error
    sys.path.insert(0, str(_kivi_root))
    from quant.new_pack import (
        triton_quantize_and_pack_along_last_dim,
        unpack_and_dequant_vcache,
    )


@dataclass(frozen=True)
class KIVIConfig:
    """KIVI cache parameters used by one adapter invocation."""

    k_bits: int
    v_bits: int
    group_size: int
    residual_length: int
    mode: str = "standard"

    def __post_init__(self) -> None:
        if self.k_bits not in (2, 4, 8):
            raise ValueError("k_bits must be one of 2, 4, or 8")
        if self.v_bits not in (2, 4, 8):
            raise ValueError("v_bits must be one of 2, 4, or 8")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.residual_length <= 0:
            raise ValueError("residual_length must be positive")
        if self.mode not in {"standard", "full_prompt_quantized"}:
            raise ValueError(
                "mode must be 'standard' or 'full_prompt_quantized'"
            )


def _legacy_layers(past_key_values: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Convert a DynamicCache or legacy cache into Shadow's layer tuples."""
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()

    layers = tuple(past_key_values)
    normalized: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index, layer in enumerate(layers):
        if len(layer) != 2:
            raise ValueError(f"layer {layer_index} must contain exactly key and value tensors")
        key, value = layer
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError(f"layer {layer_index} must contain torch.Tensor values")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(f"layer {layer_index} tensors must have shape [B, H, T, D]")
        if key.shape != value.shape:
            raise ValueError(
                f"layer {layer_index} key/value shapes differ: {key.shape} vs {value.shape}"
            )
        if not key.dtype.is_floating_point or not value.dtype.is_floating_point:
            raise TypeError(f"layer {layer_index} key/value tensors must be floating point")
        normalized.append((key, value))
    return tuple(normalized)


def _key_partition_lengths(
    sequence_length: int, config: KIVIConfig
) -> tuple[int, int]:
    """Return KIVI key prefix and full-precision suffix lengths.

    KIVI's key path groups along the sequence dimension.  Its public kernel
    therefore requires the quantized prefix to be divisible by group_size.
    We retain KIVI's residual-oriented boundary and move the boundary left to
    the nearest valid group when a caller supplies an otherwise incompatible
    sequence length.
    """
    if config.mode == "full_prompt_quantized":
        # The public KIVI key kernel groups along the sequence dimension and
        # packs several quantized values into one int32.  Padding is outside
        # the prompt and is removed again by dequantize_cache().
        required_multiple = math.lcm(config.group_size, 32 // config.k_bits)
        padded_length = math.ceil(sequence_length / required_multiple) * required_multiple
        return padded_length, 0

    if sequence_length < config.residual_length:
        return 0, sequence_length

    candidate = (
        sequence_length
        if sequence_length % config.residual_length == 0
        else sequence_length - (sequence_length % config.residual_length)
    )
    quantized_length = candidate - candidate % config.group_size
    return quantized_length, sequence_length - quantized_length


def _value_partition_lengths(
    sequence_length: int, config: KIVIConfig
) -> tuple[int, int]:
    """Return KIVI value prefix and recent full-precision suffix lengths."""
    if config.mode == "full_prompt_quantized":
        return sequence_length, 0
    if sequence_length <= config.residual_length:
        return 0, sequence_length
    quantized_length = sequence_length - config.residual_length
    return quantized_length, config.residual_length


def _quantize_key(
    key_prefix: torch.Tensor, config: KIVIConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features_per_int = 32 // config.k_bits
    if key_prefix.shape[2] % features_per_int != 0:
        raise ValueError(
            "KIVI key quantization requires quantized sequence length divisible "
            f"by {features_per_int} ({key_prefix.shape[2]} % {features_per_int} != 0)"
        )
    return triton_quantize_and_pack_along_last_dim(
        key_prefix.transpose(2, 3).contiguous(),
        config.group_size,
        config.k_bits,
    )


def _quantize_value(
    value_prefix: torch.Tensor, config: KIVIConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if value_prefix.shape[-1] % config.group_size != 0:
        raise ValueError(
            "KIVI value quantization requires head dimension divisible by "
            f"group_size ({value_prefix.shape[-1]} % {config.group_size} != 0)"
        )
    features_per_int = 32 // config.v_bits
    if value_prefix.shape[-1] % features_per_int != 0:
        raise ValueError(
            "KIVI value quantization requires head dimension divisible by "
            f"{features_per_int} ({value_prefix.shape[-1]} % {features_per_int} != 0)"
        )
    return triton_quantize_and_pack_along_last_dim(
        value_prefix.contiguous(),
        config.group_size,
        config.v_bits,
    )


def _pad_sequence(value: torch.Tensor, target_length: int) -> torch.Tensor:
    """Zero-pad the sequence axis for KIVI's grouped key kernel."""
    current_length = value.shape[2]
    if target_length < current_length:
        raise ValueError(
            f"target sequence length {target_length} is shorter than {current_length}"
        )
    if target_length == current_length:
        return value
    padding = value.new_zeros(
        (*value.shape[:2], target_length - current_length, value.shape[3])
    )
    return torch.cat((value, padding), dim=2).contiguous()


def quantize_cache(
    past_key_values: Any,
    config: KIVIConfig,
) -> dict[str, Any]:
    """Pack a Shadow cache into a serializable native KIVI representation."""
    layers = _legacy_layers(past_key_values)
    native_layers: list[dict[str, Any]] = []

    for layer_index, (key, value) in enumerate(layers):
        sequence_length = key.shape[2]
        key_quantized_length, key_residual_length = _key_partition_lengths(
            sequence_length, config
        )
        value_quantized_length, value_residual_length = _value_partition_lengths(
            sequence_length, config
        )

        key_quantized_input = key[:, :, :key_quantized_length, :]
        if key_quantized_length > sequence_length:
            key_quantized_input = _pad_sequence(key, key_quantized_length)

        if key_quantized_length:
            key_code, key_scale, key_min = _quantize_key(
                key_quantized_input, config
            )
        else:
            key_code = key_scale = key_min = None

        if value_quantized_length:
            value_code, value_scale, value_min = _quantize_value(
                value[:, :, :value_quantized_length, :], config
            )
        else:
            value_code = value_scale = value_min = None

        native_layers.append(
            {
                "key_code": key_code,
                "key_scale": key_scale,
                "key_min": key_min,
                "key_residual": key[:, :, min(key_quantized_length, sequence_length):, :].contiguous(),
                "value_code": value_code,
                "value_scale": value_scale,
                "value_min": value_min,
                "value_residual": value[:, :, value_quantized_length:, :].contiguous(),
                "seq_len": sequence_length,
                "key_quantized_length": key_quantized_length,
                "key_residual_length": key_residual_length,
                "key_valid_length": sequence_length,
                "key_padded_length": key_quantized_length,
                "value_quantized_length": value_quantized_length,
                "value_residual_length": value_residual_length,
                "value_valid_length": sequence_length,
                "key_shape": tuple(key.shape),
                "value_shape": tuple(value.shape),
                "layer_index": layer_index,
            }
        )

    residual_tokens_by_layer = [
        max(layer["key_residual_length"], layer["value_residual_length"])
        for layer in native_layers
    ]
    return {
        "format": "kivi-native-v1",
        "config": asdict(config),
        "mode": config.mode,
        "fp_residual_prompt_tokens": max(residual_tokens_by_layer, default=0),
        "fp_residual_prompt_tokens_by_layer": residual_tokens_by_layer,
        "provenance": {
            "kivi_commit": KIVI_COMMIT,
            "kvcloak_commit": KVCLOAK_COMMIT,
        },
        "num_layers": len(native_layers),
        "layers": tuple(native_layers),
    }


def _dequantize_key(
    layer: Mapping[str, Any], config: KIVIConfig
) -> torch.Tensor | None:
    code = layer["key_code"]
    if code is None:
        return None
    dequantized_transposed = unpack_and_dequant_vcache(
        code,
        layer["key_scale"].unsqueeze(-1),
        layer["key_min"].unsqueeze(-1),
        config.group_size,
        config.k_bits,
    )
    return dequantized_transposed.transpose(2, 3).contiguous()


def _dequantize_value(
    layer: Mapping[str, Any], config: KIVIConfig
) -> torch.Tensor | None:
    code = layer["value_code"]
    if code is None:
        return None
    return unpack_and_dequant_vcache(
        code,
        layer["value_scale"].unsqueeze(-1),
        layer["value_min"].unsqueeze(-1),
        config.group_size,
        config.v_bits,
    )


def _join_regions(
    quantized_region: torch.Tensor | None,
    residual_region: torch.Tensor,
) -> torch.Tensor:
    if quantized_region is None:
        return residual_region
    if residual_region.shape[2] == 0:
        return quantized_region
    return torch.cat((quantized_region, residual_region), dim=2).contiguous()


def dequantize_cache(
    quantized_cache: Mapping[str, Any],
    config: KIVIConfig,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Reconstruct Shadow-compatible K/V tuples using KIVI's public API."""
    if quantized_cache.get("format") != "kivi-native-v1":
        raise ValueError("unsupported KIVI cache format")
    stored_config = quantized_cache.get("config")
    if stored_config is not None:
        # Caches written before the mode field was introduced are standard
        # KIVI caches.  Keep them readable while rejecting true mismatches.
        stored_config = dict(stored_config)
        stored_config.setdefault("mode", "standard")
    expected_config = asdict(config)
    if stored_config is not None and stored_config != expected_config:
        raise ValueError(
            "KIVI config does not match native cache: "
            f"{stored_config} != {expected_config}"
        )

    layers = tuple(quantized_cache["layers"])
    expected_layers = quantized_cache.get("num_layers")
    if expected_layers is not None and expected_layers != len(layers):
        raise ValueError(
            f"native cache layer count is inconsistent: {expected_layers} != {len(layers)}"
        )

    restored_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in layers:
        key = _join_regions(_dequantize_key(layer, config), layer["key_residual"])
        value = _join_regions(_dequantize_value(layer, config), layer["value_residual"])

        key = key[:, :, : int(layer.get("key_valid_length", layer["seq_len"])), :]
        value = value[:, :, : int(layer.get("value_valid_length", layer["seq_len"])), :]

        if tuple(key.shape) != tuple(layer["key_shape"]):
            raise RuntimeError(
                f"KIVI key reconstruction changed shape: {key.shape} != {layer['key_shape']}"
            )
        if tuple(value.shape) != tuple(layer["value_shape"]):
            raise RuntimeError(
                f"KIVI value reconstruction changed shape: {value.shape} != {layer['value_shape']}"
            )
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise FloatingPointError("KIVI reconstruction contains non-finite values")
        restored_layers.append((key, value))

    return tuple(restored_layers)


def roundtrip_cache(
    past_key_values: Any,
    config: KIVIConfig,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Quantize and immediately reconstruct a cache for attack/eval views."""
    return dequantize_cache(quantize_cache(past_key_values, config), config)


def cache_to_device(
    past_key_values: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str,
    *,
    dtype: torch.dtype | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Move a legacy cache to a device for KIVI or model execution."""
    return tuple(
        (
            key.to(device=device, dtype=dtype) if dtype is not None else key.to(device=device),
            value.to(device=device, dtype=dtype)
            if dtype is not None
            else value.to(device=device),
        )
        for key, value in past_key_values
    )

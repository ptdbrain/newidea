"""Validate the CUDA/PyTorch runtime and execute a real KIVI smoke test."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


CUDA_RUNTIMES = {
    "cu128": "12.8",
    "cu130": "13.0",
}


@dataclass(frozen=True)
class RuntimeInfo:
    """Validated runtime metadata reported by Torch."""

    torch_version: str
    cuda_runtime: str
    gpu_name: str
    compute_capability: tuple[int, int]


def validate_runtime(
    torch_module: Any,
    variant: str,
    expected_torch_version: str,
    *,
    device: str = "cuda:0",
) -> RuntimeInfo:
    """Reject a Torch installation that cannot satisfy the Phase 1 contract."""
    try:
        expected_cuda_runtime = CUDA_RUNTIMES[variant]
    except KeyError as error:
        raise ValueError(
            f"Unsupported CUDA variant: {variant}. Supported values: cu128, cu130"
        ) from error

    torch_version = str(torch_module.__version__)
    torch_base_version = torch_version.split("+", maxsplit=1)[0]
    if torch_base_version != expected_torch_version:
        raise RuntimeError(
            f"Phase 1 expected Torch {expected_torch_version}, got {torch_base_version} "
            f"({torch_version}). Reinstall the selected {variant} build."
        )

    if not torch_module.cuda.is_available():
        raise RuntimeError(
            f"CUDA is unavailable in Torch {torch_version}. Install the {variant} "
            "wheel and submit this check on a GPU node."
        )

    cuda_runtime = getattr(torch_module.version, "cuda", None)
    if cuda_runtime != expected_cuda_runtime:
        raise RuntimeError(
            f"Phase 1 expected CUDA runtime {expected_cuda_runtime}, got "
            f"{cuda_runtime or 'none'} from Torch {torch_version}. Install the "
            f"{variant} wheel."
        )

    return RuntimeInfo(
        torch_version=torch_version,
        cuda_runtime=cuda_runtime,
        gpu_name=str(torch_module.cuda.get_device_name(device)),
        compute_capability=tuple(torch_module.cuda.get_device_capability(device)),
    )


def run_kivi_smoke(torch_module: Any, device: str) -> str:
    """Compile and execute the pinned KIVI quantization path on one small cache."""
    import triton

    from src.kivi_adapter import KIVIConfig, roundtrip_cache

    config = KIVIConfig(
        k_bits=2,
        v_bits=2,
        group_size=32,
        residual_length=32,
    )
    source = torch_module.randn(
        1,
        4,
        64,
        64,
        device=device,
        dtype=torch_module.float16,
    )
    restored = roundtrip_cache(((source, source.clone()),), config)
    restored_key, restored_value = restored[0]
    if restored_key.shape != source.shape:
        raise RuntimeError(
            f"KIVI smoke test changed the key shape: {restored_key.shape} != {source.shape}"
        )
    if restored_value.shape != source.shape:
        raise RuntimeError(
            "KIVI smoke test changed the value shape: "
            f"{restored_value.shape} != {source.shape}"
        )
    if not all(
        torch_module.isfinite(tensor).all().item()
        for tensor in (restored_key, restored_value)
    ):
        raise RuntimeError("KIVI smoke test produced non-finite values")
    torch_module.cuda.synchronize(device)
    return str(triton.__version__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(CUDA_RUNTIMES), required=True)
    parser.add_argument("--torch-version", default="2.9.1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not args.device.startswith("cuda"):
        parser.error("Phase 1 KIVI preflight requires a CUDA device")

    import torch

    info = validate_runtime(
        torch,
        args.variant,
        args.torch_version,
        device=args.device,
    )
    print(f"torch={info.torch_version}")
    print(f"cuda_runtime={info.cuda_runtime}")
    print(f"gpu={info.gpu_name}")
    print(
        "compute_capability="
        f"{info.compute_capability[0]}.{info.compute_capability[1]}"
    )

    try:
        triton_version = run_kivi_smoke(torch, args.device)
    except Exception as error:
        raise RuntimeError(
            f"KIVI CUDA smoke test failed on {args.device} with "
            f"Torch {info.torch_version} / CUDA {info.cuda_runtime}: {error}"
        ) from error
    print(f"triton={triton_version}")
    print("KIVI CUDA smoke test: OK")


if __name__ == "__main__":
    main()

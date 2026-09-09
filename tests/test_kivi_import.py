"""CUDA compatibility smoke tests for the pinned public KIVI primitives."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "third_party" / "KIVI"))

from quant.new_pack import (  # noqa: E402
    triton_quantize_and_pack_along_last_dim,
    unpack_and_dequant_vcache,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="KIVI public kernels require CUDA")
def test_public_kivi_quantize_and_dequantize():
    source = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)

    code, scale, minimum = triton_quantize_and_pack_along_last_dim(
        source, group_size=32, bit=2
    )
    reconstructed = unpack_and_dequant_vcache(
        code,
        scale.unsqueeze(-1),
        minimum.unsqueeze(-1),
        group_size=32,
        bits=2,
    )

    assert reconstructed.shape == source.shape
    assert torch.isfinite(reconstructed).all()

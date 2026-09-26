"""3-bit extension of KIVI's group quantizer.

The pinned KIVI checkout only packs 2, 4 and 8 bits (``32 // bits`` values
per int32). This module applies KIVI's own quantization formula at 3 bits and
adds the one missing piece, a dense 3-bit packing (8 values in 3 bytes).

Formula and tensor layout mirror ``quant.new_pack`` exactly:

* ``triton_quantize_and_pack_along_last_dim``: per-group min/max along the
  last dimension, ``scale = (max - min) / (2**bits - 1)``, then
  ``(x - min) / scale`` clamped to ``[0, 2**bits - 1]`` and rounded; ``scale``
  and ``min`` have shape ``[B, H, D, num_groups]``.
* ``unpack_and_dequant_vcache``: codes cast to float16, then
  ``code * scale + min`` with ``scale``/``min`` passed as ``[..., groups, 1]``.
"""

import torch


BITS = 3
VALUES_PER_PACK = 8  # 8 values x 3 bits = 3 bytes
BYTES_PER_PACK = 3


def pack_3bit(codes: torch.Tensor) -> torch.Tensor:
    """Densely pack integer codes in ``[0, 7]`` along the last dimension."""
    if codes.shape[-1] % VALUES_PER_PACK != 0:
        raise ValueError(
            f"last dimension must be divisible by {VALUES_PER_PACK}, got {codes.shape[-1]}"
        )
    groups = codes.to(torch.int32).reshape(*codes.shape[:-1], -1, VALUES_PER_PACK)
    shifts = torch.arange(VALUES_PER_PACK, device=codes.device, dtype=torch.int32) * BITS
    words = (groups << shifts).sum(dim=-1)  # 24-bit words, disjoint bit ranges
    packed = torch.stack([(words >> (8 * i)) & 0xFF for i in range(BYTES_PER_PACK)], dim=-1)
    return packed.to(torch.uint8).reshape(*codes.shape[:-1], -1)


def unpack_3bit(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_3bit`; returns int16 codes in ``[0, 7]``."""
    if packed.shape[-1] % BYTES_PER_PACK != 0:
        raise ValueError(
            f"last dimension must be divisible by {BYTES_PER_PACK}, got {packed.shape[-1]}"
        )
    triples = packed.to(torch.int32).reshape(*packed.shape[:-1], -1, BYTES_PER_PACK)
    words = triples[..., 0] | (triples[..., 1] << 8) | (triples[..., 2] << 16)
    shifts = torch.arange(VALUES_PER_PACK, device=packed.device, dtype=torch.int32) * BITS
    codes = (words.unsqueeze(-1) >> shifts) & 0x7
    return codes.to(torch.int16).reshape(*packed.shape[:-1], -1)


def quantize_and_pack_along_last_dim(
    data: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3-bit counterpart of KIVI's ``triton_quantize_and_pack_along_last_dim``.

    Returns ``(code, scale, min)`` with ``code`` as uint8 ``[B, H, D, T*3/8]``
    and ``scale``/``min`` as ``[B, H, D, T/group_size]`` in ``data.dtype``.
    """
    if data.ndim != 4:
        raise ValueError("data must have shape [B, H, D, T]")
    batch, heads, dim, length = data.shape
    if length % group_size != 0:
        raise ValueError(f"last dimension {length} must be divisible by group_size {group_size}")
    max_int = 2**BITS - 1
    groups = data.reshape(batch, heads, dim, length // group_size, group_size)
    mn = groups.amin(dim=-1)
    mx = groups.amax(dim=-1)
    scale = (mx - mn) / max_int
    codes = groups - mn.unsqueeze(-1)
    codes.div_(scale.unsqueeze(-1))
    codes = codes.clamp_(0, max_int).round_().to(torch.int32)
    return pack_3bit(codes.reshape(batch, heads, dim, length)), scale, mn


def unpack_and_dequant(
    code: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor, group_size: int
) -> torch.Tensor:
    """3-bit counterpart of KIVI's ``unpack_and_dequant_vcache``.

    ``scale`` and ``mn`` are passed with a trailing singleton dimension,
    ``[..., num_groups, 1]``, as the KIVI adapter does for KIVI's function.
    """
    data = unpack_3bit(code)
    shape = data.shape
    data = data.view(*shape[:-1], shape[-1] // group_size, group_size)
    data = data.to(torch.float16)
    data = data * scale + mn
    return data.view(shape)

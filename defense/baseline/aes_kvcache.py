# aes_kvcache.py
"""AES-GCM encryption for KV-cache protection."""

import os
from typing import Tuple, Union
import torch
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from transformers.cache_utils import DynamicCache


class KVCacheAESProtecter:
    """AES-GCM encryption protector for KV-Cache.
    
    Encrypts and decrypts KV-Cache tensors using AES-GCM mode.
    Each tensor is encrypted with a unique nonce for security.
    
    Attributes:
        key: AES key (16 or 32 bytes)
        device: Target device for tensors
        nonce_size: Size of GCM nonce (12 bytes)
    """

    def __init__(self, key: bytes, device: Union[str, torch.device] = "cpu"):
        """Initialize AES protector.

        Args:
            key: AES key (16 or 32 bytes for AES-128/256)
            device: Device for output tensors

        Raises:
            ValueError: If key length is not 16 or 32 bytes
        """
        self.nonce_size = 12  # GCM standard nonce size
        if len(key) not in [16, 32]:
            raise ValueError("AES key must be 16 or 32 bytes (AES-128 or AES-256)")
        self.key = key
        self.device = device

    def _tensor_to_bytes(
        self, tensor: torch.Tensor
    ) -> "Tuple[bytes, Tuple[int, ...], torch.dtype]":
        """Convert tensor to bytes while preserving metadata.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tuple of (bytes_data, shape, dtype)
        """
        tensor_np = tensor.cpu().numpy()
        return tensor_np.tobytes(), tensor_np.shape, tensor.dtype

    def _bytes_to_tensor(
        self, data: bytes, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert bytes back to tensor.
        
        Args:
            data: Byte data
            shape: Original tensor shape
            dtype: Original tensor dtype
            
        Returns:
            Reconstructed tensor
        """
        # Map torch dtype to numpy dtype
        dtype_str = str(dtype).split(".")[-1]
        np_dtype = np.dtype(dtype_str)
        tensor_np = np.frombuffer(data, dtype=np_dtype).copy()
        return torch.from_numpy(tensor_np).reshape(shape).to(self.device)

    def encrypt(self, past_key_values: DynamicCache) -> list:
        """Encrypt the entire KV-Cache.

        Args:
            past_key_values: KV cache to encrypt

        Returns:
            List of encrypted layers, each containing [key_tuple, value_tuple]
            where tuple is (nonce, ciphertext, shape, dtype)
        """
        encrypted_cache = []
        for key_states, value_states in self._iter_cache_layers(past_key_values):
            self._encrypt_layer(encrypted_cache, key_states, value_states)

        return encrypted_cache

    def _iter_cache_layers(self, past_key_values: DynamicCache):
        """Yield (key, value) tensors from different DynamicCache APIs."""
        def _read_attr(obj, attr_name):
            if not hasattr(obj, attr_name):
                return None
            value = getattr(obj, attr_name)
            if callable(value):
                try:
                    return value()
                except TypeError:
                    return None
            return value

        def _normalize_entry(entry):
            # Most common: (key_tensor, value_tensor)
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                key_states, value_states = entry[0], entry[1]
                if torch.is_tensor(key_states) and torch.is_tensor(value_states):
                    return key_states, value_states

            # Dict-like containers
            if isinstance(entry, dict):
                key_states = entry.get("key") or entry.get("keys")
                value_states = entry.get("value") or entry.get("values")
                if torch.is_tensor(key_states) and torch.is_tensor(value_states):
                    return key_states, value_states

            # Object-style layer containers across transformers versions
            for key_attr, value_attr in [
                ("key_states", "value_states"),
                ("key", "value"),
                ("keys", "values"),
                ("k", "v"),
            ]:
                key_states = _read_attr(entry, key_attr)
                value_states = _read_attr(entry, value_attr)
                if torch.is_tensor(key_states) and torch.is_tensor(value_states):
                    return key_states, value_states

            return None

        # 1) Prefer legacy conversion if available (widely supported).
        to_legacy = getattr(past_key_values, "to_legacy_cache", None)
        if callable(to_legacy):
            legacy_cache = to_legacy()
            for entry in legacy_cache:
                pair = _normalize_entry(entry)
                if pair is not None:
                    yield pair
            return

        # 2) Older API with key_cache/value_cache attributes.
        try:
            key_cache = past_key_values.key_cache
            value_cache = past_key_values.value_cache
            for key_states, value_states in zip(key_cache, value_cache):
                yield key_states, value_states
            return
        except Exception:
            pass

        # 3) Iterable API.
        try:
            pending_key = None
            for entry in past_key_values:
                pair = _normalize_entry(entry)
                if pair is not None:
                    yield pair
                    continue

                if torch.is_tensor(entry):
                    if pending_key is None:
                        pending_key = entry
                    else:
                        yield pending_key, entry
                        pending_key = None
                    continue

                # Some versions expose per-layer objects via .layers
                layer_pair = _normalize_entry(entry)
                if layer_pair is not None:
                    yield layer_pair
                    continue

            if pending_key is not None:
                raise TypeError("Unpaired tensor entry found in cache iterator")
            return
        except Exception as e:
            raise TypeError(f"Unsupported KV cache object type: {type(past_key_values)}") from e
    
    def _encrypt_tensor(self, aesgcm: AESGCM, tensor: torch.Tensor) -> tuple:
        """Encrypt one tensor as (nonce, ciphertext, shape, dtype)."""
        data, shape, dtype = self._tensor_to_bytes(tensor)
        nonce = os.urandom(self.nonce_size)
        return (nonce, aesgcm.encrypt(nonce, data, None), shape, dtype)

    def _decrypt_tensor(self, aesgcm: AESGCM, encrypted: tuple) -> torch.Tensor:
        """Decrypt one (nonce, ciphertext, shape, dtype) tuple into a tensor."""
        nonce, ciphertext, shape, dtype = encrypted
        data = aesgcm.decrypt(nonce, ciphertext, None)
        return self._bytes_to_tensor(data, shape, dtype)

    def _encrypt_layer(self, encrypted_cache: list, key_states, value_states):
        """Encrypt a single layer."""
        aesgcm = AESGCM(self.key)
        encrypted_key = self._encrypt_tensor(aesgcm, key_states)
        encrypted_value = self._encrypt_tensor(aesgcm, value_states)
        encrypted_cache.append([encrypted_key, encrypted_value])

    def encrypt_native(self, native_cache: dict) -> dict:
        """Encrypt every tensor field of a packed ``kivi-native-v1`` cache.

        Packed codes, scales, minimums and FP residuals are encrypted with the
        same per-tensor AES-GCM path as :meth:`encrypt`. Non-tensor metadata
        (lengths, shapes, config) stays in plaintext, as tensor shapes already
        do for the FP cache.

        Args:
            native_cache: Output of ``src.kivi_adapter.quantize_cache``

        Returns:
            Copy of ``native_cache`` whose layers are
            ``{"encrypted": {field: (nonce, ciphertext, shape, dtype)},
            "plain": {field: value}}``
        """
        aesgcm = AESGCM(self.key)
        encrypted_layers = []
        for layer in native_cache["layers"]:
            encrypted, plain = {}, {}
            for name, value in layer.items():
                if torch.is_tensor(value):
                    encrypted[name] = self._encrypt_tensor(aesgcm, value)
                else:
                    plain[name] = value
            encrypted_layers.append({"encrypted": encrypted, "plain": plain})
        return {**native_cache, "layers": tuple(encrypted_layers)}

    def decrypt_native(self, encrypted_cache: dict) -> dict:
        """Decrypt the output of :meth:`encrypt_native` back to a native cache.

        The result can be passed directly to
        ``src.kivi_adapter.dequantize_cache``.
        """
        aesgcm = AESGCM(self.key)
        layers = []
        for layer in encrypted_cache["layers"]:
            restored = dict(layer["plain"])
            for name, encrypted in layer["encrypted"].items():
                restored[name] = self._decrypt_tensor(aesgcm, encrypted)
            layers.append(restored)
        return {**encrypted_cache, "layers": tuple(layers)}

    def decrypt(self, encrypted_cache: list) -> DynamicCache:
        """Decrypt the entire KV-Cache.

        Args:
            encrypted_cache: Encrypted cache from encrypt()

        Returns:
            Decrypted DynamicCache
        """
        decrypted_kv_list = []
        
        for encrypted_key, encrypted_value in encrypted_cache:
            aesgcm = AESGCM(self.key)
            key_states = self._decrypt_tensor(aesgcm, encrypted_key)
            value_states = self._decrypt_tensor(aesgcm, encrypted_value)
            decrypted_kv_list.append((key_states, value_states))

        # Build DynamicCache in a version-compatible way.
        try:
            return DynamicCache.from_legacy_cache(decrypted_kv_list)
        except Exception:
            cache = DynamicCache()
            for layer_idx, (key_states, value_states) in enumerate(decrypted_kv_list):
                try:
                    cache.update(key_states, value_states, layer_idx=layer_idx)
                except TypeError:
                    cache.update(key_states, value_states, layer_idx)
            return cache

    def __repr__(self) -> str:
        """String representation."""
        return f"KVCacheAESProtecter(key_len={len(self.key)*8}bits, device={self.device})"

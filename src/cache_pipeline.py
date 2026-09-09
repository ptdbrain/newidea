"""Small cache roundtrip pipelines used by utility evaluators."""

from typing import Any

from .kivi_adapter import dequantize_cache, quantize_cache, roundtrip_cache


class CachePipeline:
    """Interface for a cache transformation followed by reconstruction."""

    def roundtrip(self, past_key_values: Any) -> Any:
        raise NotImplementedError


class IdentityPipeline(CachePipeline):
    """Leave the cache untouched."""

    def roundtrip(self, past_key_values: Any) -> Any:
        return past_key_values


def _as_dynamic_cache(past_key_values: Any):
    from transformers.cache_utils import DynamicCache

    if isinstance(past_key_values, DynamicCache):
        return past_key_values
    return DynamicCache.from_legacy_cache(tuple(past_key_values))


class _LegacyCacheView:
    """Expose old cache attributes to pinned KVCloak on new Transformers."""

    def __init__(self, dynamic_cache):
        self._dynamic_cache = dynamic_cache

    @property
    def key_cache(self):
        return [key for key, _ in self._dynamic_cache]

    @property
    def value_cache(self):
        return [value for _, value in self._dynamic_cache]

    def __iter__(self):
        return iter(self._dynamic_cache)

    def __getitem__(self, index):
        return self._dynamic_cache[index]

    def __getattr__(self, name):
        return getattr(self._dynamic_cache, name)


def _as_kvcloak_cache(past_key_values: Any):
    dynamic_cache = _as_dynamic_cache(past_key_values)
    if hasattr(dynamic_cache, "key_cache"):
        return dynamic_cache
    return _LegacyCacheView(dynamic_cache)


def _restore_cache_type(original: Any, dynamic_cache: Any) -> Any:
    from transformers.cache_utils import DynamicCache

    if isinstance(original, DynamicCache):
        return dynamic_cache
    return dynamic_cache.to_legacy_cache()


class KIVIPipeline(CachePipeline):
    """KIVI quantize/dequantize roundtrip."""

    def __init__(self, config):
        self.config = config

    def roundtrip(self, past_key_values: Any) -> Any:
        restored = roundtrip_cache(past_key_values, self.config)
        if hasattr(past_key_values, "to_legacy_cache"):
            return _as_dynamic_cache(restored)
        return restored


class KVCloakPipeline(CachePipeline):
    """Existing KV-Cloak obfuscation/deobfuscation roundtrip."""

    def __init__(self, kvcloak):
        self.kvcloak = kvcloak

    def roundtrip(self, past_key_values: Any) -> Any:
        dynamic_cache = _as_dynamic_cache(past_key_values)
        protected = self.kvcloak.obfuscate(dynamic_cache)
        restored = self.kvcloak.deobfuscate(_as_kvcloak_cache(protected))
        return _restore_cache_type(past_key_values, restored)


class KVCloakKIVIPipeline(CachePipeline):
    """KV-Cloak obfuscation followed by native KIVI compression."""

    def __init__(self, kvcloak, config):
        self.kvcloak = kvcloak
        self.config = config

    def roundtrip(self, past_key_values: Any) -> Any:
        dynamic_cache = _as_dynamic_cache(past_key_values)
        protected_cache = self.kvcloak.obfuscate(dynamic_cache)
        native_cache = quantize_cache(protected_cache, self.config)
        reconstructed_protected = dequantize_cache(native_cache, self.config)
        reconstructed_protected = _as_kvcloak_cache(reconstructed_protected)
        restored = self.kvcloak.deobfuscate(reconstructed_protected)
        return _restore_cache_type(past_key_values, restored)

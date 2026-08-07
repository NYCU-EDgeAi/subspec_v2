from run.core.presets import register_presets
from run.core.registry import ModelRegistry


def test_subspec_v2_fi_experimental_methods_removed():
    original_registry = dict(ModelRegistry._registry)
    try:
        ModelRegistry._registry = {}
        register_presets()
        methods = set(ModelRegistry.list_methods())
        # v2 FlashInfer is now `method: subspec_sd_v2` + `backend: flashinfer`, not a
        # separate registry name.
        assert "subspec_sd_v2" in methods
        assert "subspec_sd_v2_fi" not in methods
        assert "subspec_sd_v2_fi_v1style" not in methods
        assert "subspec_sd_v2_fi_v1style_pv" not in methods
        # The v2 entry serves the FlashInfer backend via a per-backend override.
        entry = ModelRegistry.get("subspec_sd_v2")
        assert "flashinfer" in entry.backends
        assert entry.for_backend("flashinfer").load_kv_cache_fn is not None
    finally:
        ModelRegistry._registry = original_registry

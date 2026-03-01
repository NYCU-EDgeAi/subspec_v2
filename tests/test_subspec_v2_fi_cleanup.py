from run.core.presets import register_presets
from run.core.registry import ModelRegistry


def test_subspec_v2_fi_experimental_methods_removed():
    original_registry = dict(ModelRegistry._registry)
    try:
        ModelRegistry._registry = {}
        register_presets()
        methods = set(ModelRegistry.list_methods())
        assert "subspec_sd_v2_fi" in methods
        assert "subspec_sd_v2_fi_v1style" not in methods
        assert "subspec_sd_v2_fi_v1style_pv" not in methods
    finally:
        ModelRegistry._registry = original_registry

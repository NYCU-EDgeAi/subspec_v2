"""Unit tests for the `--set` dotlist overrides + unknown-key config validation.

CPU-only; no models. Covers run/main.py::_parse_set_overrides and
run/core/configuration.py::AppConfig.update.
"""
import pytest

from run.main import _parse_set_overrides, _deep_merge_dict
from run.core.configuration import AppConfig


def test_parse_set_coerces_scalar_types():
    ov = _parse_set_overrides(["max_length=100", "temperature=0.3", "do_sample=true"])
    assert ov == {"max_length": 100, "temperature": 0.3, "do_sample": True}
    assert isinstance(ov["max_length"], int)
    assert isinstance(ov["temperature"], float)
    assert ov["do_sample"] is True


def test_parse_set_nests_dotted_keys():
    ov = _parse_set_overrides(
        ["draft_params.max_depth=8", "generator_kwargs.verify_kwargs.threshold=0.3"]
    )
    assert ov == {
        "draft_params": {"max_depth": 8},
        "generator_kwargs": {"verify_kwargs": {"threshold": 0.3}},
    }


def test_parse_set_merges_repeated_keys_under_same_parent():
    ov = _parse_set_overrides(["draft_params.max_depth=8", "draft_params.topk_len=3"])
    assert ov == {"draft_params": {"max_depth": 8, "topk_len": 3}}


def test_parse_set_keeps_colon_value_as_string():
    # `cuda:0` must stay a string, not become a YAML mapping.
    ov = _parse_set_overrides(["device=cuda:0"])
    assert ov == {"device": "cuda:0"}


def test_parse_set_string_value_stays_string():
    ov = _parse_set_overrides(["llm_path=meta-llama/Llama-3.2-1B-Instruct"])
    assert ov == {"llm_path": "meta-llama/Llama-3.2-1B-Instruct"}


@pytest.mark.parametrize("bad", ["nokeyvalue", "=value", "  =x"])
def test_parse_set_rejects_malformed(bad):
    with pytest.raises(ValueError):
        _parse_set_overrides([bad])


def test_parse_set_none_is_empty():
    assert _parse_set_overrides(None) == {}


def test_config_update_accepts_known_keys():
    c = AppConfig()
    c.update({"max_length": 256, "backend": "flashinfer", "do_sample": True})
    assert c.max_length == 256
    assert c.backend == "flashinfer"
    assert c.do_sample is True


def test_config_update_rejects_unknown_key_with_hint():
    with pytest.raises(ValueError) as exc:
        AppConfig().update({"max_lenght": 256})
    msg = str(exc.value)
    assert "max_lenght" in msg
    assert "max_length" in msg  # did-you-mean suggestion


def test_set_override_merges_over_yaml_at_dict_layer():
    # --set is deep-merged over YAML before AppConfig; a dotted override wins per key
    # without clobbering sibling keys.
    yaml_config = {"draft_params": {"max_depth": 32, "topk_len": 6}, "max_length": 4096}
    overrides = _parse_set_overrides(["draft_params.max_depth=8", "max_length=512"])
    merged = _deep_merge_dict(yaml_config, overrides)
    assert merged == {
        "draft_params": {"max_depth": 8, "topk_len": 6},  # topk_len preserved
        "max_length": 512,
    }

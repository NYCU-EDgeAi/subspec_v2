"""Unit tests for run-sweep point expansion + model-reuse detection. CPU-only, no models."""
from run.pipelines.run_sweep import _expand_points, _model_signature, _to_nested


def test_to_nested_dotted_keys():
    assert _to_nested({"draft_params.max_depth": 8, "max_length": 512}) == {
        "draft_params": {"max_depth": 8},
        "max_length": 512,
    }


def test_expand_cartesian_axes():
    spec = {"axes": {"draft_params.max_depth": [8, 16, 32], "temperature": [0.0, 0.2]}}
    points = list(_expand_points(spec))
    assert len(points) == 6  # 3 x 2 cartesian
    _, first = points[0]
    assert first == {"draft_params.max_depth": 8, "temperature": 0.0}


def test_expand_include_times_axes_with_fixed_set():
    spec = {
        "set": {"max_length": 512},
        "include": [{"llm_path": "A", "vram_limit_gb": 8}, {"llm_path": "B", "vram_limit_gb": 12}],
        "axes": {"draft_params.max_depth": [16, 32]},
    }
    points = list(_expand_points(spec))
    assert len(points) == 4  # 2 bundles x 2 depths
    # every point carries the fixed set + its bundle + its axis value
    for _, pairs in points:
        assert pairs["max_length"] == 512
        assert pairs["llm_path"] in {"A", "B"}
        assert pairs["draft_params.max_depth"] in {16, 32}


def test_expand_no_axes_no_include_is_single_point():
    assert len(list(_expand_points({"set": {"max_length": 512}}))) == 1


def test_model_signature_ignores_generator_axes():
    base = {"llm_path": "X", "method": "m"}
    s8 = _model_signature({"draft_params.max_depth": 8}, base)
    s16 = _model_signature({"draft_params.max_depth": 16}, base)
    assert s8 == s16  # draft_params is not a model field -> reuse the resident model


def test_model_signature_detects_model_change():
    base = {"llm_path": "X", "method": "m"}
    assert _model_signature({"llm_path": "Y"}, base) != _model_signature({}, base)
    assert _model_signature({"recipe.kwargs.processor": "fp8"}, base) != _model_signature({}, base)

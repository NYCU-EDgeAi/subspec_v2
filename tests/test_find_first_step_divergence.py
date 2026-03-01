import json
from pathlib import Path

import pytest

from analysis.find_first_step_divergence import compare_step_traces


def _step(
    *,
    step: int,
    is_prev_accepted: bool = False,
    skip_nodes: int = 0,
    tree_size_before_cap: int = 8,
    tree_size_after_cap: int = 8,
    decoded_tree_size: int = 8,
    root_ind_in: int = 0,
    accept_len: int = 7,
    hidden_indices_len: int = 8,
    root_ind_out: int = 3,
    post_verify_used: bool = False,
):
    return {
        "step": int(step),
        "is_prev_accepted": bool(is_prev_accepted),
        "skip_nodes": int(skip_nodes),
        "tree_size_before_cap": int(tree_size_before_cap),
        "tree_size_after_cap": int(tree_size_after_cap),
        "decoded_tree_size": int(decoded_tree_size),
        "root_ind_in": int(root_ind_in),
        "accept_len": int(accept_len),
        "hidden_indices_len": int(hidden_indices_len),
        "root_ind_out": int(root_ind_out),
        "post_verify_used": bool(post_verify_used),
    }


def _write_multi_json(path: Path, objects: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for obj in objects:
            json.dump(obj, f, indent=4)
            f.write("\n")


def test_compare_step_traces_detects_first_field_mismatch(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    cmp = tmp_path / "cmp.jsonl"

    base_conv = {
        "0": {"step_trace": [_step(step=0), _step(step=1, accept_len=6)]},
        "overall": {},
    }
    cmp_conv = {
        "0": {"step_trace": [_step(step=0), _step(step=1, accept_len=2)]},
        "overall": {},
    }
    _write_multi_json(base, [base_conv])
    _write_multi_json(cmp, [cmp_conv])

    report = compare_step_traces(base, cmp)
    assert report["status"] == "mismatch"
    mismatch = report["mismatch"]
    assert mismatch["type"] == "field_mismatch"
    assert mismatch["turn_key"] == "0"
    assert mismatch["step_index"] == 1
    assert mismatch["field"] == "accept_len"
    assert mismatch["base_value"] == 6
    assert mismatch["cmp_value"] == 2


def test_compare_step_traces_reports_match(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    cmp = tmp_path / "cmp.jsonl"

    conv = {
        "0": {"step_trace": [_step(step=0), _step(step=1, is_prev_accepted=True)]},
        "1": {"step_trace": [_step(step=0, skip_nodes=8, post_verify_used=True)]},
        "overall": {},
    }
    _write_multi_json(base, [conv, conv])
    _write_multi_json(cmp, [conv, conv])

    report = compare_step_traces(base, cmp)
    assert report["status"] == "match"
    assert report["mismatch"] is None


def test_compare_step_traces_raises_on_missing_trace_field(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    cmp = tmp_path / "cmp.jsonl"

    _write_multi_json(base, [{"0": {"step_trace": [_step(step=0)]}, "overall": {}}])
    _write_multi_json(cmp, [{"0": {}, "overall": {}}])

    with pytest.raises(ValueError, match="trace is not a list"):
        compare_step_traces(base, cmp)

import argparse
import json
from pathlib import Path
from typing import Any


TRACE_FIELDS = [
    "is_prev_accepted",
    "skip_nodes",
    "tree_size_before_cap",
    "tree_size_after_cap",
    "decoded_tree_size",
    "root_ind_in",
    "accept_len",
    "hidden_indices_len",
    "root_ind_out",
    "post_verify_used",
]


def _load_multi_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    objs: list[dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}")
        objs.append(obj)
        idx = end
    return objs


def _turn_keys(conv_obj: dict[str, Any]) -> list[str]:
    keys = []
    for key in conv_obj.keys():
        if isinstance(key, str) and key.isdigit():
            keys.append(key)
    return sorted(keys, key=lambda x: int(x))


def _validate_trace(step_trace: Any, *, conv_idx: int, turn_key: str, side: str) -> list[dict[str, Any]]:
    if not isinstance(step_trace, list):
        raise ValueError(
            f"{side} trace is not a list at conversation={conv_idx}, turn={turn_key}."
        )
    for step_idx, row in enumerate(step_trace):
        if not isinstance(row, dict):
            raise ValueError(
                f"{side} trace row is not an object at conversation={conv_idx}, turn={turn_key}, step={step_idx}."
            )
        for field in ["step", *TRACE_FIELDS]:
            if field not in row:
                raise ValueError(
                    f"{side} trace missing field '{field}' at conversation={conv_idx}, turn={turn_key}, step={step_idx}."
                )
    return step_trace


def compare_step_traces(base_log: Path, cmp_log: Path) -> dict[str, Any]:
    base_objs = _load_multi_json(base_log)
    cmp_objs = _load_multi_json(cmp_log)

    report: dict[str, Any] = {
        "status": "match",
        "base_log": str(base_log),
        "cmp_log": str(cmp_log),
        "comparison_fields": list(TRACE_FIELDS),
        "mismatch": None,
    }

    if len(base_objs) != len(cmp_objs):
        report["status"] = "mismatch"
        report["mismatch"] = {
            "type": "conversation_count_mismatch",
            "base_count": int(len(base_objs)),
            "cmp_count": int(len(cmp_objs)),
        }
        return report

    for conv_idx, (base_conv, cmp_conv) in enumerate(zip(base_objs, cmp_objs)):
        base_turns = _turn_keys(base_conv)
        cmp_turns = _turn_keys(cmp_conv)
        if base_turns != cmp_turns:
            report["status"] = "mismatch"
            report["mismatch"] = {
                "type": "turn_key_mismatch",
                "conversation_index": int(conv_idx),
                "base_turns": base_turns,
                "cmp_turns": cmp_turns,
            }
            return report

        for turn_key in base_turns:
            base_trace = _validate_trace(
                base_conv.get(turn_key, {}).get("step_trace"),
                conv_idx=conv_idx,
                turn_key=turn_key,
                side="base",
            )
            cmp_trace = _validate_trace(
                cmp_conv.get(turn_key, {}).get("step_trace"),
                conv_idx=conv_idx,
                turn_key=turn_key,
                side="cmp",
            )

            common_len = min(len(base_trace), len(cmp_trace))
            for step_idx in range(common_len):
                base_row = base_trace[step_idx]
                cmp_row = cmp_trace[step_idx]
                for field in TRACE_FIELDS:
                    if base_row[field] != cmp_row[field]:
                        report["status"] = "mismatch"
                        report["mismatch"] = {
                            "type": "field_mismatch",
                            "conversation_index": int(conv_idx),
                            "turn_key": str(turn_key),
                            "step_index": int(step_idx),
                            "field": str(field),
                            "base_value": base_row[field],
                            "cmp_value": cmp_row[field],
                            "base_step": base_row,
                            "cmp_step": cmp_row,
                        }
                        return report

            if len(base_trace) != len(cmp_trace):
                report["status"] = "mismatch"
                report["mismatch"] = {
                    "type": "trace_length_mismatch",
                    "conversation_index": int(conv_idx),
                    "turn_key": str(turn_key),
                    "base_len": int(len(base_trace)),
                    "cmp_len": int(len(cmp_trace)),
                }
                return report

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find first step-level trace divergence.")
    parser.add_argument("--base-log", required=True, type=Path, help="Path to baseline 0.jsonl")
    parser.add_argument("--cmp-log", required=True, type=Path, help="Path to compare 0.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON report path")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        report = compare_step_traces(args.base_log, args.cmp_log)
    except ValueError as exc:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "status": "error",
                    "error": str(exc),
                    "base_log": str(args.base_log),
                    "cmp_log": str(args.cmp_log),
                    "comparison_fields": list(TRACE_FIELDS),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

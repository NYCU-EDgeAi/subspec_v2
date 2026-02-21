"""Utilities for HumanEval scoring via HuggingFace `evaluate` code_eval metric."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Sequence


def _require_code_eval_enabled() -> None:
    if os.environ.get("HF_ALLOW_CODE_EVAL", "0") != "1":
        raise RuntimeError(
            "HumanEval uses `evaluate` code_eval and executes untrusted model code. "
            "Set HF_ALLOW_CODE_EVAL=1 to acknowledge risk and continue."
        )


def validate_humaneval_runtime_requirements() -> None:
    _require_code_eval_enabled()
    try:
        import evaluate  # type: ignore[import-not-found]  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "HumanEval requires HuggingFace `evaluate` package. "
            "Install with: pip install evaluate"
        ) from exc


@lru_cache(maxsize=1)
def _load_code_eval_metric():
    validate_humaneval_runtime_requirements()
    import evaluate  # type: ignore[import-not-found]
    return evaluate.load("code_eval")


def _build_reference(problem: Dict[str, Any]) -> str:
    return f"{problem['test']}\ncheck({problem['entry_point']})"


def _build_prediction(problem: Dict[str, Any], completion: str) -> str:
    # lm-eval humaneval utils.build_predictions / build_predictions_instruct behavior.
    if problem.get("prediction_style") == "instruct":
        fence_idx = completion.find("```")
        if fence_idx != -1:
            completion = completion[:fence_idx]
    return f"{problem['prompt']}{completion}"


def _run_code_eval(
    references: List[str],
    predictions: List[List[str]],
    *,
    k: Sequence[int],
    num_workers: int,
    timeout: float,
) -> tuple[Dict[str, float], Any]:
    metric = _load_code_eval_metric()
    out = metric.compute(
        references=references,
        predictions=predictions,
        k=list(k),
        num_workers=int(num_workers),
        timeout=float(timeout),
    )
    if isinstance(out, tuple):
        pass_at_k = dict(out[0])
        details = out[1] if len(out) > 1 else None
        return pass_at_k, details
    return dict(out), None


def compute_humaneval_pass_at_k(
    problems: List[Dict[str, Any]],
    completions_by_problem: List[List[str]],
    *,
    k: Sequence[int] = (1, 10, 100),
    num_workers: int = 4,
    timeout: float = 3.0,
) -> Dict[str, float]:
    if len(problems) != len(completions_by_problem):
        raise ValueError("problems and completions_by_problem must have the same length")
    if not problems:
        return {f"pass@{kk}": 0.0 for kk in k}

    references = [_build_reference(problem) for problem in problems]
    predictions = [
        [_build_prediction(problem, completion) for completion in completions]
        for problem, completions in zip(problems, completions_by_problem)
    ]

    pass_at_k, _ = _run_code_eval(
        references,
        predictions,
        k=k,
        num_workers=num_workers,
        timeout=timeout,
    )
    return {f"pass@{kk}": float(pass_at_k.get(f"pass@{kk}", 0.0)) for kk in k}


def compute_humaneval_pass_flags(
    problems: List[Dict[str, Any]],
    completions: List[str],
    *,
    num_workers: int = 4,
    timeout: float = 3.0,
) -> List[int]:
    if len(problems) != len(completions):
        raise ValueError("problems and completions must have the same length")
    if not problems:
        return []

    references = [_build_reference(problem) for problem in problems]
    predictions = [[_build_prediction(problem, completion)] for problem, completion in zip(problems, completions)]

    _, details = _run_code_eval(
        references,
        predictions,
        k=(1,),
        num_workers=num_workers,
        timeout=timeout,
    )

    if isinstance(details, dict):
        flags: List[int] = []
        parse_failed = False
        for idx in range(len(problems)):
            rec = details.get(idx)
            if rec is None:
                rec = details.get(str(idx))
            if not isinstance(rec, list) or not rec:
                parse_failed = True
                break
            result_tuple = rec[0]
            if not isinstance(result_tuple, (list, tuple)) or len(result_tuple) < 2:
                parse_failed = True
                break
            result_payload = result_tuple[1]
            if not isinstance(result_payload, dict):
                parse_failed = True
                break
            flags.append(int(bool(result_payload.get("passed", False))))
        if not parse_failed and len(flags) == len(problems):
            return flags

    # Fallback path: compute per-sample pass@1.
    flags = []
    for problem, completion in zip(problems, completions):
        one_ref = [_build_reference(problem)]
        one_pred = [[_build_prediction(problem, completion)]]
        pass_at_k, _ = _run_code_eval(
            one_ref,
            one_pred,
            k=(1,),
            num_workers=num_workers,
            timeout=timeout,
        )
        flags.append(int(float(pass_at_k.get("pass@1", 0.0)) >= 0.5))
    return flags

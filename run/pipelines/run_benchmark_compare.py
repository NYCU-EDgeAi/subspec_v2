"""Paired benchmark comparison pipeline (flips + KL/task-native metrics)."""

import os
import shutil
import json
import time
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel

from .benchmarks.registry import load_dataset, validate_benchmarks
from .benchmarks.task_registry import (
    LANE_BEHAVIOR,
    LANE_DISTRIBUTION,
    normalize_lane,
    validate_lane_compatibility,
)
from .benchmarks.utils.eval_acc import (
    _encode_context,
    _get_forward_fn,
    _loglikelihood_continuation,
    _maybe_delimit,
)
from .benchmarks.utils.code_eval import (
    compute_humaneval_pass_flags,
    validate_humaneval_runtime_requirements,
)
from .utils.benchmark_utils import cleanup_gpu, parse_benchmark_list, reset_seeds, setup_benchmark_dir
from run.core.config_utils import write_settings_yaml
from run.pipelines.utils.eval_utils import reset_kv


MC_BENCHMARKS = {"hellaswag", "piqa", "arc-c", "winogrande"}
TASK_NATIVE_ACCURACY_BENCHMARKS = {"gsm8k", "human-eval", "human-eval-instruct"}
BENCHMARKS_WITH_ANSWERS = MC_BENCHMARKS | TASK_NATIVE_ACCURACY_BENCHMARKS
SUPPORTED_COMPARE_BENCHMARKS = BENCHMARKS_WITH_ANSWERS
_PROMPT_BUDGET_WARNED: set[tuple[str, int, int]] = set()


def _append_jsonl(path: str, record: Dict[str, Any], indent: int | None = None) -> None:
    with open(path, "a+") as f:
        json.dump(record, f, indent=indent)
        f.write("\n")


def _load_indexed_jsonl(path: str) -> Dict[int, Dict[str, Any]]:
    indexed: Dict[int, Dict[str, Any]] = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            indexed[int(rec["index"])] = rec
    return indexed


def _build_flip_metrics(
    samples: int,
    flips_c2i_count: int,
    flips_i2c_count: int,
    *,
    allflips_count: int | None = None,
    output_change_count: int | None = None,
) -> Dict[str, Any]:
    flips_total = int(flips_c2i_count) + int(flips_i2c_count)
    flips_rate = float(flips_total / samples) if samples else 0.0
    flips_c2i_rate = float(flips_c2i_count / samples) if samples else 0.0
    flips_i2c_rate = float(flips_i2c_count / samples) if samples else 0.0

    allflips_rate = float(allflips_count / samples) if (allflips_count is not None and samples) else None
    output_change_rate = (
        float(output_change_count / samples) if (output_change_count is not None and samples) else None
    )
    wrong_to_wrong_change_rate = (
        max(allflips_rate - flips_rate, 0.0) if allflips_rate is not None else None
    )

    return {
        "flips_rate": flips_rate,
        "flips_c2i_count": int(flips_c2i_count),
        "flips_i2c_count": int(flips_i2c_count),
        "flips_c2i_rate": flips_c2i_rate,
        "flips_i2c_rate": flips_i2c_rate,
        "allflips_rate": allflips_rate,
        "output_change_rate": output_change_rate,
        "wrong_to_wrong_change_rate": wrong_to_wrong_change_rate,
        "flip_metric_schema": "v3_paper_aligned",
    }


def _warn_if_prompt_near_budget(task_name: str, prompt_len: int, max_length: int, max_gen_toks: int) -> None:
    """Warn once per task when prompt length leaves less than max_gen_toks headroom."""
    if max_length is None:
        return
    threshold = max_length - max_gen_toks
    if prompt_len <= threshold:
        return
    key = (task_name, max_length, max_gen_toks)
    if key in _PROMPT_BUDGET_WARNED:
        return
    _PROMPT_BUDGET_WARNED.add(key)
    logging.warning(
        "%s prompt length (%d) exceeds max_length - max_gen_toks (%d - %d = %d). "
        "Generation headroom is smaller than lm-eval-style budget.",
        task_name,
        prompt_len,
        max_length,
        max_gen_toks,
        threshold,
    )


def _load_benchmark_dataset(
    bench_name: str,
    max_samples: int | None,
    seed: int,
    shuffle: bool,
) -> List[Any]:
    return load_dataset(
        bench_name,
        max_samples=max_samples,
        seed=seed,
        shuffle=shuffle,
        with_answers=bench_name in BENCHMARKS_WITH_ANSWERS,
    )


def _validate_compare_benchmarks(bench_list: List[str]) -> None:
    unsupported = [bench for bench in bench_list if bench not in SUPPORTED_COMPARE_BENCHMARKS]
    if unsupported:
        raise ValueError(
            "run-benchmark-compare currently supports only: "
            f"{sorted(SUPPORTED_COMPARE_BENCHMARKS)}. "
            f"Unsupported benchmark(s): {unsupported}"
        )


def _softmax_from_lls(lls: List[float]) -> np.ndarray:
    arr = np.array(lls, dtype=np.float64)
    if arr.size == 0:
        return arr
    max_val = np.max(arr)
    exp = np.exp(arr - max_val)
    denom = np.sum(exp)
    if denom == 0:
        return np.full_like(exp, 1.0 / len(exp))
    return exp / denom


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    if p.size == 0 or q.size == 0:
        return 0.0
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _choice_lls(
    model,
    tokenizer,
    context: str,
    choices: List[str],
    target_delimiter: str,
    max_length: int,
    use_chat_template: bool = False,
) -> Tuple[List[float], List[int]]:
    prefix_ids = _encode_context(tokenizer, context, model.device, use_chat_template)
    if prefix_ids.shape[1] > max_length:
        return [], []
    lls: List[float] = []
    cont_lens: List[int] = []
    for choice in choices:
        cont = _maybe_delimit(context, choice, target_delimiter)
        ll, cont_len = _loglikelihood_continuation(
            model,
            tokenizer,
            prefix_ids,
            cont,
            max_length=max_length,
            return_len=True,
        )
        lls.append(ll)
        cont_lens.append(max(int(cont_len), 1))
    return lls, cont_lens


def _choice_lls_multiple_input(
    model,
    tokenizer,
    contexts: List[str],
    target: str,
    target_delimiter: str,
    max_length: int,
    use_chat_template: bool = False,
) -> Tuple[List[float], List[int]]:
    lls: List[float] = []
    cont_lens: List[int] = []
    for ctx in contexts:
        prefix_ids = _encode_context(tokenizer, ctx, model.device, use_chat_template)
        if prefix_ids.shape[1] > max_length:
            lls.append(float("-inf"))
            cont_lens.append(1)
            continue
        cont = _maybe_delimit(ctx, target, target_delimiter)
        ll, cont_len = _loglikelihood_continuation(
            model,
            tokenizer,
            prefix_ids,
            cont,
            max_length=max_length,
            return_len=True,
        )
        lls.append(ll)
        cont_lens.append(max(int(cont_len), 1))
    return lls, cont_lens


def _token_logprobs_for_continuation(
    model,
    tokenizer,
    context: str,
    continuation: str,
    target_delimiter: str,
    max_length: int,
    use_chat_template: bool = False,
) -> torch.Tensor | None:
    device = next(model.parameters()).device
    prefix_ids = _encode_context(tokenizer, context, device, use_chat_template)
    cont_text = _maybe_delimit(context, continuation, target_delimiter)
    cont_ids = tokenizer(
        cont_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    cont_len = cont_ids.shape[1]
    if cont_len == 0:
        return None
    if max_length is not None and (prefix_ids.shape[1] + cont_len) > max_length:
        return None

    cont_start = prefix_ids.shape[1]
    if cont_start == 0:
        return None

    input_ids = torch.cat([prefix_ids, cont_ids], dim=1)
    forward_fn = _get_forward_fn(model)
    with torch.inference_mode():
        logits = forward_fn(input_ids).logits

    pos = torch.arange(cont_start - 1, cont_start - 1 + cont_len, device=device)
    log_probs = torch.log_softmax(logits[0, pos, :], dim=-1)
    return log_probs.detach().cpu()


def _kl_from_logprobs(base_log_probs: torch.Tensor, compare_log_probs: torch.Tensor) -> float | None:
    if base_log_probs.shape != compare_log_probs.shape:
        return None
    base_log_probs = base_log_probs.float()
    compare_log_probs = compare_log_probs.float()
    base_probs = torch.exp(base_log_probs)
    kl_vals = (base_probs * (base_log_probs - compare_log_probs)).sum(dim=-1)
    return float(kl_vals.mean().item())


def _build_compare_builder(base_builder, compare_config_path: str):
    from run.main import _build_settings_snapshot, _load_yaml_config, _resolve_method, _apply_yaml_overrides
    from run.core.configuration import AppConfig
    from run.core.registry import ModelRegistry
    from run.core.builder import GeneratorPipelineBuilder
    from run.core.config_utils import instantiate_recipe

    yaml_config = _load_yaml_config(compare_config_path)
    method = _resolve_method(None, yaml_config)
    method_entry = ModelRegistry.get(method)
    if method_entry is None:
        raise ValueError(f"Unknown method in compare config: {method}")

    default_config = method_entry.default_config.copy()
    default_config = _apply_yaml_overrides(default_config, yaml_config)

    config = AppConfig()
    config.method = method
    config.update(default_config)

    # Align key generation settings for fair comparison.
    for key in [
        "max_length",
        "seed",
        "temperature",
        "do_sample",
        "warmup_iter",
        "device",
        "cache_implementation",
        "compile_mode",
        "generator_profiling",
        "profiling_verbose",
    ]:
        if hasattr(base_builder, key):
            setattr(config, key, getattr(base_builder, key))

    # If compare YAML omitted model paths, inherit from base builder.
    if not getattr(config, "llm_path", None):
        config.llm_path = getattr(base_builder, "llm_path", None)
    if not getattr(config, "draft_model_path", None):
        config.draft_model_path = getattr(base_builder, "draft_model_path", None)

    config.recipe = instantiate_recipe(getattr(config, "recipe", None))
    config.config_path = compare_config_path
    config.settings_snapshot = _build_settings_snapshot(
        config=config,
        config_path=compare_config_path,
        subcommand_argv=["run-benchmark-compare", "--compare-config", compare_config_path],
    )

    return GeneratorPipelineBuilder(config)


def _extract_gsm8k_answer(text: str) -> str | None:
    # Match lm-eval GSM8K flexible extraction behavior.
    from run.pipelines.benchmarks.gsm8k import (
        INVALID_ANS as GSM8K_INVALID_ANS,
        extract_answer as gsm8k_extract_answer,
    )

    answer = gsm8k_extract_answer(text)
    if answer == GSM8K_INVALID_ANS:
        return None
    return answer


def _gsm8k_exact_match(candidate: str | None, reference: str | None) -> bool:
    from run.pipelines.benchmarks.gsm8k import exact_match as gsm8k_exact_match

    if candidate is None or reference is None:
        return False
    return gsm8k_exact_match(candidate, reference)


def _generate_gsm8k_response(
    generator,
    tokenizer,
    args,
    prompt: str,
    past_key_values,
    draft_past_key_values,
) -> str | None:
    from run.pipelines.benchmarks.gsm8k import MAX_GEN_TOKS as GSM8K_MAX_GEN_TOKS
    from run.pipelines.benchmarks.gsm8k import STOP_STRINGS as GSM8K_STOP_STRINGS

    tokenizer.use_default_system_prompt = True
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(generator.device)
    if input_ids.shape[1] > args.max_length:
        return None
    _warn_if_prompt_near_budget("gsm8k", int(input_ids.shape[1]), int(args.max_length), int(GSM8K_MAX_GEN_TOKS))

    with sdpa_kernel(backends=[SDPBackend.MATH]):
        output_ids = generator.generate(
            input_ids,
            temperature=args.temperature,
            max_length=args.max_length,
            do_sample=args.do_sample,
            stop_strings=GSM8K_STOP_STRINGS,
            past_key_values=past_key_values,
            draft_past_key_values=draft_past_key_values,
        )
    reset_kv(past_key_values, draft_past_key_values)

    return tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()


def _run_gsm8k_baseline(
    base_generator,
    base_tokenizer,
    base_args,
    dataset: List[Dict[str, Any]],
    log_dir: str,
    bench_name: str,
    past_key_values,
    draft_past_key_values,
):
    base_log = os.path.join(log_dir, f"{bench_name}_base.jsonl")

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc=f"{bench_name} baseline"):
        prompt = entry["question"]
        gt = _extract_gsm8k_answer(str(entry["answer"]))

        response = _generate_gsm8k_response(
            base_generator,
            base_tokenizer,
            base_args,
            prompt,
            past_key_values,
            draft_past_key_values,
        )
        if response is None:
            continue

        pred = _extract_gsm8k_answer(response)
        correct = int(_gsm8k_exact_match(pred, gt))

        _append_jsonl(
            base_log,
            {
                "index": idx,
                "pred": pred,
                "answer": gt,
                "correct": correct,
            },
        )


def _run_gsm8k_compare(
    compare_generator,
    compare_tokenizer,
    compare_args,
    dataset: List[Dict[str, Any]],
    base_log: str,
    log_dir: str,
    bench_name: str,
    past_key_values,
    draft_past_key_values,
):
    pair_log = os.path.join(log_dir, f"{bench_name}_compare.jsonl")
    base_records = _load_indexed_jsonl(base_log)

    total = 0
    base_correct = 0
    compare_correct = 0
    flips_pos = 0
    flips_neg = 0
    allflips_count = 0

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc=f"{bench_name} compare"):
        if idx not in base_records:
            continue

        prompt = entry["question"]
        gt = _extract_gsm8k_answer(str(entry["answer"]))
        response = _generate_gsm8k_response(
            compare_generator,
            compare_tokenizer,
            compare_args,
            prompt,
            past_key_values,
            draft_past_key_values,
        )
        if response is None:
            continue

        pred_cmp = _extract_gsm8k_answer(response)
        correct_cmp = int(_gsm8k_exact_match(pred_cmp, gt))

        base_rec = base_records[idx]
        pred_base = base_rec.get("pred")
        correct_base = int(base_rec.get("correct", 0))

        if pred_cmp != pred_base:
            allflips_count += 1
        if correct_base and not correct_cmp:
            flips_pos += 1
        if (not correct_base) and correct_cmp:
            flips_neg += 1

        total += 1
        base_correct += int(correct_base)
        compare_correct += int(correct_cmp)

        _append_jsonl(
            pair_log,
            {
                "index": idx,
                "answer": gt,
                "base": {
                    "pred": pred_base,
                    "correct": int(correct_base),
                },
                "compare": {
                    "pred": pred_cmp,
                    "correct": int(correct_cmp),
                },
                "pred_changed": int(pred_cmp != pred_base),
                "flips_c2i": int(correct_base and not correct_cmp),
                "flips_i2c": int((not correct_base) and correct_cmp),
            },
        )

    return {
        "base_accuracy": float(base_correct / total) if total else 0.0,
        "compare_accuracy": float(compare_correct / total) if total else 0.0,
        "kl_choice": None,
        "kl_token": None,
        "samples": int(total),
        **_build_flip_metrics(
            int(total),
            int(flips_pos),
            int(flips_neg),
            allflips_count=int(allflips_count),
        ),
    }


def _generate_humaneval_completion(
    generator,
    tokenizer,
    args,
    prompt: str,
    past_key_values,
    draft_past_key_values,
    task_name: str = "human-eval",
) -> str | None:
    from run.pipelines.benchmarks.humaneval import MAX_GEN_TOKS as HUMANEVAL_MAX_GEN_TOKS
    from run.pipelines.benchmarks.humaneval import STOP_STRINGS as HUMANEVAL_STOP_STRINGS

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(generator.device)
    if input_ids.shape[1] > args.max_length:
        return None
    _warn_if_prompt_near_budget(
        task_name,
        int(input_ids.shape[1]),
        int(args.max_length),
        int(HUMANEVAL_MAX_GEN_TOKS),
    )

    with sdpa_kernel(backends=[SDPBackend.MATH]):
        output_ids = generator.generate(
            input_ids,
            temperature=args.temperature,
            max_length=args.max_length,
            do_sample=args.do_sample,
            stop_strings=HUMANEVAL_STOP_STRINGS,
            past_key_values=past_key_values,
            draft_past_key_values=draft_past_key_values,
        )
    reset_kv(past_key_values, draft_past_key_values)

    # Keep leading indentation for HumanEval completions.
    return tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )


def _run_humaneval_baseline(
    base_generator,
    base_tokenizer,
    base_args,
    dataset: List[Dict[str, Any]],
    log_dir: str,
    bench_name: str,
    past_key_values,
    draft_past_key_values,
):
    base_log = os.path.join(log_dir, f"{bench_name}_base.jsonl")
    timeout = float(getattr(base_args, "humaneval_timeout", 3.0))
    n_workers = int(getattr(base_args, "humaneval_workers", 4))
    pending: List[Tuple[int, Dict[str, Any], str]] = []

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc=f"{bench_name} baseline"):
        completion = _generate_humaneval_completion(
            base_generator,
            base_tokenizer,
            base_args,
            entry.get("prompt_input", entry["prompt"]),
            past_key_values,
            draft_past_key_values,
            task_name=bench_name,
        )
        if completion is None:
            continue
        pending.append((idx, entry, completion))

    if not pending:
        return

    problems = [
        {
            "prompt": entry["prompt"],
            "prediction_style": entry.get("prediction_style", "plain"),
            "test": entry["test"],
            "entry_point": entry["entry_point"],
        }
        for _, entry, _ in pending
    ]
    completions = [completion for _, _, completion in pending]
    correct_flags = compute_humaneval_pass_flags(
        problems,
        completions,
        num_workers=n_workers,
        timeout=timeout,
    )

    for (idx, entry, completion), correct in zip(pending, correct_flags):
        _append_jsonl(
            base_log,
            {
                "index": idx,
                "task_id": entry["task_id"],
                "completion": completion,
                "correct": int(correct),
            },
        )


def _run_humaneval_compare(
    compare_generator,
    compare_tokenizer,
    compare_args,
    dataset: List[Dict[str, Any]],
    base_log: str,
    log_dir: str,
    bench_name: str,
    past_key_values,
    draft_past_key_values,
):
    pair_log = os.path.join(log_dir, f"{bench_name}_compare.jsonl")
    base_records = _load_indexed_jsonl(base_log)
    timeout = float(getattr(compare_args, "humaneval_timeout", 3.0))
    n_workers = int(getattr(compare_args, "humaneval_workers", 4))

    total = 0
    base_correct = 0
    compare_correct = 0
    flips_pos = 0
    flips_neg = 0
    output_change_count = 0
    pending: List[Tuple[int, Dict[str, Any], Dict[str, Any], str]] = []

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc=f"{bench_name} compare"):
        if idx not in base_records:
            continue

        completion_cmp = _generate_humaneval_completion(
            compare_generator,
            compare_tokenizer,
            compare_args,
            entry.get("prompt_input", entry["prompt"]),
            past_key_values,
            draft_past_key_values,
            task_name=bench_name,
        )
        if completion_cmp is None:
            continue
        pending.append((idx, entry, base_records[idx], completion_cmp))

    if not pending:
        return {
            "base_accuracy": 0.0,
            "compare_accuracy": 0.0,
            "kl_choice": None,
            "kl_token": None,
            "samples": 0,
            **_build_flip_metrics(0, 0, 0, output_change_count=0),
        }

    problems = [
        {
            "prompt": entry["prompt"],
            "prediction_style": entry.get("prediction_style", "plain"),
            "test": entry["test"],
            "entry_point": entry["entry_point"],
        }
        for _, entry, _, _ in pending
    ]
    completions = [completion_cmp for _, _, _, completion_cmp in pending]
    correct_flags = compute_humaneval_pass_flags(
        problems,
        completions,
        num_workers=n_workers,
        timeout=timeout,
    )

    for (idx, entry, base_rec, completion_cmp), correct_cmp in zip(pending, correct_flags):
        correct_cmp = int(correct_cmp)
        completion_base = base_rec.get("completion", "")
        correct_base = int(base_rec.get("correct", 0))
        output_changed = int(completion_cmp != str(completion_base))

        if output_changed:
            output_change_count += 1
        if correct_base and not correct_cmp:
            flips_pos += 1
        if (not correct_base) and correct_cmp:
            flips_neg += 1

        total += 1
        base_correct += int(correct_base)
        compare_correct += int(correct_cmp)

        _append_jsonl(
            pair_log,
            {
                "index": idx,
                "task_id": entry["task_id"],
                "base": {
                    "correct": int(correct_base),
                },
                "compare": {
                    "correct": int(correct_cmp),
                },
                "output_changed": output_changed,
                "flips_c2i": int(correct_base and not correct_cmp),
                "flips_i2c": int((not correct_base) and correct_cmp),
            },
        )

    return {
        "base_accuracy": float(base_correct / total) if total else 0.0,
        "compare_accuracy": float(compare_correct / total) if total else 0.0,
        "kl_choice": None,
        "kl_token": None,
        "samples": int(total),
        **_build_flip_metrics(
            int(total),
            int(flips_pos),
            int(flips_neg),
            output_change_count=int(output_change_count),
        ),
    }


def _run_multichoice_baseline(
    base_generator,
    base_tokenizer,
    base_args,
    dataset: List[Dict[str, Any]],
    log_dir: str,
    bench_name: str,
    token_kl_cache: Dict[str, Dict[int, List[torch.Tensor | None]]] | None,
):
    base_log = os.path.join(log_dir, f"{bench_name}_base.jsonl")

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc=f"{bench_name} baseline"):
        answer_index = int(entry.get("answer_index", -1))
        target_delimiter = entry.get("target_delimiter", " ")
        use_chat_template = bool(entry.get("use_chat_template", False))

        if entry.get("multiple_input", False):
            lls_base, _ = _choice_lls_multiple_input(
                base_generator.target_model,
                base_tokenizer,
                entry["contexts"],
                entry["target"],
                target_delimiter,
                base_args.max_length,
                use_chat_template,
            )
        else:
            lls_base, _ = _choice_lls(
                base_generator.target_model,
                base_tokenizer,
                entry["context"],
                entry["choices"],
                target_delimiter,
                base_args.max_length,
                use_chat_template,
            )

        if not lls_base or answer_index < 0:
            continue

        pred_base = int(np.argmax(np.array(lls_base)))
        correct_base = int(pred_base == answer_index)

        record = {
            "index": idx,
            "lls": lls_base,
            "pred": pred_base,
            "answer_index": answer_index,
            "correct": correct_base,
        }
        _append_jsonl(base_log, record)

        if token_kl_cache is not None:
            token_logs: List[torch.Tensor | None] = []
            if entry.get("multiple_input", False):
                for ctx in entry["contexts"]:
                    token_logs.append(
                        _token_logprobs_for_continuation(
                            base_generator.target_model,
                            base_tokenizer,
                            ctx,
                            entry["target"],
                            target_delimiter,
                            base_args.max_length,
                            use_chat_template,
                        )
                    )
            else:
                for choice in entry["choices"]:
                    token_logs.append(
                        _token_logprobs_for_continuation(
                            base_generator.target_model,
                            base_tokenizer,
                            entry["context"],
                            choice,
                            target_delimiter,
                            base_args.max_length,
                            use_chat_template,
                        )
                    )

            if any(log is not None for log in token_logs):
                token_kl_cache.setdefault(bench_name, {})[idx] = token_logs


def _run_multichoice_compare(
    compare_generator,
    compare_tokenizer,
    compare_args,
    dataset: List[Dict[str, Any]],
    base_log: str,
    log_dir: str,
    bench_name: str,
    token_kl_cache: Dict[str, Dict[int, List[torch.Tensor | None]]] | None,
    token_kl: bool,
    token_kl_tokenizer,
):
    pair_log = os.path.join(log_dir, f"{bench_name}_compare.jsonl")

    base_records = _load_indexed_jsonl(base_log)

    total = 0
    base_correct = 0
    compare_correct = 0
    flips_pos = 0
    flips_neg = 0
    allflips_count = 0
    kl_choice_sum = 0.0
    kl_token_sum = 0.0
    kl_token_count = 0

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc=f"{bench_name} compare"):
        if idx not in base_records:
            continue

        answer_index = int(entry.get("answer_index", -1))
        target_delimiter = entry.get("target_delimiter", " ")
        use_chat_template = bool(entry.get("use_chat_template", False))

        if entry.get("multiple_input", False):
            lls_cmp, _ = _choice_lls_multiple_input(
                compare_generator.target_model,
                compare_tokenizer,
                entry["contexts"],
                entry["target"],
                target_delimiter,
                compare_args.max_length,
                use_chat_template,
            )
        else:
            lls_cmp, _ = _choice_lls(
                compare_generator.target_model,
                compare_tokenizer,
                entry["context"],
                entry["choices"],
                target_delimiter,
                compare_args.max_length,
                use_chat_template,
            )

        if not lls_cmp or answer_index < 0:
            continue

        base_rec = base_records[idx]
        pred_base = int(base_rec["pred"])
        correct_base = int(base_rec["correct"])

        pred_cmp = int(np.argmax(np.array(lls_cmp)))
        correct_cmp = int(pred_cmp == answer_index)

        if pred_cmp != pred_base:
            allflips_count += 1
        if correct_base and not correct_cmp:
            flips_pos += 1
        if (not correct_base) and correct_cmp:
            flips_neg += 1

        p_base = _softmax_from_lls(base_rec["lls"])
        p_cmp = _softmax_from_lls(lls_cmp)
        kl_choice = _kl_divergence(p_base, p_cmp)
        kl_choice_sum += kl_choice

        token_kl_val = None
        if token_kl and token_kl_cache is not None:
            base_logs = token_kl_cache.get(bench_name, {}).get(idx)
            if base_logs:
                token_kls: List[float] = []
                if entry.get("multiple_input", False):
                    for i, ctx in enumerate(entry["contexts"]):
                        base_lp = base_logs[i] if i < len(base_logs) else None
                        if base_lp is None:
                            continue
                        cmp_lp = _token_logprobs_for_continuation(
                            compare_generator.target_model,
                            token_kl_tokenizer,
                            ctx,
                            entry["target"],
                            target_delimiter,
                            compare_args.max_length,
                            use_chat_template,
                        )
                        if cmp_lp is None:
                            continue
                        kl_val = _kl_from_logprobs(base_lp, cmp_lp)
                        if kl_val is not None:
                            token_kls.append(kl_val)
                else:
                    for i, choice in enumerate(entry["choices"]):
                        base_lp = base_logs[i] if i < len(base_logs) else None
                        if base_lp is None:
                            continue
                        cmp_lp = _token_logprobs_for_continuation(
                            compare_generator.target_model,
                            token_kl_tokenizer,
                            entry["context"],
                            choice,
                            target_delimiter,
                            compare_args.max_length,
                            use_chat_template,
                        )
                        if cmp_lp is None:
                            continue
                        kl_val = _kl_from_logprobs(base_lp, cmp_lp)
                        if kl_val is not None:
                            token_kls.append(kl_val)

                if token_kls:
                    token_kl_val = float(np.mean(token_kls))
                    kl_token_sum += token_kl_val
                    kl_token_count += 1

        total += 1
        base_correct += int(correct_base)
        compare_correct += int(correct_cmp)

        pair_record = {
            "index": idx,
            "answer_index": answer_index,
            "base": {
                "pred": pred_base,
                "correct": int(correct_base),
                "lls": base_rec["lls"],
            },
            "compare": {
                "pred": pred_cmp,
                "correct": int(correct_cmp),
                "lls": lls_cmp,
            },
            "pred_changed": int(pred_cmp != pred_base),
            "flips_c2i": int(correct_base and not correct_cmp),
            "flips_i2c": int((not correct_base) and correct_cmp),
            "kl_choice": kl_choice,
            "kl_token": token_kl_val,
        }
        _append_jsonl(pair_log, pair_record)

    results = {
        "base_accuracy": float(base_correct / total) if total else 0.0,
        "compare_accuracy": float(compare_correct / total) if total else 0.0,
        "kl_choice": float(kl_choice_sum / total) if total else 0.0,
        "kl_token": float(kl_token_sum / kl_token_count) if kl_token_count else 0.0,
        "samples": int(total),
        **_build_flip_metrics(
            int(total),
            int(flips_pos),
            int(flips_neg),
            allflips_count=int(allflips_count),
        ),
    }
    return results


def _run_baseline_for_benchmark(
    bench_name: str,
    dataset: List[Any],
    log_dir: str,
    base_generator,
    base_tokenizer,
    base_args,
    base_past_kv,
    base_draft_kv,
    token_kl_cache,
):
    if bench_name in MC_BENCHMARKS:
        _run_multichoice_baseline(
            base_generator,
            base_tokenizer,
            base_args,
            dataset,
            log_dir,
            bench_name,
            token_kl_cache,
        )
    elif bench_name == "gsm8k":
        _run_gsm8k_baseline(
            base_generator,
            base_tokenizer,
            base_args,
            dataset,
            log_dir,
            bench_name,
            base_past_kv,
            base_draft_kv,
        )
    elif bench_name in {"human-eval", "human-eval-instruct"}:
        _run_humaneval_baseline(
            base_generator,
            base_tokenizer,
            base_args,
            dataset,
            log_dir,
            bench_name,
            base_past_kv,
            base_draft_kv,
        )
    else:
        raise ValueError(
            f"Unsupported benchmark '{bench_name}' for run-benchmark-compare. "
            f"Supported benchmarks: {sorted(SUPPORTED_COMPARE_BENCHMARKS)}"
        )


def _run_compare_for_benchmark(
    bench_name: str,
    dataset: List[Any],
    base_log: str,
    log_dir: str,
    compare_generator,
    compare_tokenizer,
    compare_args,
    compare_past_kv,
    compare_draft_kv,
    token_kl_cache,
    token_kl: bool,
    base_tokenizer,
):
    if bench_name in MC_BENCHMARKS:
        return _run_multichoice_compare(
            compare_generator,
            compare_tokenizer,
            compare_args,
            dataset,
            base_log,
            log_dir,
            bench_name,
            token_kl_cache,
            token_kl,
            base_tokenizer,
        )
    if bench_name == "gsm8k":
        return _run_gsm8k_compare(
            compare_generator,
            compare_tokenizer,
            compare_args,
            dataset,
            base_log,
            log_dir,
            bench_name,
            compare_past_kv,
            compare_draft_kv,
        )
    if bench_name in {"human-eval", "human-eval-instruct"}:
        return _run_humaneval_compare(
            compare_generator,
            compare_tokenizer,
            compare_args,
            dataset,
            base_log,
            log_dir,
            bench_name,
            compare_past_kv,
            compare_draft_kv,
        )
    raise ValueError(
        f"Unsupported benchmark '{bench_name}' for run-benchmark-compare. "
        f"Supported benchmarks: {sorted(SUPPORTED_COMPARE_BENCHMARKS)}"
    )


def main(
    builder,
    benchmarks: str = None,
    max_samples: int = None,
    compare_config: str = None,
    compare_name: str = "compare",
    seed: int = 0,
    shuffle: bool = True,
    token_kl: bool = True,
    lane: str = LANE_DISTRIBUTION,
    reuse_baseline_dir: str | None = None,
):
    if not compare_config:
        raise ValueError("--compare-config is required for run-benchmark-compare")

    reset_seeds(seed)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())

    bench_list = parse_benchmark_list(benchmarks)
    if not bench_list:
        raise ValueError("--benchmarks is required for run-benchmark-compare")
    validate_benchmarks(bench_list, with_answers=True)

    lane = normalize_lane(lane or LANE_DISTRIBUTION)
    validate_lane_compatibility(bench_list, lane)
    _validate_compare_benchmarks(bench_list)
    if "human-eval" in bench_list or "human-eval-instruct" in bench_list:
        validate_humaneval_runtime_requirements()

    if lane == LANE_BEHAVIOR:
        token_kl = False

    token_kl_cache: Dict[str, Dict[int, List[torch.Tensor | None]]] | None = {} if token_kl else None

    base_generator = base_tokenizer = base_past_kv = base_draft_kv = None
    base_args = builder.args

    default_log_dir_base = os.path.join(
        builder.args.log_dir,
        time.strftime("%Y%m%d-%H%M%S"),
        "run_benchmark_compare",
    )
    log_dir_base = default_log_dir_base
    bench_jobs: List[Tuple[str, str, str]] = []

    if reuse_baseline_dir:
        if not os.path.isdir(reuse_baseline_dir):
            raise ValueError(f"reuse_baseline_dir does not exist: {reuse_baseline_dir}")
        for bench_name in bench_list:
            base_log_dir = os.path.join(reuse_baseline_dir, bench_name)
            if not os.path.isdir(base_log_dir):
                raise ValueError(f"Missing baseline directory for {bench_name}: {base_log_dir}")
            log_dir = setup_benchmark_dir(log_dir_base, bench_name, getattr(builder.args, "settings_snapshot", None))
            bench_jobs.append((bench_name, log_dir, base_log_dir))
    else:
        builder.generator_profiling = True
        builder.profiling_verbose = False
        base_generator, base_tokenizer, base_past_kv, base_draft_kv = builder.build()

        for bench_name in tqdm(bench_list, desc="Baseline phase"):
            reset_seeds(seed)
            log_dir = setup_benchmark_dir(log_dir_base, bench_name, getattr(builder.args, "settings_snapshot", None))
            bench_jobs.append((bench_name, log_dir, log_dir))

            dataset = _load_benchmark_dataset(bench_name, max_samples, seed, shuffle)
            _run_baseline_for_benchmark(
                bench_name,
                dataset,
                log_dir,
                base_generator,
                base_tokenizer,
                base_args,
                base_past_kv,
                base_draft_kv,
                token_kl_cache,
            )

        del base_past_kv
        del base_draft_kv
        del base_generator
        cleanup_gpu()

    cleanup_gpu()
    compare_builder = _build_compare_builder(builder, compare_config)
    compare_builder.generator_profiling = True
    compare_builder.profiling_verbose = False
    compare_generator, compare_tokenizer, compare_past_kv, compare_draft_kv = compare_builder.build()
    compare_args = compare_builder.args

    if reuse_baseline_dir and token_kl:
        print("Warning: token-level KL requires in-memory baseline cache; disabling token KL.")
        token_kl = False
        token_kl_cache = None

    if lane == LANE_DISTRIBUTION and token_kl and token_kl_cache is not None:
        if (
            base_tokenizer.__class__ != compare_tokenizer.__class__
            or getattr(base_tokenizer, "vocab_size", None) != getattr(compare_tokenizer, "vocab_size", None)
        ):
            print("Warning: tokenizer mismatch; disabling token-level KL.")
            token_kl = False
            token_kl_cache = None

    for bench_name, log_dir, base_log_dir in tqdm(bench_jobs, desc="Compare phase"):
        reset_seeds(seed)
        # Preserve both snapshots: base config and compare config.
        write_settings_yaml(log_dir, getattr(builder.args, "settings_snapshot", None), filename="settings_base.yaml")
        write_settings_yaml(log_dir, getattr(compare_args, "settings_snapshot", None), filename="settings.yaml")
        write_settings_yaml(log_dir, getattr(compare_args, "settings_snapshot", None), filename="settings_compare.yaml")

        dataset = _load_benchmark_dataset(bench_name, max_samples, seed, shuffle)

        base_log = os.path.join(base_log_dir, f"{bench_name}_base.jsonl")
        if reuse_baseline_dir:
            local_base_log = os.path.join(log_dir, f"{bench_name}_base.jsonl")
            if not os.path.exists(local_base_log) and os.path.exists(base_log):
                shutil.copy2(base_log, local_base_log)
        metrics = _run_compare_for_benchmark(
            bench_name,
            dataset,
            base_log,
            log_dir,
            compare_generator,
            compare_tokenizer,
            compare_args,
            compare_past_kv,
            compare_draft_kv,
            token_kl_cache,
            token_kl,
            base_tokenizer,
        )

        metrics = {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}
        metrics["compare_name"] = compare_name
        metrics["compare_config"] = compare_config
        metrics["token_kl"] = bool(token_kl)
        metrics["lane"] = lane
        _append_jsonl(os.path.join(log_dir, "results.jsonl"), {bench_name: metrics}, indent=4)

    cleanup_gpu()

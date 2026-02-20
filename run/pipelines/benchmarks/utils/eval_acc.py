"""Benchmark evaluation helpers (accuracy + perf metrics)."""

import base64
import gc
import json
import os
import pickle
import random
import re
import subprocess
import tempfile
import time
import zlib
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from .utils import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)
from specdecodes.models.utils.wandb_logger import wandb_logger
from run.pipelines.utils.eval_utils import reset_kv, maybe_init_cuda_graph_runner
from .human_eval.evaluation import evaluate_functional_correctness
from .human_eval.data import write_jsonl
from .lcb_runner.evaluation.compute_code_generation_metrics import check_correctness as lcb_check_correctness
from .lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results as lcb_compute_metrics
from .lcb_runner.utils.extraction_utils import extract_code as lcb_extract_code
from .lcb_runner.lm_styles import LMStyle

DATASET_TO_METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench_p": code_sim_score,
}
dataset2metric = DATASET_TO_METRIC


# ---- Perf tracking -------------------------------------------------------

def _run_warmup(
    generator,
    tokenizer,
    past_key_values,
    draft_past_key_values,
    args,
    warmup_prompt,
    max_length=None,
    max_new_tokens=None,
    warmup_iter=None,
    show_progress=False,
):
    original_profiling = generator.profiling
    generator.profiling = False
    n_iter = args.warmup_iter if warmup_iter is None else warmup_iter
    if n_iter <= 0:
        generator.profiling = original_profiling
        return

    iterator = tqdm(range(n_iter), desc="Warming up") if show_progress else range(n_iter)
    for _ in iterator:
        tokenizer.use_default_system_prompt = True
        warmup_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": warmup_prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)

        gen_kwargs = dict(
            temperature=args.temperature,
            do_sample=args.do_sample,
            past_key_values=past_key_values,
            draft_past_key_values=draft_past_key_values,
        )
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        else:
            gen_kwargs["max_length"] = max_length if max_length is not None else args.max_length

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            generator.generate(warmup_ids, **gen_kwargs)

        reset_kv(past_key_values, draft_past_key_values)

    generator.profiling = original_profiling


def _init_perf():
    return {
        "tput_list": [],
        "tacc_list": [],
        "total_iter": 0,
        "total_draft_time": 0.0,
        "total_target_time": 0.0,
    }


def _accum_perf(perf, record):
    perf["tput_list"].append(record["tput"])
    perf["tacc_list"].append(record["avg_sampled"])
    n_iter = record["n_iter"]
    perf["total_iter"] += n_iter
    perf["total_draft_time"] += record["avg_draft_time"] * n_iter
    perf["total_target_time"] += record["avg_target_time"] * n_iter


def _finalize_perf(perf, generator):
    tput_list = perf["tput_list"]
    tacc_list = perf["tacc_list"]
    total_iter = perf["total_iter"]
    total_draft_time = perf["total_draft_time"]
    total_target_time = perf["total_target_time"]

    tput_mean, tput_std = (np.mean(tput_list), np.std(tput_list)) if tput_list else (0, 0)
    tacc_mean, tacc_std = (np.mean(tacc_list), np.std(tacc_list)) if tacc_list else (0, 0)
    avg_draft_time = (total_draft_time / total_iter) if total_iter > 0 else 0
    avg_target_time = (total_target_time / total_iter) if total_iter > 0 else 0
    peak_memory = torch.cuda.max_memory_reserved(generator.device) / (1024 ** 3)

    return {
        "tput_mean": float(tput_mean),
        "tput_std": float(tput_std),
        "tacc_mean": float(tacc_mean),
        "tacc_std": float(tacc_std),
        "avg_draft_time": float(avg_draft_time),
        "avg_target_time": float(avg_target_time),
        "peak_memory_gib": float(peak_memory),
    }


def _print_summary(
    title,
    perf_stats,
    accuracy=None,
    correct_q=None,
    total_q=None,
    accuracy_norm=None,
    draft_time_note: str | None = None,
):
    print(f"Final {title} Results:")
    print(f"\tThroughput       : {perf_stats['tput_mean']:.3f} ± {perf_stats['tput_std']:.3f} tokens/sec")
    print(f"\tToken Acceptance : {perf_stats['tacc_mean']:.3f} ± {perf_stats['tacc_std']:.3f}")
    if accuracy is not None:
        if correct_q is not None and total_q is not None:
            print(f"\tAnswer Accuracy  : {accuracy:.3f} ({correct_q}/{total_q})")
        else:
            print(f"\tAnswer Accuracy  : {accuracy:.3f}")
    if accuracy_norm is not None:
        if correct_q is not None and total_q is not None:
            print(f"\tAnswer Acc (norm): {accuracy_norm:.3f} ({correct_q}/{total_q})")
        else:
            print(f"\tAnswer Acc (norm): {accuracy_norm:.3f}")
    if draft_time_note is not None:
        print(f"\tAvg Draft Time   : {draft_time_note}")
    else:
        print(f"\tAvg Draft Time   : {perf_stats['avg_draft_time']:.3f} sec")
    print(f"\tAvg Target Time  : {perf_stats['avg_target_time']:.3f} sec")
    print(f"\tPeak Memory      : {perf_stats['peak_memory_gib']:.3f} GiB")

# ---- GSM8K ---------------------------------------------------------------
def run_gsm8k_eval(generator, tokenizer, past_key_values, draft_past_key_values, args, dataset, log_dir):
    """
    Evaluate GSM8K dataset accuracy alongside performance metrics.

    Args:
        generator: the model generator instance
        tokenizer: tokenizer with chat template functionality
        past_key_values: primary past key values for autoregressive generation
        draft_past_key_values: draft past key values for speculative decoding (optional)
        args: namespace containing temperature, max_length, do_sample, warmup_iter
        dataset: list of dicts, each with keys:
            "question": the prompt string
            "answer": full original GSM8K answer text (contains "#### <answer>")
        log_dir: directory path for writing per-sample JSONL logs

    Returns:
        A tuple of metrics:
        (tput_mean, tput_std, tacc_mean, tacc_std,
         answer_accuracy, avg_draft_time, avg_target_time, peak_memory)
    """

    warmup_prompt = "Solve this math problem. Give the reasoning steps ...\nWhat is 1 + 1?"
    _run_warmup(
        generator,
        tokenizer,
        past_key_values,
        draft_past_key_values,
        args,
        warmup_prompt,
        max_length=args.max_length,
    )

    # 2. Main evaluation loop
    log_file = os.path.join(log_dir, "0.jsonl")

    # Lists to accumulate throughput, token acceptance, draft/target times
    perf = _init_perf()

    # Counters for overall question accuracy
    total_q = 0
    correct_q = 0

    # Match the official grade-school-math extraction behavior.
    from run.pipelines.benchmarks.gsm8k import (
        INVALID_ANS as GSM8K_INVALID_ANS,
        extract_answer as gsm8k_extract_answer,
    )

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating GSM8K"):
        prompt = entry["question"]
        ground_truth_text = entry["answer"]  # includes "Answer: N"

        # 2.1 Generate model output IDs (same as original)
        tokenizer.use_default_system_prompt = True
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)

        if input_ids.shape[1] > args.max_length:
            # Skip prompts that exceed max_length
            continue

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(
                input_ids,
                temperature=args.temperature,
                max_length=args.max_length,
                do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values
            )

        reset_kv(past_key_values, draft_past_key_values)

        # 2.2 Extract original performance logs
        record = {**wandb_logger.log_data}
        output_str = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        record.update({
            "query": prompt,
            "response": output_str,
            "answer": ground_truth_text.strip(),
            "peak_memory": torch.cuda.max_memory_reserved(generator.device) / (1024 ** 3)
        })

        # 2.3 Compute per-sample correctness
        pred_answer = gsm8k_extract_answer(output_str)
        gt_answer = gsm8k_extract_answer(ground_truth_text)
        is_correct = (
            pred_answer != GSM8K_INVALID_ANS
            and gt_answer != GSM8K_INVALID_ANS
            and pred_answer == gt_answer
        )
        total_q += 1
        if is_correct:
            correct_q += 1

        # Include per-sample Accuracy flag in JSON record
        record["Accuracy"] = int(is_correct)
        record["pred_answer"] = None if pred_answer == GSM8K_INVALID_ANS else pred_answer
        record["answer_extracted"] = None if gt_answer == GSM8K_INVALID_ANS else gt_answer

        # Append metrics lists
        _accum_perf(perf, record)

        # Write JSONL entry
        with open(log_file, "a+") as f:
            json.dump(record, f)
            f.write("\n")

        # Clean up
        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Aggregate overall metrics
    answer_accuracy = correct_q / total_q if total_q > 0 else 0
    perf_stats = _finalize_perf(perf, generator)
    _print_summary("GSM8K", perf_stats, accuracy=answer_accuracy, correct_q=correct_q, total_q=total_q)

    # 5. Return metrics as a JSON-serializable dict for better scalability
    return {
        **perf_stats,
        "accuracy": float(answer_accuracy),
    }

# ---- AIME ----------------------------------------------------------------
def run_aime_eval(generator, tokenizer,
                  past_key_values, draft_past_key_values,
                  args, dataset, log_dir):
    """
    Evaluate AIME‑2024 accuracy alongside performance metrics.

    Args:
        generator:       model generator instance with .generate and .exp_log
        tokenizer:       tokenizer supporting .apply_chat_template and .decode
        past_key_values: primary past key values for autoregressive generation
        draft_past_key_values: optional speculative-decoding pasts
        args:            namespace with temperature, max_length, do_sample, warmup_iter
        dataset:         list of dicts, each with keys:
                         "question": the full prompt string
                         "answer"  : ground truth string (just the numeric answer)
        log_dir:         directory for per-sample JSONL logs

    Returns:
        (tput_mean, tput_std, tacc_mean, tacc_std,
         answer_accuracy, avg_draft_time, avg_target_time, peak_memory)
    """

    warmup_prompt = "Solve this math problem. Give the reasoning steps ...\nWhat is 1 + 1?"
    _run_warmup(
        generator,
        tokenizer,
        past_key_values,
        draft_past_key_values,
        args,
        warmup_prompt,
        max_length=args.max_length,
    )

    # 2. Main loop
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "aime_eval.jsonl")

    perf = _init_perf()
    total_q, correct_q = 0, 0
    int_regex = re.compile(r"[-+]?\d+")

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating AIME"):
        prompt = entry["question"]
        ground_truth = entry["answer"].strip()

        tokenizer.use_default_system_prompt = True
        input_ids = tokenizer.apply_chat_template(
            [{"role":"user","content":prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)

        if input_ids.shape[1] > args.max_length:
            continue

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(
                input_ids,
                temperature=args.temperature,
                max_length=args.max_length,
                do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values
            )
        reset_kv(past_key_values, draft_past_key_values)

        response = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        # Build record
        record = {
            **wandb_logger.log_data,
            "query": prompt,
            "response": response,
            "answer": ground_truth,
            "peak_memory": torch.cuda.max_memory_reserved(generator.device) / (1024**3)
        }

        # Extract integers
        pred_match = int_regex.search(response.splitlines()[-1])
        gt_match   = int_regex.search(ground_truth.splitlines()[-1])
        pred_int = pred_match.group(0).lstrip("+").lstrip("0") or "0" if pred_match else None
        gt_int   = gt_match.group(0).lstrip("+").lstrip("0") or "0" if gt_match else None

        is_correct = (pred_int is not None and gt_int is not None and pred_int == gt_int)
        total_q += 1
        if is_correct:
            correct_q += 1
        record["Accuracy"] = int(is_correct)

        # Aggregate perf metrics
        _accum_perf(perf, record)

        # Log per sample
        with open(log_file, "a+") as f:
            json.dump(record, f)
            f.write("\n")

        # Cleanup
        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Aggregate overall
    accuracy = correct_q / total_q if total_q else 0
    perf_stats = _finalize_perf(perf, generator)
    _print_summary("AIME", perf_stats, accuracy=accuracy, correct_q=correct_q, total_q=total_q)

    # Return JSON-like dict for scalability
    return {
        **perf_stats,
        "accuracy": float(accuracy),
    }

# ---- MMLU-Pro -------------------------------------------------------------
def _mmlu_pro_extract_answer(text: str) -> str | None:
    match = re.search(r"answer is \\(?([A-J])\\)?", text)
    if match:
        return match.group(1)
    match = re.search(r".*[aA]nswer:\\s*([A-J])", text)
    if match:
        return match.group(1)
    match = re.search(r"\\b[A-J]\\b(?!.*\\b[A-J]\\b)", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def run_mmlu_pro_eval(generator, tokenizer,
                      past_key_values, draft_past_key_values,
                      args, dataset, log_dir):
    """
    Evaluate MMLU‑Pro multiple‑choice accuracy + perf metrics.
    `dataset` should be the list from load_mmlu_pro_dataset_answer().
    """
    from run.pipelines.benchmarks import mmlu_pro as mmlu_pro_utils

    warmup = "What is 1 + 1?"
    warmup_prompt = f"{warmup}\\n\\nA. 0\\nB. 1\\nC. 2\\nD. 3\\nE. 4\\nF. 5\\nG. 6\\nH. 7\\nI. 8\\nJ. 9\\n\\nAnswer:"
    _run_warmup(
        generator,
        tokenizer,
        past_key_values,
        draft_past_key_values,
        args,
        warmup_prompt,
        max_length=args.max_length,
    )

    random.seed(12345)
    _, val_df = mmlu_pro_utils.load_mmlu_pro_splits()
    ntrain = getattr(args, "ntrain", 5)
    max_model_length = args.max_length
    max_new_tokens = min(2048, max_model_length // 2)
    prompt_limit = max_model_length - max_new_tokens
    if prompt_limit <= 0:
        prompt_limit = max_model_length

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "mmlu_pro.jsonl")

    perf = _init_perf()
    total_q, correct_q = 0, 0

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc="Eval MMLU‑Pro"):
        prompt = None
        input_ids = None
        k = ntrain
        while k >= 0:
            prompt = mmlu_pro_utils.generate_cot_prompt(val_df, entry, k)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(generator.device)
            if input_ids.shape[1] <= prompt_limit:
                break
            k -= 1

        if input_ids is None or input_ids.shape[1] > args.max_length:
            continue

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(
                input_ids,
                temperature=args.temperature,
                max_length=args.max_length,
                do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values,
            )
        reset_kv(past_key_values, draft_past_key_values)

        resp = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        if "Question:" in resp:
            resp = resp.split("Question:")[0].strip()

        pred = _mmlu_pro_extract_answer(resp)
        random_guess = False
        if pred is None:
            random_guess = True
            rand_idx = random.randint(0, len(entry["options"]) - 1)
            pred = mmlu_pro_utils.CHOICES[rand_idx]
            is_correct = rand_idx == entry["answer_index"]
        else:
            is_correct = (pred == entry["answer"])

        total_q += 1
        if is_correct:
            correct_q += 1

        record = {
            **wandb_logger.log_data,
            "query": prompt,
            "response": resp,
            "answer": entry["answer"],
            "pred": pred,
            "random_guess": random_guess,
            "k_shot": k,
            "Accuracy": int(is_correct),
            "peak_memory": torch.cuda.max_memory_reserved(generator.device) / (1024**3),
        }
        _accum_perf(perf, record)

        with open(log_file, "a+") as f:
            json.dump(record, f)
            f.write("\n")

        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    accuracy = correct_q / total_q if total_q else 0
    perf_stats = _finalize_perf(perf, generator)
    _print_summary("MMLU‑Pro", perf_stats, accuracy=accuracy, correct_q=correct_q, total_q=total_q)

    return {
        **perf_stats,
        "accuracy": float(accuracy),
    }


def _maybe_delimit(prefix: str | None, suffix: str | None, delimiter: str = " ") -> str:
    """Return continuation text with lm-eval-style delimiter behavior."""
    if suffix is None:
        return ""
    if not prefix or not suffix:
        return suffix
    if delimiter is None:
        return suffix
    if prefix[-1].isspace() or suffix[0].isspace():
        return suffix
    return f"{delimiter}{suffix}"


def _get_forward_fn(model):
    return getattr(model, "_orig_forward", model.forward)


def _encode_context(tokenizer, text: str, device, use_chat_template: bool = False):
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    return tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)


def _loglikelihood_continuation(
    model,
    tokenizer,
    prefix_ids,
    continuation_text: str,
    max_length: int | None = None,
    return_len: bool = False,
):
    cont_ids = tokenizer(
        continuation_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(prefix_ids.device)
    cont_len = cont_ids.shape[1]
    if cont_len == 0:
        return (0.0, 0) if return_len else 0.0
    if max_length is not None and (prefix_ids.shape[1] + cont_len) > max_length:
        return (float("-inf"), cont_len) if return_len else float("-inf")

    input_ids = torch.cat([prefix_ids, cont_ids], dim=1)
    forward_fn = _get_forward_fn(model)
    with torch.no_grad():
        outputs = forward_fn(input_ids)

    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    cont_start = prefix_ids.shape[1]
    cont_len = cont_ids.shape[1]
    if cont_start == 0:
        return 0.0

    target_ids = input_ids[:, cont_start:cont_start + cont_len]
    pos = torch.arange(cont_start - 1, cont_start - 1 + cont_len, device=input_ids.device)
    token_log_probs = log_probs[0, pos, target_ids[0]]
    ll = float(token_log_probs.sum().item())
    return (ll, cont_len) if return_len else ll


# ---- Multi-choice (LL-based) ----------------------------------------------
def run_multichoice_ll_eval(
    generator,
    tokenizer,
    past_key_values,
    draft_past_key_values,
    args,
    dataset,
    log_dir,
    bench_name: str = None,
    max_new_tokens: int = 64,
):
    """
    Loglikelihood-based multiple-choice evaluation (LM harness style).
    Expected fields per entry:
      - context: str (prefix)
      - choices: List[str]
      - answer_index: int
    For Winogrande-style partial evaluation:
      - multiple_input: True
      - contexts: List[str]
      - target: str
    """
    warmup_prompt = (
        "Choose the correct answer.\n"
        "Question: What is 1 + 1?\nAnswer:"
    )
    _run_warmup(
        generator,
        tokenizer,
        past_key_values,
        draft_past_key_values,
        args,
        warmup_prompt,
        max_new_tokens=max_new_tokens,
    )

    os.makedirs(log_dir, exist_ok=True)
    file_tag = bench_name if bench_name is not None else "multichoice_ll"
    log_file = os.path.join(log_dir, f"{file_tag}.jsonl")

    perf = _init_perf()
    total_q, correct_q = 0, 0
    acc_norm_total = 0.0

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc=f"Evaluating {file_tag}"):
        target_delimiter = entry.get("target_delimiter", " ")
        answer_index = int(entry.get("answer_index", -1))
        total_tokens = 0
        total_time = 0.0

        if entry.get("multiple_input", False):
            contexts = entry["contexts"]
            lls = []
            cont_lens = []
            use_chat_template = bool(entry.get("use_chat_template", False))
            for ctx in contexts:
                cont = _maybe_delimit(ctx, entry["target"], target_delimiter)
                prefix_ids = _encode_context(tokenizer, ctx, generator.device, use_chat_template)
                if prefix_ids.shape[1] > args.max_length:
                    lls.append(float("-inf"))
                    cont_lens.append(1)
                    continue
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                ll, cont_len = _loglikelihood_continuation(
                    generator.target_model,
                    tokenizer,
                    prefix_ids,
                    cont,
                    max_length=args.max_length,
                    return_len=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_time += time.perf_counter() - start
                total_tokens += int(prefix_ids.shape[1] + cont_len)
                lls.append(ll)
                cont_lens.append(max(cont_len, 1))
            norm_lengths = np.array(cont_lens, dtype=np.float32)
        else:
            context = entry["context"]
            choices = entry["choices"]
            use_chat_template = bool(entry.get("use_chat_template", False))
            prefix_ids = _encode_context(tokenizer, context, generator.device, use_chat_template)
            if prefix_ids.shape[1] > args.max_length:
                continue
            lls = []
            cont_lens = []
            for choice in choices:
                cont = _maybe_delimit(context, choice, target_delimiter)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                ll, cont_len = _loglikelihood_continuation(
                    generator.target_model,
                    tokenizer,
                    prefix_ids,
                    cont,
                    max_length=args.max_length,
                    return_len=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_time += time.perf_counter() - start
                total_tokens += int(prefix_ids.shape[1] + cont_len)
                lls.append(ll)
                cont_lens.append(max(cont_len, 1))
            norm_lengths = np.array(cont_lens, dtype=np.float32)

        if not lls or answer_index < 0:
            continue

        pred = int(np.argmax(lls))
        total_q += 1
        if pred == answer_index:
            correct_q += 1

        pred_norm = int(np.argmax(np.array(lls) / norm_lengths))
        acc_norm_total += 1.0 if pred_norm == answer_index else 0.0

        if total_time > 0:
            perf_record = {
                "tput": float(total_tokens / total_time),
                "avg_sampled": 1.0,
                "n_iter": 1,
                "avg_draft_time": 0.0,
                "avg_target_time": float(total_time),
            }
            _accum_perf(perf, perf_record)

        record = {
            "index": idx,
            "lls": lls,
            "pred": pred,
            "answer_index": answer_index,
        }
        with open(log_file, "a+") as f:
            json.dump(record, f)
            f.write("\n")

    accuracy = correct_q / total_q if total_q else 0.0
    acc_norm = acc_norm_total / total_q if total_q else 0.0
    perf_stats = _finalize_perf(perf, generator)
    title = bench_name if bench_name is not None else "Multiple-Choice-LL"
    _print_summary(
        title,
        perf_stats,
        accuracy=accuracy,
        correct_q=correct_q,
        total_q=total_q,
        accuracy_norm=acc_norm,
        draft_time_note="N/A (LL eval)",
    )

    return {
        **perf_stats,
        "accuracy": float(accuracy),
        "acc_norm": float(acc_norm),
    }


# --- Utility functions consolidated from lcb_runner ---

def _extract_code(text: str) -> str:
    """Extracts code from a ```python ... ``` block."""
    match = re.search(r"```(?:python)?\n(.*?)\n```", text, re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def _build_humaneval_program(prompt: str, completion: str, entry_point: str) -> str:
    completion = _extract_code(completion)
    if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", completion):
        lines = prompt.splitlines()
        preamble_lines = []
        for line in lines:
            if line.lstrip().startswith("def "):
                break
            preamble_lines.append(line)
        preamble = "\n".join(preamble_lines).rstrip()
        if preamble:
            preamble += "\n\n"
        return preamble + completion
    prompt_text = prompt if prompt.endswith("\n") else prompt + "\n"
    return prompt_text + completion


# ---- HumanEval ------------------------------------------------------------
def check_humaneval_correctness(problem: dict, completion: str, timeout: float = 3.0) -> dict:
    """
    Minimal HumanEval correctness checker using the dataset's test harness.
    """
    program = _build_humaneval_program(problem["prompt"], completion, problem["entry_point"])
    test_code = problem["test"]
    entry_point = problem["entry_point"]

    with tempfile.TemporaryDirectory() as temp_dir:
        code_path = os.path.join(temp_dir, "main.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(program)
            f.write("\n\n")
            f.write(test_code)
            f.write("\n\n")
            f.write(f"check({entry_point})\n")

        try:
            proc = subprocess.run(
                ["python", code_path],
                capture_output=True,
                timeout=timeout,
            )
            return {"passed": proc.returncode == 0}
        except (subprocess.TimeoutExpired, Exception):
            return {"passed": False}


def run_humaneval_eval(
    generator,
    tokenizer,
    past_key_values,
    draft_past_key_values,
    args,
    dataset,
    log_dir,
    n_samples: int = 1,
    test_timeout: float = 3.0,
):
    """
    Evaluate HumanEval using the official human-eval evaluation (pass@k).
    Dataset entries should include:
        "question", "prompt", "test", "entry_point", "task_id"
    """
    os.makedirs(log_dir, exist_ok=True)

    humaneval_samples = os.path.join(log_dir, "humaneval_samples.jsonl")
    humaneval_problems = os.path.join(log_dir, "humaneval_problems.jsonl")
    log_file = os.path.join(log_dir, "humaneval.jsonl")

    n_samples = int(getattr(args, "humaneval_n_samples", n_samples))
    n_workers = int(getattr(args, "humaneval_workers", 4))
    test_timeout = float(getattr(args, "humaneval_timeout", test_timeout))

    perf = _init_perf()
    samples = []
    problem_records = []

    for idx, problem in tqdm(enumerate(dataset), total=len(dataset), desc="Generating HumanEval samples"):
        prompt = problem["prompt"]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(generator.device)
        if input_ids.shape[1] > args.max_length:
            continue

        for _ in range(n_samples):
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                output_ids = generator.generate(
                    input_ids,
                    temperature=args.temperature,
                    max_length=args.max_length,
                    do_sample=args.do_sample,
                    past_key_values=past_key_values,
                    draft_past_key_values=draft_past_key_values,
                )

            reset_kv(past_key_values, draft_past_key_values)

            # Keep leading indentation for HumanEval completions.
            completion = tokenizer.decode(
                output_ids[0, input_ids.shape[1]:],
                skip_special_tokens=True
            )

            samples.append({
                "task_id": problem["task_id"],
                "completion": completion,
            })

            record = {
                **wandb_logger.log_data,
                "task_id": problem.get("task_id"),
                "completion": completion,
                "peak_memory": torch.cuda.max_memory_reserved(generator.device) / (1024 ** 3),
            }
            _accum_perf(perf, record)
            with open(log_file, "a+") as f:
                json.dump(record, f)
                f.write("\n")

        problem_records.append({
            "task_id": problem["task_id"],
            "prompt": problem["prompt"],
            "test": problem["test"],
            "entry_point": problem["entry_point"],
        })

        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    write_jsonl(humaneval_samples, samples)
    write_jsonl(humaneval_problems, problem_records)

    pass_at_k = evaluate_functional_correctness(
        humaneval_samples,
        k=[1, 10, 100],
        n_workers=n_workers,
        timeout=test_timeout,
        problem_file=humaneval_problems,
    )

    accuracy = float(pass_at_k.get("pass@1", 0.0))
    perf_stats = _finalize_perf(perf, generator)
    _print_summary("HumanEval", perf_stats, accuracy=accuracy)

    return {
        **perf_stats,
        **pass_at_k,
    }

# ---- LiveCodeBench --------------------------------------------------------
def _decode_test_cases(field: Any) -> List[Dict[str, str]]:
    """
    Robustly decodes LiveCodeBench public/private test-cases.
    This logic is critical for handling the various data formats.
    """
    if isinstance(field, list):
        return field

    if isinstance(field, bytes):
        s = field.decode("utf-8", errors="ignore").strip()
    else:
        s = str(field).strip()

    if s.lstrip().startswith("["):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass # Fall through

    try:
        data = base64.b64decode(s)
        if data.startswith(b'\x78\x9c'): # zlib compressed
            data = zlib.decompress(data)
        
        try: # Try JSON first
            return json.loads(data.decode("utf-8"))
        except: # Fall back to pickle
            return pickle.loads(data)
    except Exception as e:
        raise ValueError(f"Could not decode test case data: {e}") from None

def _run_single_test(python_src: str, test_case: dict, timeout: float) -> bool:
    """Runs a single test case against the provided Python source."""
    with tempfile.TemporaryDirectory() as temp_dir:
        code_path = os.path.join(temp_dir, "main.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(python_src)

        try:
            proc = subprocess.run(
                ["python", code_path],
                input=test_case["input"].encode("utf-8"),
                capture_output=True,
                timeout=timeout,
            )
            # Compare stripped stdout to expected output
            return proc.stdout.decode("utf-8").strip() == test_case["output"].strip()
        except (subprocess.TimeoutExpired, Exception):
            return False

# --- Main function to replace the library call ---

def check_correctness(problem: dict, completion: str, timeout: float = 2.0) -> dict:
    """
    Self-contained function to grade a model's completion for a given problem.

    Args:
        problem: The problem dictionary from the dataset.
        completion: The string response generated by the model.
        timeout: Timeout in seconds for each test case.

    Returns:
        A dictionary with a "passed" boolean key.
    """
    solution_code = _extract_code(completion)
    if not solution_code:
        return {"passed": False}

    try:
        public_tests = _decode_test_cases(problem["public_test_cases"])
        private_tests = _decode_test_cases(problem["private_test_cases"])
        all_tests = public_tests + private_tests
    except ValueError:
        return {"passed": False} # Failed to decode tests

    for test_case in all_tests:
        if not _run_single_test(solution_code, test_case, timeout):
            return {"passed": False} # Failed a test case

    return {"passed": True} # Passed all test cases

def run_livecodebench_eval(
    generator,
    tokenizer,
    past_key_values,
    draft_past_key_values,
    args,
    dataset,
    log_dir,
    n_samples=1,
    test_timeout=2.0,
):
    """
    LiveCodeBench evaluation using the official lcb_runner execution and pass@k.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "livecodebench.jsonl")

    n_samples = int(getattr(args, "livecodebench_n_samples", n_samples))
    test_timeout = float(getattr(args, "livecodebench_timeout", test_timeout))
    lm_style_name = getattr(args, "livecodebench_lm_style", "OpenAIChat")
    try:
        lm_style = LMStyle[lm_style_name]
    except KeyError:
        lm_style = LMStyle.OpenAIChat

    perf = _init_perf()
    results = {}

    for i, entry in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating LiveCodeBench"):
        prompt = entry["question"]
        problem = entry["problem"]
        eval_sample = entry["eval_sample"]
        task_id = getattr(problem, "question_id", f"idx_{i}")

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(generator.device)
        if input_ids.shape[1] > args.max_length:
            continue

        responses = []
        graded_list = []
        result_list = []
        for _ in range(n_samples):
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                output_ids = generator.generate(
                    input_ids,
                    temperature=args.temperature,
                    max_length=args.max_length,
                    do_sample=args.do_sample,
                    past_key_values=past_key_values,
                    draft_past_key_values=draft_past_key_values,
                )

            reset_kv(past_key_values, draft_past_key_values)

            response = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            responses.append(response)

            extracted = lcb_extract_code(response, lm_style)
            if not extracted:
                extracted = response

            res, _metadata = lcb_check_correctness(eval_sample, extracted, timeout=test_timeout, debug=False)
            result_list.append(res)
            graded_list.append(all(r > 0 for r in res))

        results.setdefault(task_id, []).extend(result_list)

        pass1 = int(graded_list[0] if graded_list else 0)
        record = {
            **wandb_logger.log_data,
            "query": prompt,
            "responses": responses,
            "graded_list": graded_list,
            "pass@1": pass1,
            "n": n_samples,
            "platform": getattr(problem, "platform", None),
            "difficulty": getattr(problem, "difficulty", None),
            "contest_date": getattr(problem, "contest_date", None),
            "question_id": task_id,
            "peak_memory": torch.cuda.max_memory_reserved(generator.device) / (1024 ** 3),
        }
        _accum_perf(perf, record)
        with open(log_file, "a+") as f:
            json.dump(record, f)
            f.write("\n")

        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    metrics = lcb_compute_metrics(results, k_list=[1, 5])
    pass1 = float(metrics.get("pass@1", 0.0))
    perf_stats = _finalize_perf(perf, generator)
    _print_summary("LiveCodeBench", perf_stats, accuracy=pass1)

    return {
        **perf_stats,
        **metrics,
    }

# ---- LongBench ------------------------------------------------------------
def run_longbench_eval(generator, tokenizer, past_key_values, draft_past_key_values, args, dataset, log_dir, bench_name):
    """
    Evaluate longbench dataset accuracy alongside performance metrics.
    Ex. "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", 
        "gov_report", "qmsum", "multi_news",  "trec", "triviaqa", "samsum",
        "passage_count", "passage_retrieval_en",  "lcc", "repobench_p"
    """
    print("bench name", bench_name)

    with open("run/pipelines/benchmarks/utils/config/dataset2maxlen.json", "r", encoding="utf-8") as f:
        benchmark_max_len = json.load(f)

    max_new_tokens = benchmark_max_len.get(bench_name, args.max_length)

    warmup_prompt = "Solve this math problem. Give the reasoning steps ...\nWhat is 1 + 1?" * 64
    _run_warmup(
        generator,
        tokenizer,
        past_key_values,
        draft_past_key_values,
        args,
        warmup_prompt,
        max_new_tokens=max_new_tokens,
        show_progress=True,
    )

    # Optional CUDA-graph capture for FlashInfer, after warmup (stabilizes kernels/allocations).
    maybe_init_cuda_graph_runner(generator, past_key_values, draft_past_key_values, args.device, args.warmup_iter)

    log_file = os.path.join(log_dir, "0.jsonl")
    perf = _init_perf()
    total_q = 0
    correct_q = 0

    for _, entry in tqdm(enumerate(dataset), total=len(dataset), desc=f"Evaluating {bench_name}"):
        prompt = entry["question"]
        ground_truth_list = entry["answer"]
        all_classes = entry.get("classes", None)

        tokenizer.use_default_system_prompt = True
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(
                input_ids,
                temperature=args.temperature,
                max_new_tokens=max_new_tokens,
                do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values,
            )

        reset_kv(past_key_values, draft_past_key_values)

        record = {**wandb_logger.log_data}
        record.update({
            "query": prompt,
            "response": tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            ),
            "answer": ground_truth_list,
            "peak_memory": torch.cuda.max_memory_reserved(generator.device) / (1024 ** 3),
        })

        response = record["response"]
        if bench_name in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = response.lstrip("\n").split("\n")[0]
        else:
            prediction = response

        score = 0
        for ground_truth in ground_truth_list:
            score = max(score, dataset2metric[bench_name](prediction, ground_truth, all_classes=all_classes))

        total_q += 1
        correct_q += score
        record["Accuracy"] = score

        _accum_perf(perf, record)

        with open(log_file, "a+") as f:
            json.dump(record, f)
            f.write("\n")

        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    answer_accuracy = round(100 * correct_q / total_q, 2) if total_q > 0 else 0
    perf_stats = _finalize_perf(perf, generator)
    _print_summary(bench_name, perf_stats, accuracy=answer_accuracy, correct_q=correct_q, total_q=total_q)

    return {
        **perf_stats,
    }

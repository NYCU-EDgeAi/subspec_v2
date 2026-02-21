"""Benchmark pipeline for accuracy evaluation."""

import inspect
import logging
import os
import time

from tqdm import tqdm

from .benchmarks.registry import load_dataset, validate_benchmarks
from .benchmarks.task_registry import (
    get_lane_evaluator,
    normalize_lane,
    resolve_lane_for_task,
    validate_lane_compatibility,
)
from .benchmarks.utils.code_eval import validate_humaneval_runtime_requirements
from .utils.benchmark_utils import (
    append_benchmark_result,
    cleanup_gpu,
    parse_benchmark_list,
    prepare_output_dir,
    reset_seeds,
    setup_benchmark_dir,
)


def _invoke_eval(
    eval_fn,
    generator,
    tokenizer,
    past_kv,
    draft_past_kv,
    args,
    dataset,
    log_dir,
    bench_name,
):
    if "bench_name" in inspect.signature(eval_fn).parameters:
        return eval_fn(generator, tokenizer, past_kv, draft_past_kv, args, dataset, log_dir, bench_name)
    return eval_fn(generator, tokenizer, past_kv, draft_past_kv, args, dataset, log_dir)


def main(builder, benchmarks=None, max_samples=None, lane: str | None = None):
    """Run accuracy benchmarks on specified datasets."""
    reset_seeds(0)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    requested_lane = normalize_lane(lane)

    bench_list = parse_benchmark_list(benchmarks)
    if not bench_list:
        raise ValueError("--benchmarks is required for run-benchmark-acc")
    validate_benchmarks(bench_list, with_answers=True)
    validate_lane_compatibility(bench_list, requested_lane)
    if "human-eval" in bench_list or "human-eval-instruct" in bench_list:
        validate_humaneval_runtime_requirements()
    print(f"Benchmarks to run: {bench_list}")
    print(f"Lane: {requested_lane or 'auto'}")

    builder.generator_profiling = True
    builder.profiling_verbose = False
    generator, tokenizer, past_kv, draft_past_kv = builder.build()
    args = builder.args

    prepare_output_dir(args.out_dir)

    log_dir_base = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"), "run_benchmark_acc")
    for bench_name in tqdm(bench_list, desc="Running benchmarks"):
        reset_seeds(0)
        log_dir = setup_benchmark_dir(log_dir_base, bench_name, getattr(args, "settings_snapshot", None))
        print(f"Log directory: {log_dir}")

        dataset = load_dataset(bench_name, max_samples=max_samples, seed=0, shuffle=True, with_answers=True)
        print(f"Running benchmark: {bench_name}, samples: {len(dataset)}")

        cleanup_gpu()

        eval_start = time.perf_counter()
        task_lane = resolve_lane_for_task(bench_name, requested_lane)
        print(f"Using lane '{task_lane}' for {bench_name}")
        eval_fn = get_lane_evaluator(bench_name, task_lane)
        metrics_json = _invoke_eval(
            eval_fn,
            generator,
            tokenizer,
            past_kv,
            draft_past_kv,
            args,
            dataset,
            log_dir,
            bench_name,
        )
        eval_time_s = time.perf_counter() - eval_start

        cleanup_gpu()

        metrics_json["total_eval_time_s"] = eval_time_s
        append_benchmark_result(log_dir, bench_name, metrics_json, digits=3)

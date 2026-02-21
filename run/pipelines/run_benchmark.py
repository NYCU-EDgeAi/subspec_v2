"""Benchmark pipeline for throughput evaluation."""

import os
import time
import logging
from tqdm import tqdm

from .benchmarks.registry import load_dataset, validate_benchmarks
from .benchmarks.utils.eval import run_common_eval, run_mtbench_eval
from .utils.benchmark_utils import (
    append_benchmark_result,
    cleanup_gpu,
    parse_benchmark_list,
    prepare_output_dir,
    reset_seeds,
    setup_benchmark_dir,
)

BENCHMARK_EVALUATORS = {
    "mt-bench": run_mtbench_eval,
    "human-eval": run_common_eval,
    "human-eval-instruct": run_common_eval,
    "gsm8k": run_common_eval,
    "alpaca": run_common_eval,
    "cnn-dm": run_common_eval,
    "aime": run_common_eval,
    "gpqa": run_common_eval,
    "math-500": run_common_eval,
    "livecodebench": run_common_eval,
    "hotpotqa": run_common_eval,
    "narrativeqa": run_common_eval,
    "qasper": run_common_eval,
    "multifieldqa_en": run_common_eval,
    "2wikimqa": run_common_eval,
    "musique": run_common_eval,
    "gov_report": run_common_eval,
    "qmsum": run_common_eval,
    "multi_news": run_common_eval,
    "trec": run_common_eval,
    "triviaqa": run_common_eval,
    "samsum": run_common_eval,
    "passage_count": run_common_eval,
    "passage_retrieval_en": run_common_eval,
    "lcc": run_common_eval,
    "repobench_p": run_common_eval,
}


def _validate_pipeline_benchmarks(bench_list):
    unsupported = [bench for bench in bench_list if bench not in BENCHMARK_EVALUATORS]
    if unsupported:
        raise ValueError(
            "run-benchmark does not support benchmark(s): "
            f"{unsupported}. Supported: {sorted(BENCHMARK_EVALUATORS.keys())}"
        )


def main(builder, benchmarks=None, max_samples=None):
    """Run throughput benchmarks on specified datasets."""
    reset_seeds(0)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())

    # Validate benchmarks
    bench_list = parse_benchmark_list(benchmarks)
    if not bench_list:
        raise ValueError("--benchmarks is required for run-benchmark")
    validate_benchmarks(bench_list)
    _validate_pipeline_benchmarks(bench_list)
    print(f"Benchmarks to run: {bench_list}")

    builder.generator_profiling = True
    builder.profiling_verbose = False
    generator, tokenizer, past_kv, draft_past_kv = builder.build()
    args = builder.args

    # Handle output directories
    prepare_output_dir(args.out_dir)
        
    # Run benchmarks
    log_dir_base = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"), "run_benchmark")
    for bench_name in tqdm(bench_list, desc="Running benchmarks"):
        reset_seeds(0)
        log_dir = setup_benchmark_dir(log_dir_base, bench_name, getattr(args, "settings_snapshot", None))
        print(f"Log directory: {log_dir}")
        
        dataset = load_dataset(bench_name, max_samples=max_samples, seed=0, shuffle=True)
        print(f"Running benchmark: {bench_name}, samples: {len(dataset)}")
        
        cleanup_gpu()

        # Evaluate
        eval_start = time.perf_counter()
        metrics_json = BENCHMARK_EVALUATORS[bench_name](generator, tokenizer, past_kv, draft_past_kv, args, dataset, log_dir)
        eval_time_s = time.perf_counter() - eval_start
        
        cleanup_gpu()
        
        # Save results
        metrics_json["total_eval_time_s"] = eval_time_s
        append_benchmark_result(log_dir, bench_name, metrics_json, digits=3)

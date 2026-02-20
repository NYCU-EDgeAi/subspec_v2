"""Benchmark pipeline for agent evaluation."""

import os
import time
import logging
from tqdm import tqdm

from .benchmarks.registry import load_dataset, validate_benchmarks
from .benchmarks.utils.eval_agent import run_agent_eval
from .utils.benchmark_utils import (
    append_benchmark_result,
    cleanup_gpu,
    parse_benchmark_list,
    prepare_output_dir,
    reset_seeds,
    setup_benchmark_dir,
)

BENCHMARK_EVALUATORS = {
    "hotpotqa": run_agent_eval,
}


def _validate_pipeline_benchmarks(bench_list):
    unsupported = [bench for bench in bench_list if bench not in BENCHMARK_EVALUATORS]
    if unsupported:
        raise ValueError(
            "run-benchmark-agent does not support benchmark(s): "
            f"{unsupported}. Supported: {sorted(BENCHMARK_EVALUATORS.keys())}"
        )


def main(builder, benchmarks=None, max_samples=None):
    """Run agent benchmarks on specified datasets."""
    reset_seeds(0)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())

    # Validate benchmarks
    bench_list = parse_benchmark_list(benchmarks)
    if not bench_list:
        raise ValueError("--benchmarks is required for run-benchmark-agent")
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
    log_dir_base = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"), "run_benchmark_agent")
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

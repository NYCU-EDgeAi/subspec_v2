"""Shared utilities for benchmark pipelines."""

import json
import os
import gc
import random
import shutil
import torch
from typing import Any, Dict, List

from run.core.config_utils import write_settings_yaml


def reset_seeds(seed: int = 0) -> None:
    """Reset random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)


def cleanup_gpu() -> None:
    """Clean up GPU memory."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()


def setup_benchmark_dir(log_dir_base: str, bench_name: str, settings_snapshot=None) -> str:
    """Create benchmark output directory and write settings.
    
    Args:
        log_dir_base: Base directory for logs
        bench_name: Name of the benchmark
        settings_snapshot: Optional settings dict to write
        
    Returns:
        Path to the created benchmark directory
    """
    log_dir = os.path.join(log_dir_base, bench_name)
    os.makedirs(log_dir, exist_ok=True)
    write_settings_yaml(log_dir, settings_snapshot)
    return log_dir


def parse_benchmark_list(benchmarks: str | None) -> List[str]:
    """Parse comma-separated benchmark names into a cleaned list."""
    if not benchmarks:
        return []
    return [name.strip() for name in benchmarks.split(",") if name.strip()]


def prepare_output_dir(out_dir: str | None) -> None:
    """Recreate an output directory when one is configured."""
    if out_dir is None:
        return
    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"Deleted old {out_dir}")
    os.makedirs(out_dir, exist_ok=True)


def round_metric_values(metrics: Dict[str, Any], digits: int = 3) -> Dict[str, Any]:
    """Round float values in a top-level metrics dict."""
    return {k: round(v, digits) if isinstance(v, float) else v for k, v in metrics.items()}


def append_benchmark_result(
    log_dir: str,
    bench_name: str,
    metrics: Dict[str, Any],
    *,
    digits: int = 3,
) -> None:
    """Append a benchmark result record to results.jsonl."""
    record = {bench_name: round_metric_values(metrics, digits=digits)}
    with open(os.path.join(log_dir, "results.jsonl"), "a") as f:
        json.dump(record, f, indent=4)
        f.write("\n")

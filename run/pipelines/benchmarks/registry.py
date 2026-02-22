"""Centralized benchmark registry with lazy loading.

Provides lazy-loaded benchmark dataset loaders to avoid importing
heavy dependencies (like `datasets`) until actually needed.
"""

from typing import Callable, List, Optional, Any
import random


AVAILABLE_BENCHMARKS = [
    "mt-bench",
    "human-eval",
    "human-eval-instruct",
    "gsm8k",
    "hellaswag",
    "piqa",
    "arc-c",
    "winogrande",
    "alpaca",
    "cnn-dm",
    "aime",
    "gpqa",
    "math-500",
    "livecodebench",
    "mmlu_pro",
    "hotpotqa",
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench_p",
]

_LOADER_CONFIG = {
    "mt-bench": (".mtbench", "load_mtbench_dataset", "load_mtbench_dataset_answer"),
    "human-eval": (".humaneval", "load_humaneval_dataset", "load_humaneval_dataset_answer"),
    "human-eval-instruct": (
        ".humaneval",
        "load_humaneval_instruct_dataset",
        "load_humaneval_instruct_dataset_answer",
    ),
    "gsm8k": (".gsm8k", "load_gsm8k_dataset", "load_gsm8k_dataset_answer"),
    "hellaswag": (".hellaswag", "load_hellaswag_dataset", "load_hellaswag_dataset_answer"),
    "piqa": (".piqa", "load_piqa_dataset", "load_piqa_dataset_answer"),
    "arc-c": (".arc_c", "load_arc_c_dataset", "load_arc_c_dataset_answer"),
    "winogrande": (".winogrande", "load_winogrande_dataset", "load_winogrande_dataset_answer"),
    "alpaca": (".alpaca", "load_alpaca_dataset", None),
    "cnn-dm": (".cnndm", "load_cnndm_dataset", None),
    "aime": (".aime", "load_aime_dataset", "load_aime_dataset_answer"),
    "gpqa": (".gpqa", "load_gpqa_dataset", None),
    "math-500": (".math500", "load_math500_dataset", None),
    "livecodebench": (".livecodebench", "load_livecodebench_dataset", "load_livecodebench_dataset_answer"),
    "mmlu_pro": (".mmlu_pro", "load_mmlu_pro_dataset", "load_mmlu_pro_dataset_answer"),
    "hotpotqa": (".hotpotqa", "load_hotpotqa_dataset", "load_hotpotqa_dataset_answer"),
    "narrativeqa": (".narrativeqa", "load_narrativeqa_dataset", "load_narrativeqa_dataset_answer"),
    "qasper": (".qasper", "load_qasper_dataset", "load_qasper_dataset_answer"),
    "multifieldqa_en": (".multifieldqa_en", "load_multifieldqa_en_dataset", "load_multifieldqa_en_dataset_answer"),
    "2wikimqa": ("._2wikimqa", "load_2wikimqa_dataset", "load_2wikimqa_dataset_answer"),
    "musique": (".musique", "load_musique_dataset", "load_musique_dataset_answer"),
    "gov_report": (".gov_report", "load_gov_report_dataset", "load_gov_report_dataset_answer"),
    "qmsum": (".qmsum", "load_qmsum_dataset", "load_qmsum_dataset_answer"),
    "multi_news": (".multi_news", "load_multi_news_dataset", "load_multi_news_dataset_answer"),
    "trec": (".trec", "load_trec_dataset", "load_trec_dataset_answer"),
    "triviaqa": (".triviaqa", "load_triviaqa_dataset", "load_triviaqa_dataset_answer"),
    "samsum": (".samsum", "load_samsum_dataset", "load_samsum_dataset_answer"),
    "passage_count": (".passage_count", "load_passage_count_dataset", "load_passage_count_dataset_answer"),
    "passage_retrieval_en": (".passage_retrieval_en", "load_passage_retrieval_en_dataset", "load_passage_retrieval_en_dataset_answer"),
    "lcc": (".lcc", "load_lcc_dataset", "load_lcc_dataset_answer"),
    "repobench_p": (".repobench_p", "load_repobench_p_dataset", "load_repobench_p_dataset_answer"),
}


def _import_error_message(bench_name: str, package: str = "datasets") -> str:
    return (
        f"Benchmark '{bench_name}' requires the '{package}' library.\n"
        f"Install with: pip install {package}"
    )


def get_loader(bench_name: str, with_answers: bool = False) -> Callable:
    """Get a dataset loader function for the specified benchmark.
    
    Args:
        bench_name: Name of the benchmark (e.g., "mt-bench", "gsm8k")
        with_answers: If True, return loader that includes answer data (for accuracy eval)
        
    Returns:
        A callable that loads the dataset when invoked
        
    Raises:
        ValueError: If benchmark name is unknown or doesn't support with_answers
        ImportError: If required dependencies are not installed
    """
    if bench_name not in AVAILABLE_BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: '{bench_name}'. "
            f"Available: {', '.join(AVAILABLE_BENCHMARKS)}"
        )
    
    module_name, func_name, func_name_answer = _LOADER_CONFIG[bench_name]
    
    if with_answers:
        if func_name_answer is None:
            raise ValueError(f"Benchmark '{bench_name}' does not support with_answers=True")
        target_func = func_name_answer
    else:
        target_func = func_name
    
    try:
        import importlib
        module = importlib.import_module(module_name, package="run.pipelines.benchmarks")
        return getattr(module, target_func)
    except ImportError as e:
        raise ImportError(_import_error_message(bench_name)) from e


def extract_prompt(item: Any) -> str:
    """Extract prompt text from various benchmark formats.
    
    Handles:
        - str: Return as-is
        - list: MT-Bench format (list of turns), return first turn
        - dict: Try common keys (query, prompt, question, input, text)
    """
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        return item[0] if item else ""
    if isinstance(item, dict):
        for key in ["query", "prompt", "question", "input", "text"]:
            if key in item:
                return item[key]
    return str(item)


def load_dataset(
    bench_name: str,
    max_samples: Optional[int] = None,
    seed: int = 0,
    shuffle: bool = True,
    with_answers: bool = False,
) -> List[Any]:
    """Load a benchmark dataset with optional shuffling and sampling.
    
    Args:
        bench_name: Name of the benchmark
        max_samples: Maximum number of samples to return (None = all)
        seed: Random seed for shuffling
        shuffle: Whether to shuffle the dataset
        with_answers: If True, load dataset with answer data
        
    Returns:
        List of dataset items
    """
    loader = get_loader(bench_name, with_answers=with_answers)
    dataset = loader()
    
    if shuffle:
        random.seed(seed)
        random.shuffle(dataset)
    
    if max_samples is not None:
        dataset = dataset[:max_samples]
    
    return dataset


def validate_benchmarks(bench_list: List[str], with_answers: bool = False) -> None:
    """Validate that all benchmark names are known and support requested mode.
    
    Args:
        bench_list: List of benchmark names to validate
        with_answers: If True, also validate that benchmarks support with_answers mode
        
    Raises:
        ValueError: If any benchmark name is unknown or doesn't support with_answers
    """
    unknown = [b for b in bench_list if b not in AVAILABLE_BENCHMARKS]
    if unknown:
        raise ValueError(
            f"Unknown benchmark(s): {', '.join(unknown)}. "
            f"Available: {', '.join(AVAILABLE_BENCHMARKS)}"
        )
    
    if with_answers:
        no_answer = [b for b in bench_list if _LOADER_CONFIG[b][2] is None]
        if no_answer:
            raise ValueError(
                f"Benchmark(s) do not support with_answers=True: {', '.join(no_answer)}"
            )

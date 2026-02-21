"""Task registry for lane-based benchmark evaluation."""

from dataclasses import dataclass
from typing import Callable, Dict, List

from run.pipelines.benchmarks.utils import eval_acc as evals

LANE_DISTRIBUTION = "distribution"
LANE_BEHAVIOR = "behavior"
SUPPORTED_LANES = {LANE_DISTRIBUTION, LANE_BEHAVIOR}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    evaluators: Dict[str, Callable]


_TASK_SPECS: Dict[str, TaskSpec] = {
    # Distribution lane (LL-based, canonical MC evaluation).
    "hellaswag": TaskSpec(
        "hellaswag",
        {LANE_DISTRIBUTION: evals.run_multichoice_ll_eval},
    ),
    "piqa": TaskSpec(
        "piqa",
        {LANE_DISTRIBUTION: evals.run_multichoice_ll_eval},
    ),
    "arc-c": TaskSpec(
        "arc-c",
        {LANE_DISTRIBUTION: evals.run_multichoice_ll_eval},
    ),
    "winogrande": TaskSpec(
        "winogrande",
        {LANE_DISTRIBUTION: evals.run_multichoice_ll_eval},
    ),
    # Behavior lane (generation-based, SD-aware).
    "gsm8k": TaskSpec("gsm8k", {LANE_BEHAVIOR: evals.run_gsm8k_eval}),
    "human-eval": TaskSpec("human-eval", {LANE_BEHAVIOR: evals.run_humaneval_eval}),
    "human-eval-instruct": TaskSpec("human-eval-instruct", {LANE_BEHAVIOR: evals.run_humaneval_eval}),
    "aime": TaskSpec("aime", {LANE_BEHAVIOR: evals.run_aime_eval}),
    "livecodebench": TaskSpec("livecodebench", {LANE_BEHAVIOR: evals.run_livecodebench_eval}),
    "mmlu_pro": TaskSpec("mmlu_pro", {LANE_BEHAVIOR: evals.run_mmlu_pro_eval}),
    # LongBench family (generation-based).
    "narrativeqa": TaskSpec("narrativeqa", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "qasper": TaskSpec("qasper", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "multifieldqa_en": TaskSpec("multifieldqa_en", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "hotpotqa": TaskSpec("hotpotqa", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "2wikimqa": TaskSpec("2wikimqa", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "musique": TaskSpec("musique", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "gov_report": TaskSpec("gov_report", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "qmsum": TaskSpec("qmsum", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "multi_news": TaskSpec("multi_news", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "trec": TaskSpec("trec", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "triviaqa": TaskSpec("triviaqa", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "samsum": TaskSpec("samsum", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "passage_count": TaskSpec("passage_count", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "passage_retrieval_en": TaskSpec("passage_retrieval_en", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "lcc": TaskSpec("lcc", {LANE_BEHAVIOR: evals.run_longbench_eval}),
    "repobench_p": TaskSpec("repobench_p", {LANE_BEHAVIOR: evals.run_longbench_eval}),
}


def normalize_lane(lane: str) -> str:
    if lane is None:
        return lane
    lane_norm = lane.strip().lower()
    if lane_norm not in SUPPORTED_LANES:
        raise ValueError(f"Unknown lane '{lane}'. Supported lanes: {sorted(SUPPORTED_LANES)}")
    return lane_norm


def get_task_spec(bench_name: str) -> TaskSpec:
    if bench_name not in _TASK_SPECS:
        raise ValueError(f"No TaskSpec configured for benchmark '{bench_name}'")
    return _TASK_SPECS[bench_name]


def get_lane_evaluator(bench_name: str, lane: str) -> Callable:
    lane = resolve_lane_for_task(bench_name, lane)
    task = get_task_spec(bench_name)
    return task.evaluators[lane]


def resolve_lane_for_task(bench_name: str, lane: str | None) -> str:
    task = get_task_spec(bench_name)
    if lane is None:
        if LANE_DISTRIBUTION in task.evaluators:
            return LANE_DISTRIBUTION
        if LANE_BEHAVIOR in task.evaluators:
            return LANE_BEHAVIOR
        raise ValueError(f"Benchmark '{bench_name}' has no configured evaluators.")

    lane = normalize_lane(lane)
    if lane not in task.evaluators:
        supported = sorted(task.evaluators.keys())
        raise ValueError(
            f"Benchmark '{bench_name}' does not support lane '{lane}'. "
            f"Supported lanes: {supported}"
        )
    return lane


def list_supported_lanes(bench_name: str) -> List[str]:
    task = get_task_spec(bench_name)
    return sorted(task.evaluators.keys())


def list_benchmarks_for_lane(lane: str) -> List[str]:
    lane = normalize_lane(lane)
    return sorted(name for name, spec in _TASK_SPECS.items() if lane in spec.evaluators)


def validate_lane_compatibility(bench_list: List[str], lane: str | None) -> None:
    missing_specs = [bench for bench in bench_list if bench not in _TASK_SPECS]
    if missing_specs:
        raise ValueError(
            f"No TaskSpec configured for benchmark(s): {missing_specs}. "
            "Add them to run/pipelines/benchmarks/task_registry.py."
        )

    if lane is None:
        return

    lane = normalize_lane(lane)

    incompatible = []
    for bench in bench_list:
        supported = list_supported_lanes(bench)
        if lane not in supported:
            incompatible.append((bench, supported))

    if incompatible:
        details = ", ".join(f"{bench} (supported: {supported})" for bench, supported in incompatible)
        lane_benchmarks = list_benchmarks_for_lane(lane)
        raise ValueError(
            f"Lane '{lane}' is incompatible with benchmark(s): {details}. "
            f"Benchmarks that support lane '{lane}': {lane_benchmarks}"
        )

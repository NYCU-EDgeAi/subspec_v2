"""Code-generation metrics for LiveCodeBench-style pass@k evaluation."""

# borrowed and extended from
# https://github.com/Naman-ntc/codescratch/blob/main/evaluation/bigcode-evaluation-harness/lm_eval/tasks/custom_metrics/apps_custom_metrics/utils.py

import json
import multiprocessing
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

from .pass_k_utils import compute_metrics_from_results
from .testing_util import run_test


sys.set_int_max_str_digits(50000)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ProblemSample = Dict[str, Any]
GenerationResults = Dict[int, List[List[Any]]]
GenerationMetadata = Dict[int, List[Dict[str, Any]]]


def _run_in_subprocess(
    sample: ProblemSample,
    generation: str,
    debug: bool,
    result,
    metadata_list,
    timeout: int,
) -> None:
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def _num_inputs(sample: ProblemSample) -> int:
    parsed = json.loads(sample["input_output"])
    inputs = parsed.get("inputs", [])
    if isinstance(inputs, list):
        return len(inputs)
    # Some malformed tasks store a single serialized input as string.
    return 1


def _global_timeout_seconds(sample: ProblemSample, per_test_timeout: int) -> int:
    return (per_test_timeout + 1) * _num_inputs(sample) + 5


def _timeout_fallback(sample: ProblemSample) -> Tuple[List[int], Dict[str, Any]]:
    total = _num_inputs(sample)
    return ([-1 for _ in range(total)], {"error_code": -3, "error_message": "GlobalTimeout"})


def _normalize_result_values(values: Iterable[Any]) -> List[Any]:
    normalized: List[Any] = []
    for value in values:
        if isinstance(value, np.ndarray):
            value = value.item(0)
        if isinstance(value, np.bool_):
            value = bool(value)
        normalized.append(value)
    return normalized


def check_correctness(sample: ProblemSample, generation: str, timeout: int, debug: bool = True):
    """Run tests with an outer watchdog timeout around `run_test`."""
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    process = multiprocessing.Process(
        target=_run_in_subprocess,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    process.start()
    process.join(timeout=_global_timeout_seconds(sample, timeout))
    if process.is_alive():
        process.kill()

    if not result:
        if debug:
            print("global timeout")
        return _timeout_fallback(sample)

    metadata = metadata_list[0] if metadata_list else {"error_code": -4, "error_message": "MissingMetadata"}
    return result[0], metadata


def evaluate_generations_by_problem(args):
    problem_generations: List[str] = args[0]
    sample = args[1]
    debug: bool = args[2]
    timeout: int = args[3]

    results_for_problem: List[List[Any]] = []
    metadata_for_problem: List[Dict[str, Any]] = []

    for generation_idx, generation in enumerate(problem_generations):
        curr_res: List[Any] = [-2]
        curr_metadata: Dict[str, Any] = {"error_code": -2, "error_message": "UnknownError"}
        try:
            curr_res, curr_metadata = check_correctness(
                sample,
                generation,
                timeout=timeout,
                debug=debug,
            )
            if debug:
                print(f"\nSuccessful compilation of task {generation_idx}!")
            curr_res = _normalize_result_values(curr_res)
            if not np.all(curr_res) and debug:
                print(f"Results were not True for all test cases {curr_res=}\n")
        except Exception as exc:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(exc)}{exc}\n")
            curr_metadata = {
                "error": repr(exc),
                "error_code": -5,
                "error_message": "TestRunnerError",
            }
        finally:
            assert isinstance(curr_res, list), curr_res
            assert isinstance(curr_metadata, dict), curr_metadata
            results_for_problem.append(curr_res)
            metadata_for_problem.append(curr_metadata)

    if debug:
        for idx, generation in enumerate(problem_generations):
            print("Sample\n")
            print(generation)
            print("\nResult\n")
            print(results_for_problem[idx])
            print("*" * 30 + "\n\n")
    return results_for_problem, metadata_for_problem


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout: int = 6,
) -> Tuple[GenerationResults, GenerationMetadata]:
    """Compile generations and execute unit tests in parallel."""
    jobs = [
        ((generations_list[index], samples_list[index], debug, timeout), index)
        for index in range(len(generations_list))
    ]

    results: GenerationResults = {}
    metadata: GenerationMetadata = {}
    with tqdm(total=len(jobs)) as pbar:
        with ProcessPoolExecutor(max_workers=1 if debug else num_process_evaluate) as executor:
            future_to_index = {
                executor.submit(evaluate_generations_by_problem, job_args): index
                for job_args, index in jobs
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index], metadata[index] = future.result()
                pbar.update(1)

    assert len(results) == len(jobs), f"results = {len(results)} inputs = {len(jobs)} {results=}"
    return results, metadata


def _flatten_generations(
    samples_list: list,
    generations_list: list[list[str]],
) -> Tuple[List[ProblemSample], List[List[str]], List[int]]:
    samples_linear: List[ProblemSample] = []
    generations_linear: List[List[str]] = []
    remap_index: List[int] = []
    for idx, (sample, generation_list) in enumerate(zip(samples_list, generations_list)):
        assert isinstance(generation_list, list), generations_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)
    return samples_linear, generations_linear, remap_index


def _serialize_metadata(metadatas: Dict[int, List[Dict[str, Any]]], expected_per_problem: int) -> List[List[str]]:
    final_metadata: List[List[str]] = []
    for key in sorted(list(metadatas.keys())):
        serialized = [json.dumps(item) for item in metadatas[key]]
        assert len(serialized) == expected_per_problem, f"{len(serialized)=}"
        final_metadata.append(serialized)
    return final_metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
    num_process_evaluate: int = 16,
    timeout: int = 6,
    debug: bool = False,
):
    samples_linear, generations_linear, remap_index = _flatten_generations(samples_list, generations_list)
    print(f"Evaluating {len(samples_linear)}...")

    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        results[remap_index[idx]].append(sub_results[0])
    for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        metadatas[remap_index[idx]].append(sub_metadatas[0])

    metrics = compute_metrics_from_results(results, k_list=k_list)
    final_metadata = _serialize_metadata(metadatas, expected_per_problem=len(generations_list[0]))
    return [metrics, results, final_metadata]


if __name__ == "__main__":
    # print(
    #     check_correctness(
    #         {
    #             "input_output": json.dumps(
    #                 {
    #                     "inputs": [
    #                         json.dumps([1] * 100000)
    #                         + "\n"
    #                         + json.dumps([100000, -100000] * (100000 // 2))
    #                     ],
    #                     "outputs": [json.dumps([100000, 0] * (100000 // 2))],
    #                     "fn_name": "mostFrequentIDs",
    #                 }
    #             )
    #         },
    #         "class Solution:\n    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:\n        from collections import defaultdict\n        \n        # Count of each ID\n        count = defaultdict(int)\n        # How many IDs exist for a given frequency\n        freq_of_count = defaultdict(int)\n        \n        max_freq = 0\n        ans = []\n        \n        for i in range(len(nums)):\n            x = nums[i]\n            change = freq[i]\n            \n            old_freq = count[x]\n            new_freq = old_freq + change\n            \n            # If there was an old frequency, decrease its usage\n            if old_freq > 0:\n                freq_of_count[old_freq] -= 1\n                if freq_of_count[old_freq] == 0:\n                    del freq_of_count[old_freq]\n            \n            # Update with the new frequency\n            count[x] = new_freq\n            freq_of_count[new_freq] += 1\n            \n            # Update max_freq if needed\n            if new_freq > max_freq:\n                max_freq = new_freq\n            \n            # If the collection at max_freq is empty, reduce max_freq until we find a non-empty bin\n            while max_freq > 0 and max_freq not in freq_of_count:\n                max_freq -= 1\n            \n            # If the collection is empty, max_freq will be 0\n            ans.append(max_freq)\n        \n        return ans",
    #         6,
    #         debug=True,
    #     )
    # )

    print(
        check_correctness(
            {
                "input_output": json.dumps(
                    {
                        "inputs": ")))))",
                        "outputs": "0",
                    },
                )
            },
            "\nMOD = 998244353\n\nS = input().strip()\nn = len(S)\n\nif n % 2 != 0:\n    print(0)\n    exit()\n\n# Initialize DP table\ndp = [[0] * (n + 2) for _ in range(n + 1)]\ndp[0][0] = 1\n\nfor i in range(1, n + 1):\n    c = S[i-1]\n    for b in range(n + 1):\n        if dp[i-1][b] == 0:\n            continue\n        if c == '(':\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        elif c == ')':\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        else:  # '?'\n            # Replace with '('\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n            # Replace with ')'\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n\nprint(dp[n][0] % MOD)\n",
            6,
            debug=True,
        )
    )

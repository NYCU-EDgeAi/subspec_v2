# Evaluation follows LMSYS MT-Bench (FastChat question set).
# https://github.com/lm-sys/FastChat
import json

QUERY_TEMPLATE = """
{Prompt}
""".strip()

# MTBENCH
def load_mtbench_dataset():
    with open("run/pipelines/benchmarks/data/mt_bench.jsonl") as f:
        dataset = [json.loads(line)['turns'] for line in f]  # list of list of prompts
    return dataset


def load_mtbench_dataset_answer():
    # MT-Bench is generation-only for this pipeline. Keep API parity with
    # `with_answers=True` loaders by returning the same turn-structured data.
    return load_mtbench_dataset()

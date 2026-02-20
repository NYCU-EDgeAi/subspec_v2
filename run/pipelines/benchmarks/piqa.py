# Evaluation follows EleutherAI lm-evaluation-harness (piqa task).
# https://github.com/EleutherAI/lm-evaluation-harness
from datasets import load_dataset

DATASET_PATH = "baber/piqa"
DATASET_REVISION = "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1"

QUERY_TEMPLATE = "Question: {goal}\nAnswer:"


def load_piqa_dataset():
    dataset = load_dataset(DATASET_PATH, split="validation", revision=DATASET_REVISION)
    return [QUERY_TEMPLATE.format(goal=entry["goal"]) for entry in dataset]


def load_piqa_dataset_answer():
    dataset = load_dataset(DATASET_PATH, split="validation", revision=DATASET_REVISION)
    examples = []
    for entry in dataset:
        answer_index = int(entry["label"])
        examples.append({
            "context": QUERY_TEMPLATE.format(goal=entry["goal"]),
            "choices": [entry["sol1"], entry["sol2"]],
            "answer_index": answer_index,
            "target_delimiter": " ",
        })
    return examples

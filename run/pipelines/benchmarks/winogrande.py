# Evaluation follows EleutherAI lm-evaluation-harness (winogrande task).
# https://github.com/EleutherAI/lm-evaluation-harness
from datasets import load_dataset

DATASET_PATH = "allenai/winogrande"
DATASET_NAME = "winogrande_xl"
DATASET_REVISION = "01e74176c63542e6b0bcb004dcdea22d94fb67b5"


def _format_example_ll(entry):
    sentence = entry["sentence"]
    idx = sentence.index("_")
    prefix = sentence[:idx]
    suffix = sentence[idx + 1:].strip()
    contexts = [prefix + entry["option1"], prefix + entry["option2"]]
    answer_index = 0 if entry["answer"] == "1" else 1
    return contexts, suffix, answer_index


def load_winogrande_dataset():
    dataset = load_dataset(DATASET_PATH, DATASET_NAME, split="validation", revision=DATASET_REVISION)
    return [entry["sentence"] for entry in dataset]


def load_winogrande_dataset_answer():
    dataset = load_dataset(DATASET_PATH, DATASET_NAME, split="validation", revision=DATASET_REVISION)
    examples = []
    for entry in dataset:
        contexts, suffix, answer_index = _format_example_ll(entry)
        examples.append({
            "contexts": contexts,
            "target": suffix,
            "answer_index": answer_index,
            "multiple_input": True,
            "target_delimiter": " ",
        })
    return examples

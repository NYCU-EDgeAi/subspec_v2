# Evaluation follows EleutherAI lm-evaluation-harness (hellaswag task + preprocess).
# https://github.com/EleutherAI/lm-evaluation-harness
import re
from datasets import load_dataset

DATASET_PATH = "Rowan/hellaswag"
DATASET_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"


def _preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _format_example_ll(entry):
    ctx = entry["ctx_a"] + " " + entry["ctx_b"].capitalize()
    query = _preprocess(entry["activity_label"] + ": " + ctx)
    choices = [_preprocess(ending) for ending in entry["endings"]]
    answer_index = int(entry["label"])
    return query, choices, answer_index


def load_hellaswag_dataset():
    dataset = load_dataset(DATASET_PATH, split="validation", revision=DATASET_REVISION)
    return [_format_example_ll(entry)[0] for entry in dataset]


def load_hellaswag_dataset_answer():
    dataset = load_dataset(DATASET_PATH, split="validation", revision=DATASET_REVISION)
    examples = []
    for entry in dataset:
        query, choices, answer_index = _format_example_ll(entry)
        examples.append({
            "context": query,
            "choices": choices,
            "answer_index": answer_index,
            "target_delimiter": " ",
        })
    return examples

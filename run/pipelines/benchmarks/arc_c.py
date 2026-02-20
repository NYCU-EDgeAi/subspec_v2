# Evaluation follows EleutherAI lm-evaluation-harness (arc_challenge task).
# https://github.com/EleutherAI/lm-evaluation-harness
from datasets import load_dataset

DATASET_PATH = "allenai/ai2_arc"
DATASET_NAME = "ARC-Challenge"
DATASET_REVISION = "210d026faf9955653af8916fad021475a3f00453"


def _normalize_answer_key(answer_key, choice_labels):
    if answer_key in choice_labels:
        return choice_labels.index(answer_key)
    if isinstance(answer_key, str) and answer_key.isdigit():
        idx = int(answer_key) - 1
        if 0 <= idx < len(choice_labels):
            return idx
    return None


def _format_example(entry):
    question_field = entry.get("question")
    if isinstance(question_field, dict):
        question = question_field.get("stem", "")
        choices_field = question_field.get("choices", [])
    else:
        question = question_field or entry.get("question_stem", "")
        choices_field = entry.get("choices", {})

    if isinstance(choices_field, dict):
        choice_texts = choices_field.get("text", [])
        choice_labels = choices_field.get("label", [])
    else:
        choice_texts = [choice.get("text", "") for choice in choices_field]
        choice_labels = [choice.get("label", "") for choice in choices_field]

    answer_key = entry.get("answerKey", "")
    answer_idx = _normalize_answer_key(answer_key, choice_labels)
    if answer_idx is None:
        return None

    return question, choice_texts, answer_idx


def load_arc_c_dataset():
    dataset = load_dataset(DATASET_PATH, DATASET_NAME, split="test", revision=DATASET_REVISION)
    prompts = []
    for entry in dataset:
        formatted = _format_example(entry)
        if formatted is None:
            continue
        question = formatted[0]
        prompts.append(f"Question: {question}\nAnswer:")
    return prompts


def load_arc_c_dataset_answer():
    dataset = load_dataset(DATASET_PATH, DATASET_NAME, split="test", revision=DATASET_REVISION)
    examples = []
    for entry in dataset:
        formatted = _format_example(entry)
        if formatted is None:
            continue
        question, options, answer_idx = formatted
        examples.append({
            "context": f"Question: {question}\nAnswer:",
            "choices": options,
            "answer_index": answer_idx,
            "target_delimiter": " ",
        })
    return examples

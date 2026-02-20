# Evaluation follows OpenAI human-eval reference implementation.
# https://github.com/openai/human-eval
from datasets import load_dataset

QUERY_TEMPLATE = "{Question}"

# HUMANEVAL
def load_humaneval_dataset():
    dataset = load_dataset("openai/openai_humaneval")
    formatted_dataset = [QUERY_TEMPLATE.format(Question=entry["prompt"]) for entry in dataset["test"]]
    return formatted_dataset


def load_humaneval_dataset_answer():
    dataset = load_dataset("openai/openai_humaneval")
    examples = []
    for entry in dataset["test"]:
        prompt = QUERY_TEMPLATE.format(Question=entry["prompt"])
        examples.append({
            "question": prompt,
            "prompt": entry["prompt"],
            "test": entry["test"],
            "entry_point": entry["entry_point"],
            "task_id": entry["task_id"],
        })
    return examples

# Evaluation follows EleutherAI lm-evaluation-harness (humaneval / humaneval_instruct).
# https://github.com/EleutherAI/lm-evaluation-harness
# https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/humaneval/humaneval_instruct.yaml
from datasets import load_dataset

QUERY_TEMPLATE = "{Question}"
INSTRUCT_DOC_TO_TEXT = (
    "Write a solution to the following problem and make sure that it passes the tests:\n"
    "```python\n"
    "{prompt}\n"
    "```\n"
)
INSTRUCT_GEN_PREFIX = (
    "Here is the completed function:\n"
    "```python\n"
    "{prompt}\n"
)
MAX_GEN_TOKS = 1024
STOP_STRINGS = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]


def _build_instruct_prompt(prompt: str) -> str:
    return f"{INSTRUCT_DOC_TO_TEXT.format(prompt=prompt)}{INSTRUCT_GEN_PREFIX.format(prompt=prompt)}"

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
            "prompt_input": entry["prompt"],
            "prediction_style": "plain",
            "test": entry["test"],
            "entry_point": entry["entry_point"],
            "task_id": entry["task_id"],
        })
    return examples


def load_humaneval_instruct_dataset():
    dataset = load_dataset("openai/openai_humaneval")
    return [_build_instruct_prompt(entry["prompt"]) for entry in dataset["test"]]


def load_humaneval_instruct_dataset_answer():
    dataset = load_dataset("openai/openai_humaneval")
    examples = []
    for entry in dataset["test"]:
        base_prompt = entry["prompt"]
        examples.append({
            "question": _build_instruct_prompt(base_prompt),
            "prompt": base_prompt,
            "prompt_input": _build_instruct_prompt(base_prompt),
            "prediction_style": "instruct",
            "test": entry["test"],
            "entry_point": entry["entry_point"],
            "task_id": entry["task_id"],
        })
    return examples

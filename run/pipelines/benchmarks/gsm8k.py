# Evaluation follows OpenAI grade-school-math extraction (GSM8K).
# https://github.com/openai/grade-school-math

import re

from datasets import load_dataset

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

QUERY_TEMPLATE = """
Solve this math problem. Show your reasoning.
On the final line, write exactly: #### <final numeric answer>

{Question}
""".strip()


def extract_answer(text: str) -> str:
    """Match grade-school-math extraction logic for GSM8K."""
    match = ANS_RE.search(text)
    if match:
        return match.group(1).strip().replace(",", "")
    return INVALID_ANS


def is_correct(model_completion: str, gt_answer: str) -> bool:
    """Match grade-school-math is_correct logic."""
    gt = extract_answer(gt_answer)
    if gt == INVALID_ANS:
        return False
    return extract_answer(model_completion) == gt


# GSM8K
def load_gsm8k_dataset():
    dataset = load_dataset("openai/gsm8k", "main")
    formatted_dataset = [QUERY_TEMPLATE.format(Question=entry['question']) for entry in dataset['test']]
    return formatted_dataset


def load_gsm8k_dataset_answer():
    raw = load_dataset("openai/gsm8k", "main")['test']
    examples = []
    for entry in raw:
        q_str = QUERY_TEMPLATE.format(Question=entry['question'])
        a_str = entry['answer']
        examples.append({
            "question": q_str,
            "answer": a_str
        })
    return examples

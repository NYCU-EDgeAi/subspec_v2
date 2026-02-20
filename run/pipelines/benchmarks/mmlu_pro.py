# Evaluation follows TIGER-AI-Lab/MMLU-Pro (evaluate_from_local.py + initial_prompt.txt).
# https://github.com/TIGER-AI-Lab/MMLU-Pro
from datasets import load_dataset

DATASET_PATH = "TIGER-Lab/MMLU-Pro"
DATASET_REVISION = "527feea0afed1de15a8c115abf7be4c912123315"

CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
INITIAL_PROMPT = (
    "The following are multiple choice questions (with answers) about {$}. "
    "Think step by step and then finish your answer with \"the answer is (X)\" where X "
    "is the correct letter choice.\n\n"
)


def _preprocess(dataset):
    res = []
    for each in dataset:
        options = [opt for opt in each["options"] if opt != "N/A"]
        item = dict(each)
        item["options"] = options
        res.append(item)
    return res


def _select_by_category(dataset, subject):
    return [each for each in dataset if each["category"] == subject]


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    prompt += example["question"] + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(example["options"]):
        prompt += "{}. {}\n".format(CHOICES[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.",
            "Answer: Let's think step by step.",
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    subject = curr["category"]
    prompt = INITIAL_PROMPT.replace("{$}", subject) + "\n"
    val_df = _select_by_category(val_df, subject)[:k]
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def load_mmlu_pro_splits():
    dataset = load_dataset(DATASET_PATH, revision=DATASET_REVISION)
    test_df, val_df = dataset["test"], dataset["validation"]
    return _preprocess(test_df), _preprocess(val_df)


# MMLU-PRO
def load_mmlu_pro_dataset():
    test_df, val_df = load_mmlu_pro_splits()
    return [generate_cot_prompt(val_df, ex, k=5) for ex in test_df]


def load_mmlu_pro_dataset_answer():
    test_df, _ = load_mmlu_pro_splits()
    examples = []
    for ex in test_df:
        answer_index = int(ex["answer_index"])
        examples.append({
            "category": ex["category"],
            "question": ex["question"],
            "options": ex["options"],
            "answer": CHOICES[answer_index],
            "answer_index": answer_index,
        })
    return examples

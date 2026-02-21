# Evaluation follows EleutherAI lm-evaluation-harness (gsm8k task).
# https://github.com/EleutherAI/lm-evaluation-harness

import re
import random

from datasets import load_dataset

ANS_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
INVALID_ANS = "[invalid]"
FEWSHOT_K = 5
FEWSHOT_SEED = 1234
MAX_GEN_TOKS = 1024
STOP_STRINGS = ["Question:", "</s>", "<|im_end|>"]
EM_REGEXES_TO_IGNORE = [r",", r"\$", r"(?s).*#### ", r"\.$"]
EM_IGNORE_CASE = True

QUERY_TEMPLATE = "Question: {question}\nAnswer:"


def _normalize_answer(text: str) -> str:
    return text.strip().replace(",", "").replace("$", "")


def normalize_for_exact_match(text: str) -> str:
    """Apply lm-eval exact_match normalization rules for GSM8K."""
    out = str(text or "")
    for pattern in EM_REGEXES_TO_IGNORE:
        out = re.sub(pattern, "", out)
    if EM_IGNORE_CASE:
        out = out.lower()
    return out.strip()


def exact_match(candidate: str, reference: str) -> bool:
    return normalize_for_exact_match(candidate) == normalize_for_exact_match(reference)


def _build_query(question: str, fewshot_examples: list[dict]) -> str:
    parts = []
    for ex in fewshot_examples:
        parts.append(QUERY_TEMPLATE.format(question=ex["question"]))
        parts.append(ex["answer"])
    parts.append(QUERY_TEMPLATE.format(question=question))
    return "\n\n".join(parts)


def extract_answer(text: str) -> str:
    """LM-eval-style flexible extractor for GSM8K."""
    matches = ANS_RE.findall(text or "")
    if not matches:
        return INVALID_ANS
    match = matches[-1]
    if isinstance(match, tuple):
        non_empty = [m for m in match if m]
        if not non_empty:
            return INVALID_ANS
        return _normalize_answer(non_empty[0])
    return _normalize_answer(str(match))


# GSM8K
def load_gsm8k_dataset():
    dataset = load_dataset("openai/gsm8k", "main")
    train_split = list(dataset["train"])
    rng = random.Random(FEWSHOT_SEED)
    formatted_dataset = []
    for entry in dataset["test"]:
        fewshot = rng.sample(train_split, FEWSHOT_K)
        formatted_dataset.append(_build_query(entry["question"], fewshot))
    return formatted_dataset


def load_gsm8k_dataset_answer():
    dataset = load_dataset("openai/gsm8k", "main")
    train_split = list(dataset["train"])
    rng = random.Random(FEWSHOT_SEED)
    examples = []
    for entry in dataset["test"]:
        fewshot = rng.sample(train_split, FEWSHOT_K)
        q_str = _build_query(entry["question"], fewshot)
        a_str = entry["answer"]
        examples.append({
            "question": q_str,
            "answer": a_str
        })
    return examples

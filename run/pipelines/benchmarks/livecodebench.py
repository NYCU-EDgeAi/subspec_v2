# Evaluation follows LiveCodeBench lcb_runner (code_generation).
# https://github.com/LiveCodeBench/LiveCodeBench
from datasets import load_dataset
from .utils.lcb_runner.benchmarks.code_generation import CodeGenerationProblem

FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem and "
    "enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters "
    "as follows. Ensure that when the python program runs, it reads the inputs, runs the "
    "algorithm and writes output to STDOUT."
)


def _format_prompt(problem: CodeGenerationProblem) -> str:
    prompt = f"### Question:\n{problem.question_content}\n\n"
    if problem.starter_code:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt

# LIVECODEBENCH
def load_livecodebench_dataset():

    dataset = load_dataset("livecodebench/code_generation_lite", "v4_v5", trust_remote_code=True) # problems released between Aug 2024 and Jan 2025. The deepseek eval dataset setting
    problems = [CodeGenerationProblem(**entry) for entry in dataset["test"]]
    formatted_dataset = [_format_prompt(problem) for problem in problems]

    return formatted_dataset

def load_livecodebench_dataset_answer():
    """
    Returns a list of dicts like:
    {
    "question": <formatted prompt>,
    "problem": <CodeGenerationProblem>,
    "eval_sample": <input_output payload>
    }
    which is what run_livecodebench_eval above expects.
    """
    ds = load_dataset("livecodebench/code_generation_lite", "v4_v5", trust_remote_code=True)['test']
    examples = []
    for row in ds:
        problem = CodeGenerationProblem(**row)
        q = _format_prompt(problem)
        eval_sample = problem.get_evaluation_sample()
        examples.append({
            "question": q,
            "problem": problem,
            "eval_sample": eval_sample,
        })
    return examples


# def load_livecodebench_dataset_answer():
#     """
#     Returns a list of dicts, each containing:
#     - 'prompt' : the templated question string
#     - 'public_tests' : JSON string of public testcases
#     - 'private_tests': JSON string of hidden testcases
#     - 'starter_code' : any provided starter code string (may be empty)
#     - 'difficulty' : problem difficulty tag
#     - 'metadata' : extra metadata
#     """
#     ds = load_dataset(
#     "livecodebench/code_generation_lite",
#     split="test",
#     version_tag="v4_v5",
#     trust_remote_code=True
#     )
#     examples = []
#     for ex in ds:
#         examples.append({
#             "prompt": QUERY_TEMPLATE.format(Question=ex["question_content"]),
#             "public_tests" : ex["public_test_cases"],
#             "private_tests": ex["private_test_cases"],
#             "starter_code" : ex["starter_code"],
#             "difficulty"   : ex["difficulty"],
#             "metadata"     : ex["metadata"],
#         })
#     return examples

"""Behavioral safety net for the Phase-3 backend-seam refactor.

The refactor extracts a `SpecDecodeBackend` and rewrites the per-method `_generate`
loop to drive it. These tests lock in the CURRENT generator output so the refactor
cannot silently change tokens.

Goldens (Llama-3.2-1B, greedy, prompt below), all verified reproducible:
  - subspec_sd       (v1 SDPA)       "The capital of France is Paris."
  - subspec_sd_fi    (v1 FlashInfer) "The capital of France is Paris."
  - subspec_sd_v2    (v2 SDPA)       "The capital of France is Paris."
  - subspec_sd_v2_fi (v2 FlashInfer) "The capital of France is Paris."

subspec_sd_fi previously produced garbage on 1B: its target tree-rewrite window did
not grow the request cache to the full tree footprint (the draft appends fewer KV
slots than the tree has nodes), so the window ate into committed prefix KV. Fixed in
generators/subspec_sd_fi.py by syncing the request cache to `position_offset +
num_tokens` before the rewrite (mirrors subspec_sd_v2_fi's `_sync_request_cache_to_len`).

SDPA and FlashInfer diverge slightly by design (different attention kernels), so these
are per-backend goldens, NOT a cross-backend equality assertion. FlashInfer goldens are
also max_length-sensitive (paged-cache layout), so `n_new` is pinned.

Isolation: each real-model build runs in a FRESH SUBPROCESS. Building 3+ quantized
models in one process corrupts later builds (global HQQ/gemlite/dynamo state that
torch.cuda.empty_cache() cannot clear), so the goldens must not share a process.

Gated behind SUBSPEC_RUN_REAL_MODEL_TESTS=1 (repo convention for real-weights tests)
and skipped without CUDA.
"""
import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("SUBSPEC_RUN_REAL_MODEL_TESTS") != "1",
        reason="Set SUBSPEC_RUN_REAL_MODEL_TESTS=1 to enable real-model tests",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

GOLDEN_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
GOLDEN_PROMPT = "What is the capital of France?"
GOLDENS = {
    "subspec_sd":       [791, 6864, 315, 9822, 374, 12366, 13, 128009],   # "...is Paris."
    "subspec_sd_fi":    [791, 6864, 315, 9822, 374, 12366, 13, 128009],   # "...is Paris." (fixed)
    "subspec_sd_v2":    [791, 6864, 315, 9822, 374, 12366, 13, 128009],   # "...is Paris."
    "subspec_sd_v2_fi": [791, 6864, 315, 9822, 374, 12366, 13, 128009],   # "...is Paris."
}


def _greedy_new_ids(method: str, *, llm_path: str, prompt: str, n_new: int = 32) -> list[int]:
    """Build `method` and return greedy new token ids, mirroring run_test.main's
    warmup + cuda-graph init sequence (required for the FlashInfer path).

    Intended to run in its own process (see __main__); do not call from an in-process
    test that also builds other quantized models."""
    from run.core.registry import ModelRegistry
    from run.core.presets import register_presets
    from run.core.configuration import AppConfig
    from run.core.builder import GeneratorPipelineBuilder
    from specdecodes.models.utils.utils import DraftParams
    from run.pipelines.utils.eval_utils import reset_kv, maybe_init_cuda_graph_runner
    from torch.nn.attention import SDPBackend, sdpa_kernel

    register_presets()
    entry = ModelRegistry.get(method)
    assert entry is not None, f"method {method} is not registered"

    cfg = AppConfig()
    cfg.method = method
    cfg.update(entry.default_config)          # recipe + defaults
    cfg.llm_path = llm_path
    cfg.device = "cuda:0"
    cfg.dtype = torch.float16
    cfg.cache_implementation = "static"
    cfg.compile_mode = None                   # skip compile: fast + deterministic
    cfg.warmup_iter = 0
    cfg.do_sample = False
    cfg.temperature = 0.0
    cfg.generator_profiling = False
    cfg.draft_params = DraftParams(temperature=0.2, max_depth=32, topk_len=6)
    cfg.generator_kwargs = {"prefill_chunk_size": 256, "verify_method": "exact", "verify_kwargs": {}}

    generator, tokenizer, past_kv, draft_past_kv = GeneratorPipelineBuilder(cfg).build()
    generator.profiling = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(cfg.device)
    max_length = int(input_ids.shape[1]) + n_new

    def _gen():
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            return generator.generate(
                input_ids, temperature=0.0, max_length=max_length, do_sample=False,
                past_key_values=past_kv, draft_past_key_values=draft_past_kv,
            )

    _gen()                                                   # warmup
    reset_kv(past_kv, draft_past_kv)
    maybe_init_cuda_graph_runner(generator, past_kv, draft_past_kv, cfg.device, 1)
    out = _gen()                                             # measured
    return out[0][input_ids.shape[1]:].tolist()


@pytest.mark.parametrize("method", list(GOLDENS))
def test_greedy_output_is_stable(method):
    """Each backend must keep producing its golden greedy continuation through the refactor."""
    if method.endswith("_fi"):
        pytest.importorskip("flashinfer")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, SUBSPEC_RUN_REAL_MODEL_TESTS="1")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), method],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=900,
    )
    new_ids = None
    for line in proc.stdout.splitlines():
        if line.startswith("NEW_IDS:"):
            new_ids = json.loads(line.split("NEW_IDS:", 1)[1].strip())
    assert new_ids is not None, (
        f"runner produced no NEW_IDS for {method} (rc={proc.returncode})\n"
        f"--- stderr tail ---\n{proc.stderr[-3000:]}"
    )
    assert new_ids == GOLDENS[method]


if __name__ == "__main__":
    # Isolated runner: `python tests/test_backend_seam_parity.py <method>` -> prints NEW_IDS.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _method = sys.argv[1]
    _ids = _greedy_new_ids(_method, llm_path=GOLDEN_MODEL, prompt=GOLDEN_PROMPT)
    print("NEW_IDS:", json.dumps(_ids))

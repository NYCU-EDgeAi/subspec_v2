"""Behavioral safety net for the Phase-3 backend-seam refactor.

The refactor extracts a `SpecDecodeBackend` and rewrites the per-method `_generate`
loop to drive it. These tests lock in the CURRENT generator output so the refactor
cannot silently change tokens.

Scope note (which variants have e2e goldens, and why):
- The v2 algorithm works correctly with a small model (Llama-3.2-1B) for BOTH
  backends, so both get real e2e goldens here:
    * subspec_sd_v2    (SDPA)       -> "The capital of France is Paris."
    * subspec_sd_v2_fi (FlashInfer) -> "The capital of France is currently Paris."
  (SDPA and FI diverge slightly by design — different attention kernels — so these
  are per-backend goldens, NOT a cross-backend equality assertion.)
- subspec_sd (v1 SDPA) also decodes cleanly on 1B and is pinned too.
- subspec_sd_fi (v1 FlashInfer) produces GARBAGE on 1B even via the real run-test
  entrypoint, so it has no cheap e2e golden; it is guarded by the existing
  dummy-model unit tests (test_subspec_fi_*, test_fi_request_cache_reuse).

The FlashInfer goldens require the same warmup + cuda-graph init sequence that
run_test.main performs, so `_greedy_new_ids` mirrors it exactly.

Gated behind SUBSPEC_RUN_REAL_MODEL_TESTS=1 (matches the repo convention for tests
that load real weights) and skipped without CUDA.
"""
import os

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("SUBSPEC_RUN_REAL_MODEL_TESTS") != "1",
        reason="Set SUBSPEC_RUN_REAL_MODEL_TESTS=1 to enable real-model tests",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

# A small, locally-cached model that decodes deterministically under greedy SD.
GOLDEN_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
GOLDEN_PROMPT = "What is the capital of France?"

# Golden greedy continuations captured from the CURRENT generators (commit prior to
# the backend-seam extraction), Llama-3.2-1B, prompt above, warmup+cudagraph setup.
# NOTE: FlashInfer goldens are sensitive to max_length (paged-cache layout affects
# kernel numerics), so `n_new` is pinned in `_greedy_new_ids`. Under that fixed config
# all three are deterministic (verified reproducible run-to-run).
GOLDENS = {
    "subspec_sd":       [791, 6864, 315, 9822, 374, 12366, 13, 128009],   # "...is Paris."
    "subspec_sd_v2":    [791, 6864, 315, 9822, 374, 12366, 13, 128009],   # "...is Paris."
    "subspec_sd_v2_fi": [791, 6864, 315, 9822, 374, 12366, 13, 128009],   # "...is Paris."
}


def _greedy_new_ids(method: str, *, llm_path: str, prompt: str, n_new: int = 32) -> list[int]:
    """Build `method` and return greedy new token ids, mirroring run_test.main's
    warmup + cuda-graph init sequence (required for the FlashInfer path)."""
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
    new_ids = _greedy_new_ids(method, llm_path=GOLDEN_MODEL, prompt=GOLDEN_PROMPT)
    assert new_ids == GOLDENS[method]

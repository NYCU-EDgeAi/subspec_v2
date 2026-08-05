"""Behavioral safety net for the Phase-3 backend-seam refactor.

The refactor extracts a `SpecDecodeBackend` and rewrites the per-method `_generate`
loop to drive it. These tests lock in the CURRENT generator output so the refactor
cannot silently change tokens.

Scope note (why SDPA only, for now):
- SDPA (`subspec_sd`) greedy decoding is deterministic and works with a small model
  on CPU-less CUDA without compile — an easy, fast golden captured here.
- FlashInfer (`subspec_sd_fi`) produces garbage with Llama-3.2-1B even via the real
  `run-test` entrypoint (kernel/model-config incompatibility at this size), and the
  8B+compile path is minutes-slow — so there is no cheap FI e2e golden. The FI path
  is instead guarded by the existing dummy-model unit tests (test_subspec_fi_*,
  test_fi_request_cache_reuse) plus the backend-adapter unit tests added alongside
  the extraction. See refactor/backend-seam notes.

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

# Golden greedy continuation captured from the current `subspec_sd` generator
# (commit prior to the backend-seam extraction). "The capital of France is Paris.<eot>"
GOLDEN_SUBSPEC_SD_IDS = [791, 6864, 315, 9822, 374, 12366, 13, 128009]


def _greedy_new_ids(method: str, *, llm_path: str, prompt: str, n_new: int = 32) -> list[int]:
    """Build `method` with a fast/deterministic config and return greedy new token ids."""
    from run.core.registry import ModelRegistry
    from run.core.presets import register_presets
    from run.core.configuration import AppConfig
    from run.core.builder import GeneratorPipelineBuilder
    from specdecodes.models.utils.utils import DraftParams
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

    with sdpa_kernel(backends=[SDPBackend.MATH]):
        out = generator.generate(
            input_ids, temperature=0.0, max_length=max_length, do_sample=False,
            past_key_values=past_kv, draft_past_key_values=draft_past_kv,
        )
    return out[0][input_ids.shape[1]:].tolist()


def test_subspec_sd_greedy_output_is_stable():
    """SDPA backend must keep producing the golden greedy continuation through the refactor."""
    new_ids = _greedy_new_ids("subspec_sd", llm_path=GOLDEN_MODEL, prompt=GOLDEN_PROMPT)
    assert new_ids == GOLDEN_SUBSPEC_SD_IDS

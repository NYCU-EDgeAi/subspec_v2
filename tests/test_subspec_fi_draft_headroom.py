from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flashinfer")

from specdecodes.models.draft_models.subspec_sd_fi import SubSpecSDDraftModel


class _DummyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(8, 8, bias=False)
        self.config = SimpleNamespace(
            num_attention_heads=1,
            num_key_value_heads=1,
            hidden_size=8,
        )

    def forward(self, input_ids, *args, **kwargs):
        batch, seqlen = input_ids.shape
        logits = torch.ones((batch, seqlen, 16), dtype=torch.float32, device=input_ids.device)
        return SimpleNamespace(logits=logits)


class _DummyWrapper:
    def prepareAttention(self, *args, **kwargs):
        return None


class _DummyPool:
    def __init__(self):
        self.page_len = 6
        self.max_pages = 1
        self.cache_data = torch.zeros((1, 1, 2, 1, 1, 1), dtype=torch.float16)


class _DummyRequestKvCache:
    def __init__(self):
        self.kv_len = 0
        self.kvCachePool = _DummyPool()

    def increment(self, num_tokens: int = 1):
        self.kv_len += int(num_tokens)

    def get_seq_length(self):
        return int(self.kv_len)


def test_speculate_does_not_add_tree_nodes_without_kv_headroom(monkeypatch):
    monkeypatch.setattr(
        "specdecodes.models.draft_models.subspec_sd_fi.getKvCacheBatchPosition",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    model = SubSpecSDDraftModel(base_model=_DummyLM(), eos_token_id=2)
    model.draft_params = SimpleNamespace(topk_len=3, max_depth=2, temperature=1.0)
    model.flashinferWrapper = _DummyWrapper()

    def _fake_topk_sampling(sampled_probs, parent_probs, sample_k):
        token_ids = torch.arange(sample_k, dtype=torch.long).unsqueeze(0)
        child_probs = torch.ones((1, sample_k), dtype=sampled_probs.dtype, device=sampled_probs.device)
        parent_indices = torch.zeros((1, sample_k), dtype=torch.long, device=sampled_probs.device)
        return token_ids, child_probs, parent_indices

    model.topk_sampling = _fake_topk_sampling
    request_kv_cache = _DummyRequestKvCache()

    tree = model.speculate(torch.tensor([[1]], dtype=torch.long), request_kv_cache)

    # Limit is 6 tokens. Sequence grows:
    # decode +1 => 1, first tree level forward +3 => 4. The next level can still be
    # sampled into the tree, but no additional KV append forward is allowed.
    assert int(request_kv_cache.get_seq_length()) == 4
    assert int(tree.size()) == 7

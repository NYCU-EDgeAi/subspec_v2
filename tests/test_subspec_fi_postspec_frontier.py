from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flashinfer")

from specdecodes.models.draft_models.subspec_sd_fi import SubSpecSDDraftModel
from specdecodes.models.draft_models.base import TreeMaskCache
from specdecodes.models.utils.cpu_tree import Tree


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


class _DummyRequestKvCache:
    def __init__(self, seq_len: int):
        self._seq_len = int(seq_len)

    def get_seq_length(self):
        return int(self._seq_len)


def _build_tree() -> Tree:
    tree = Tree(torch.tensor(10, dtype=torch.long), prob_dtype=torch.float32)
    token_ids = torch.tensor([[[20, 21], [30, 31]]], dtype=torch.long)
    child_probs = torch.tensor([[[0.6, 0.4], [0.36, 0.24]]], dtype=torch.float32)
    parent_indices = torch.tensor([[[0, 0], [0, 1]]], dtype=torch.long)
    tree.add_nodes(token_ids, child_probs, parent_indices)
    return tree


def _build_wide_tree() -> Tree:
    tree = Tree(torch.tensor(10, dtype=torch.long), prob_dtype=torch.float32)
    token_ids = torch.tensor([[[11, 12, 13, 14]]], dtype=torch.long)
    child_probs = torch.tensor([[[0.1, 0.7, 0.4, 0.2]]], dtype=torch.float32)
    parent_indices = torch.tensor([[[0, 0, 0, 0]]], dtype=torch.long)
    tree.add_nodes(token_ids, child_probs, parent_indices)
    return tree


def _make_model(max_cache_len):
    model = SubSpecSDDraftModel(base_model=_DummyLM(), eos_token_id=2)
    model.draft_params = SimpleNamespace(topk_len=2, max_depth=2, temperature=1.0)
    model.had_first_speculate = True
    model.tree = _build_tree()
    model.request_kv_cache = _DummyRequestKvCache(seq_len=11)  # prefix_len(7) + tree_size(5) - 1
    model.tree_mask_cache = TreeMaskCache(
        prefix_len=7,
        sample_len=2,
        max_cache_len=max_cache_len,
        dtype=torch.float32,
        device="cpu",
    )
    # Seed stale frontier state to ensure init_postspec rebuilds from tree.
    model.token_ids = torch.tensor([[999, 999]], dtype=torch.long)
    model.parent_probs = torch.tensor([[9.0, 9.0]], dtype=torch.float32)
    model.position_ids = torch.tensor([[999, 999]], dtype=torch.long)
    return model


def _expected_frontier_mask(tree: Tree) -> torch.Tensor:
    leaves = torch.tensor(tree.available_leaves, dtype=torch.long)
    full_mask = tree.create_attention_mask(prefix_length=6, skip_nodes=0, device="cpu")
    return full_mask[:, :, leaves, :]


def test_init_postspec_rebuilds_frontier_dynamic_mask():
    model = _make_model(max_cache_len=None)
    model.init_postspec(rebuild_frontier=True)

    assert model.postspec_count == 0
    assert model.token_ids.cpu().tolist() == [[30, 31]]
    assert model.position_ids.cpu().tolist() == [[8, 8]]
    assert torch.allclose(model.parent_probs.cpu(), torch.tensor([[0.36, 0.24]], dtype=torch.float32))

    expected_mask = _expected_frontier_mask(model.tree)
    assert torch.equal(model.tree_mask_cache.tree_mask_cache.cpu(), expected_mask)


def test_init_postspec_rebuilds_frontier_static_mask():
    model = _make_model(max_cache_len=32)
    model.init_postspec(rebuild_frontier=True)

    assert model.postspec_count == 0
    assert model.token_ids.cpu().tolist() == [[30, 31]]
    assert model.position_ids.cpu().tolist() == [[8, 8]]
    assert int(model.tree_mask_cache.current_len) == 11

    expected_mask = _expected_frontier_mask(model.tree)
    actual_mask = model.tree_mask_cache.tree_mask_cache[:, :, :2, :11].cpu()
    assert torch.equal(actual_mask, expected_mask)


def test_init_postspec_limits_frontier_width_to_topk_len():
    model = _make_model(max_cache_len=None)
    model.tree = _build_wide_tree()
    model.request_kv_cache = _DummyRequestKvCache(seq_len=9)  # prefix_len(5) + tree_size(5) - 1
    model.tree_mask_cache = TreeMaskCache(
        prefix_len=5,
        sample_len=2,
        max_cache_len=None,
        dtype=torch.float32,
        device="cpu",
    )

    model.init_postspec(rebuild_frontier=True)

    assert tuple(model.token_ids.shape) == (1, 2)
    assert model.token_ids.cpu().tolist() == [[11, 12]]
    assert model.position_ids.cpu().tolist() == [[5, 5]]


def test_init_postspec_recomputes_prefix_len_from_request_cache_when_stale():
    model = _make_model(max_cache_len=None)
    # Deliberately poison cached prefix metadata; live request/tree geometry
    # implies prefix_len=7.
    model.tree_mask_cache.prefix_len = 3

    model.init_postspec(rebuild_frontier=True)

    assert int(model.tree_mask_cache.prefix_len) == 7
    assert model.position_ids.cpu().tolist() == [[8, 8]]


def test_init_postspec_default_keeps_live_frontier_state():
    model = _make_model(max_cache_len=None)
    model.init_postspec()

    assert model.postspec_count == 0
    # Default path should not rebuild from tree; keep live frontier tensors.
    assert model.token_ids.cpu().tolist() == [[999, 999]]
    assert model.position_ids.cpu().tolist() == [[999, 999]]


def test_postspec_honors_suspend_flag():
    model = _make_model(max_cache_len=None)
    model._suspend_postspec = True
    model.had_first_speculate = True
    model.postspec_count = 0

    assert model.postspec() is False
    assert int(model.postspec_count) == 0

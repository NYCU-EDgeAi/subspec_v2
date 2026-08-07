"""Unit tests for the v2 post-verify backend logic.

Since the backend-seam collapse, the SDPA/FlashInfer KV-cache + attention methods live
on the `SubSpecV2Backend` adapters (`SdpaV2Backend` / `FlashInferV2Backend`), not on the
generator. These tests build a bare unified generator, wrap it in the relevant adapter,
and exercise the adapter's `_post_verify` / `_draft_tree_decoding` / `_build_rewrite_batch_position`
/ `_commit_seed_postspec_before_post_verify` directly.
"""
from types import SimpleNamespace

import torch

from specdecodes.models.generators.subspec_sd_v2 import SubSpecSDGeneratorBase
from specdecodes.models.generators.subspec_sd_v2_loop import (
    SdpaV2Backend,
    FlashInferV2Backend,
)


class _DummyTree:
    def __init__(self):
        self.pruned_to = None

    def size(self):
        return 3

    def get_tree_data(self, skip_nodes=0):
        start = int(skip_nodes)
        token_ids = torch.arange(start, 3, dtype=torch.long)
        depths = torch.arange(start, 3, dtype=torch.long)
        return {"token_ids": token_ids, "depths": depths}

    def prune_to_depth(self, depth: int):
        self.pruned_to = int(depth)
        return torch.tensor([0, 1, 2], dtype=torch.long)


class _DummyRequestCache:
    def __init__(self, seq_len: int, page_len: int = 16):
        self._seq_len = int(seq_len)
        self.page_len = int(page_len)
        self.kv_page_indices = list(range(max(1, (int(self._seq_len) + self.page_len - 1) // self.page_len)))
        self.kv_last_page_len = 0 if int(self._seq_len) == 0 else ((int(self._seq_len) - 1) % self.page_len) + 1

    def get_seq_length(self):
        return int(self._seq_len)

    def increment(self, num_tokens: int = 1):
        self._seq_len += int(num_tokens)
        self.kv_last_page_len = ((int(self._seq_len) - 1) % self.page_len) + 1
        needed_pages = max(1, (int(self._seq_len) + self.page_len - 1) // self.page_len)
        while len(self.kv_page_indices) < needed_pages:
            self.kv_page_indices.append(len(self.kv_page_indices))

    def decrement(self, num_tokens: int = 1):
        self._seq_len = max(0, int(self._seq_len) - int(num_tokens))
        if int(self._seq_len) == 0:
            self.kv_last_page_len = 0
            self.kv_page_indices = [0]
            return
        self.kv_last_page_len = ((int(self._seq_len) - 1) % self.page_len) + 1
        needed_pages = max(1, (int(self._seq_len) + self.page_len - 1) // self.page_len)
        self.kv_page_indices = self.kv_page_indices[:needed_pages]


class _DummyDraftModel:
    def __init__(self):
        self.init_postspec_calls = 0
        self.postspec_calls = 0
        self.update_calls = 0
        self.init_postspec_kwargs = []

    def init_postspec(self, *args, **kwargs):
        self.init_postspec_calls += 1
        self.init_postspec_kwargs.append(dict(kwargs))

    def postspec(self):
        self.postspec_calls += 1
        return True

    def update_tree_after_post(self):
        self.update_calls += 1
        return self._tree_ref


def _make_bare_generator():
    """A generator instance with just the attributes the adapter methods read."""
    gen = SubSpecSDGeneratorBase.__new__(SubSpecSDGeneratorBase)
    gen.draft_params = SimpleNamespace(max_depth=4)
    gen.draft_model = _DummyDraftModel()
    gen.post_verify_count = 0
    # sampled_tokens has len=3, but accept_len returned by verifier is 2
    gen._verify = (
        lambda *args, **kwargs: (
            torch.tensor([[11, 22, 33]], dtype=torch.long),
            torch.tensor([0, 1, 2], dtype=torch.long),
            (3, 2),
        )
    )
    return gen


def _make_sdpa_backend():
    gen = _make_bare_generator()
    backend = SdpaV2Backend(gen)
    backend._draft_tree_decoding = lambda *args, **kwargs: torch.zeros((1, 3, 10))
    return gen, backend


def _make_fi_backend():
    gen = _make_bare_generator()
    backend = FlashInferV2Backend(gen)

    def _fi_draft_tree_decoding(*args, **kwargs):
        request_kv_cache = args[1]
        decoded_tokens = 3
        if kwargs.get("append_tokens", True):
            request_kv_cache.increment(decoded_tokens)
        return torch.zeros((1, decoded_tokens, 10)), decoded_tokens

    backend._draft_tree_decoding = _fi_draft_tree_decoding
    return gen, backend


def test_subspec_v2_postverify_prunes_by_accept_len():
    gen, backend = _make_sdpa_backend()
    tree = _DummyTree()
    gen.draft_model._tree_ref = tree

    out_tree, kept_old_indices = backend._post_verify(
        tree=tree,
        root_ind=0,
        past_key_values=None,
        position_offset=0,
        cache_position=None,
        last_tree_depth=10,
        skip_nodes=0,
        logits_processor=None,
        device=torch.device("cpu"),
    )

    assert out_tree is tree
    assert kept_old_indices.tolist() == [0, 1, 2]
    assert tree.pruned_to == 12  # last_tree_depth + accept_len(2)
    assert gen.draft_model.init_postspec_calls == 1
    assert gen.draft_model.postspec_calls == 2  # max_depth(4) - accept_len(2)


def test_subspec_v2_fi_postverify_prunes_by_accept_len():
    gen, backend = _make_fi_backend()
    gen.post_verify_count = 0
    tree = _DummyTree()
    gen.draft_model._tree_ref = tree
    request_kv_cache = _DummyRequestCache(seq_len=3)

    out_tree, kept_old_indices = backend._post_verify(
        tree=tree,
        root_ind=0,
        request_kv_cache=request_kv_cache,
        position_offset=0,
        last_tree_depth=10,
        skip_nodes=0,
        logits_processor=None,
        device=torch.device("cpu"),
    )

    assert out_tree is tree
    assert kept_old_indices.tolist() == [0, 1, 2]
    assert tree.pruned_to == 12  # last_tree_depth + accept_len(2)
    assert gen.draft_model.init_postspec_calls == 1
    assert gen.draft_model.postspec_calls == 2  # max_depth(4) - accept_len(2)
    assert gen.post_verify_count == 1


class _PruningTree:
    def __init__(self, size_before: int, size_after: int):
        self._size = int(size_before)
        self._size_after = int(size_after)

    def size(self):
        return int(self._size)

    def get_tree_data(self, skip_nodes=0):
        start = int(skip_nodes)
        token_ids = torch.arange(start, int(self._size), dtype=torch.long)
        depths = torch.arange(start, int(self._size), dtype=torch.long)
        return {"token_ids": token_ids, "depths": depths}

    def prune_to_depth(self, _depth: int):
        self._size = int(self._size_after)
        return torch.arange(int(self._size_after), dtype=torch.long)


class _FixedTree:
    def __init__(self, size: int):
        self._size = int(size)

    def size(self):
        return int(self._size)


class _AssertingDraftModel(_DummyDraftModel):
    def __init__(self, request_kv_cache: _DummyRequestCache, expected_len_at_init: int):
        super().__init__()
        self._request_kv_cache = request_kv_cache
        self._expected_len_at_init = int(expected_len_at_init)

    def init_postspec(self, *args, **kwargs):
        super().init_postspec()
        assert int(self._request_kv_cache.get_seq_length()) == int(self._expected_len_at_init)

    def postspec(self):
        self.postspec_calls += 1
        return False


def test_subspec_v2_fi_postverify_syncs_request_cache_after_prune_before_postspec():
    gen = SubSpecSDGeneratorBase.__new__(SubSpecSDGeneratorBase)
    gen.draft_params = SimpleNamespace(max_depth=4)
    gen.post_verify_count = 0
    backend = FlashInferV2Backend(gen)

    tree = _PruningTree(size_before=5, size_after=2)
    request_kv_cache = _DummyRequestCache(seq_len=0)
    position_offset = 10
    expected_len_after_prune = int(position_offset) + int(tree._size_after)
    gen.draft_model = _AssertingDraftModel(request_kv_cache, expected_len_after_prune)
    gen.draft_model._tree_ref = tree

    def _fi_draft_tree_decoding(*args, **kwargs):
        request_kv_cache = args[1]
        decoded_tokens = 4
        if kwargs.get("append_tokens", True):
            request_kv_cache.increment(decoded_tokens)
        return torch.zeros((1, decoded_tokens, 10)), decoded_tokens

    backend._draft_tree_decoding = _fi_draft_tree_decoding
    gen._verify = (
        lambda *args, **kwargs: (
            torch.tensor([[11, 22]], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
            (2, 1),
        )
    )

    out_tree, _ = backend._post_verify(
        tree=tree,
        root_ind=0,
        request_kv_cache=request_kv_cache,
        position_offset=position_offset,
        last_tree_depth=0,
        skip_nodes=1,
        logits_processor=None,
        device=torch.device("cpu"),
    )

    assert out_tree is tree
    assert int(request_kv_cache.get_seq_length()) == int(expected_len_after_prune)
    assert gen.draft_model.init_postspec_calls == 1
    assert gen.draft_model.postspec_calls == 1


def test_subspec_v2_fi_rewrite_batch_position_targets_trailing_window():
    gen = SubSpecSDGeneratorBase.__new__(SubSpecSDGeneratorBase)
    backend = FlashInferV2Backend(gen)
    request_kv_cache = _DummyRequestCache(seq_len=26, page_len=16)

    batch_position = backend._build_rewrite_batch_position(
        request_kv_cache,
        num_tokens=6,
        device=torch.device("cpu"),
    )

    assert batch_position.seq_indptr.tolist() == [0, 6]
    assert batch_position.batch_indices.tolist() == [0, 0, 0, 0, 0, 0]
    assert batch_position.positions.tolist() == [20, 21, 22, 23, 24, 25]
    assert batch_position.kv_page_indptr.tolist() == [0, 2]


def test_subspec_v2_fi_commit_seed_rebuilds_frontier_from_committed_boundary():
    gen = SubSpecSDGeneratorBase.__new__(SubSpecSDGeneratorBase)
    gen.draft_params = SimpleNamespace(max_depth=4, topk_len=6)
    gen.post_verify_count = 0
    backend = FlashInferV2Backend(gen)
    tree = _FixedTree(size=5)
    request_kv_cache = _DummyRequestCache(seq_len=11)
    gen.draft_model = _DummyDraftModel()
    gen.draft_model._tree_ref = tree

    out_tree = backend._commit_seed_postspec_before_post_verify(
        tree=tree,
        request_kv_cache=request_kv_cache,
        position_offset=10,
    )

    assert out_tree is tree
    assert gen.draft_model.init_postspec_calls == 1
    assert gen.draft_model.init_postspec_kwargs[0].get("rebuild_frontier") is True
    assert gen.draft_model.postspec_calls == 1
    assert gen.draft_model.update_calls == 1
    assert int(request_kv_cache.get_seq_length()) == 15


def test_subspec_v2_fi_postverify_debug_emits_rewrite_invariants():
    gen = SubSpecSDGeneratorBase.__new__(SubSpecSDGeneratorBase)
    gen.draft_params = SimpleNamespace(max_depth=4)
    gen.post_verify_count = 0
    gen.step_trace_enabled = True
    gen.step_trace_debug_verify = True
    backend = FlashInferV2Backend(gen)

    tree = _PruningTree(size_before=5, size_after=3)
    request_kv_cache = _DummyRequestCache(seq_len=13)
    position_offset = 10
    expected_len_after_prune = int(position_offset) + int(tree._size_after)
    gen.draft_model = _AssertingDraftModel(request_kv_cache, expected_len_after_prune)
    gen.draft_model._tree_ref = tree

    def _fi_draft_tree_decoding(*args, **kwargs):
        request_kv_cache = args[1]
        decoded_tokens = 4
        if kwargs.get("append_tokens", True):
            request_kv_cache.increment(decoded_tokens)
        return torch.zeros((1, decoded_tokens, 10)), decoded_tokens

    backend._draft_tree_decoding = _fi_draft_tree_decoding
    gen._verify = (
        lambda *args, **kwargs: (
            torch.tensor([[11, 22, 33]], dtype=torch.long),
            torch.tensor([0, 1, 2], dtype=torch.long),
            (3, 2),
        )
    )

    out_tree, _ = backend._post_verify(
        tree=tree,
        root_ind=0,
        request_kv_cache=request_kv_cache,
        position_offset=position_offset,
        last_tree_depth=0,
        skip_nodes=1,
        logits_processor=None,
        device=torch.device("cpu"),
    )

    assert out_tree is tree
    debug = getattr(gen, "_last_post_verify_debug", None)
    assert isinstance(debug, dict)
    assert debug["post_verify_rewrite_req_len_before"] == 13
    assert debug["post_verify_rewrite_req_len_after_sync"] == 15
    assert debug["post_verify_rewrite_window_start"] == 11
    assert debug["post_verify_rewrite_window_end"] == 14
    assert debug["post_verify_rewrite_req_len_after_decode"] == 15
    assert debug["post_verify_accept_len"] == 2
    assert debug["post_verify_kept_old_len"] == 3
    assert debug["post_verify_tree_token_hash"] == 8

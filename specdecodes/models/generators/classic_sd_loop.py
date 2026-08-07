"""Shared classic SD speculative-decoding loop, driven through a backend seam.

generators/classic_sd.py (SDPA) and generators/classic_sd_fi.py (FlashInfer) carried the
same-shaped classic `_generate`; only the KV-cache lifecycle, prefill, and attention
execution differed. `run_classic_generate` is that one loop, driven by a `ClassicBackend`
adapter chosen by the `backend:` config field.

Classic differs from the subspec v1 seam in two ways the hooks model: it keeps a **separate
draft KV cache** (FlashInfer maintains `request_kv_cache` for the target *and*
`draft_request_kv_cache` for the draft — reordered/pruned in lockstep), and `decode_headroom`
gates on the *draft* cache. `SdpaClassicBackend` delegates target-side ops to the shared
`ClassicSDGeneratorBase` helpers (`_tree_decoding`, `_speculate`); `FlashInferClassicBackend`
owns the FlashInfer tree-forward + draft speculate. SDPA's draft cache is the static/dynamic
`Cache` the draft model already holds, so its draft-side hooks are near-no-ops.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import nvtx


# --------------------------------------------------------------------------- #
# The classic backend seam.
# --------------------------------------------------------------------------- #
class ClassicBackend(ABC):
    """Attention/KV-cache backend that the shared classic SD loop drives."""

    @abstractmethod
    def begin(self, input_ids: torch.LongTensor, past_key_values: Any, draft_past_key_values: Any) -> torch.Tensor:
        """Set up backend state (+ draft cache) and run chunked prefill; return logits."""

    @abstractmethod
    def finalize(self, input_ids: torch.LongTensor) -> None:
        """End-of-generate hook. FlashInfer records reuse tokens; SDPA is a no-op."""

    def decode_headroom(self) -> int | None:
        """Appendable draft tokens before capacity, or ``None`` for an unbounded cache
        (SDPA). The loop stops a round early when this is ``<= 0``."""
        return None

    @abstractmethod
    def speculate(self, input_ids: torch.LongTensor) -> Any:
        """Run the draft model to propose a candidate tree."""

    def after_cap(
        self, tree_size_before: int, tree_size_after: int, input_ids: torch.LongTensor, decoded_tree_size: int
    ) -> None:
        """Reconcile the draft cache with the budget-capped tree (FlashInfer syncs
        request metadata; SDPA crops the dynamic draft cache).

        Runs *before* the ``decoded_tree_size <= 0`` break so FlashInfer always syncs
        (as the original FI loop did); ``decoded_tree_size`` lets SDPA reproduce its
        original crop-only-when-proceeding behavior."""
        return None

    @abstractmethod
    def current_kv_len(self) -> int:
        """Committed target KV length — the ``prev_kv_len`` source passed to ``commit``."""

    @abstractmethod
    def tree_forward(self, tree: Any, *, position_offset: int, decoded_tree_size: int, device: Any) -> Any:
        """Run the target model over the tree and return its outputs (``.logits`` used)."""

    @abstractmethod
    def commit(
        self,
        *,
        hidden_indices: torch.Tensor,
        prev_kv_len: int,
        decoded_tree_size: int,
        finished: bool,
        prune_tokens: int,
    ) -> None:
        """Keep the accepted prefix in the target (and draft) KV cache, prune on finish."""


# --------------------------------------------------------------------------- #
# The one shared classic loop. Drives any ClassicBackend; no backend branching.
# --------------------------------------------------------------------------- #
def run_classic_generate(
    gen,
    backend: "ClassicBackend",
    input_ids: torch.LongTensor,
    stopping_criteria,
    logits_processor,
    do_sample: bool,
    **model_kwargs,
):
    assert gen.target_model is not None, "target_model must be provided"
    assert gen.draft_model is not None, "draft_model must be provided"
    assert gen.tokenizer is not None, "tokenizer must be provided"

    input_ids = input_ids.clone()
    batch_size, _ = input_ids.shape
    assert batch_size == 1, "Only support batch_size=1 for now."

    if stopping_criteria.max_length is None and gen.cache_implementation == "static":
        raise ValueError(
            "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
        )
    if model_kwargs.get("past_key_values") is None:
        raise ValueError("past_key_values and draft_past_key_values should both be provided")
    past_key_values = model_kwargs["past_key_values"]
    draft_past_key_values = model_kwargs.get("draft_past_key_values")

    stream_callback = model_kwargs.get("stream_callback", None)

    with nvtx.annotate("prefill_chunked", color="orange"):
        next_token_logits = backend.begin(input_ids, past_key_values, draft_past_key_values)

    remaining = gen._remaining_token_budget(input_ids, stopping_criteria)
    if remaining is not None and int(remaining) <= 0:
        backend.finalize(input_ids)
        return input_ids

    with nvtx.annotate("sample"):
        sampled_tokens = gen._sample_token(next_token_logits, logits_processor, do_sample)

    with nvtx.annotate("state_update"):
        input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
        gen._maybe_stream(stream_callback, sampled_tokens)

    with nvtx.annotate("decode_loop"):
        finished = False
        while not finished:
            remaining = gen._remaining_token_budget(input_ids, stopping_criteria)
            if remaining is not None and int(remaining) <= 0:
                break
            headroom = backend.decode_headroom()
            if headroom is not None and int(headroom) <= 0:
                break

            with nvtx.annotate("speculate", color="cyan"):
                tree = backend.speculate(input_ids)
                tree_size_before_cap = int(tree.size())
                decoded_tree_size = gen._cap_tree_to_budget(
                    tree,
                    input_ids,
                    stopping_criteria,
                )
                tree_size_after_cap = int(tree.size())
                # Reconcile the draft cache before the break: the FI backend must roll
                # back its speculative increment even on a cap-to-zero terminal round
                # (matching the pre-seam loop); SDPA crops only when proceeding.
                backend.after_cap(
                    tree_size_before_cap,
                    tree_size_after_cap,
                    input_ids,
                    decoded_tree_size=int(decoded_tree_size),
                )
                if decoded_tree_size <= 0:
                    break

            with nvtx.annotate("target_decode", color="orange"):
                prev_kv_len = backend.current_kv_len()
                position_offset = int(input_ids.shape[1]) - 1
                outputs = backend.tree_forward(
                    tree,
                    position_offset=position_offset,
                    decoded_tree_size=int(decoded_tree_size),
                    device=input_ids.device,
                )
                next_token_logits = outputs.logits if outputs is not None else None
                del outputs

            with nvtx.annotate("verify"):
                sampled_tokens, hidden_indices, _ = gen._verify(
                    tree,
                    0,
                    next_token_logits,
                    logits_processor,
                    do_sample,
                )
                sampled_tokens = sampled_tokens.to(input_ids.device)
                del next_token_logits

            with nvtx.annotate("state_update"):
                input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)

            with nvtx.annotate("stop_check"):
                finished, input_ids, kept, prune_tokens = gen._apply_tokenwise_stopping_criteria(
                    input_ids=input_ids,
                    sampled_tokens=sampled_tokens,
                    stopping_criteria=stopping_criteria,
                )
            if kept.numel() > 0:
                gen._maybe_stream(stream_callback, kept)

            with nvtx.annotate("kv_reorder"):
                backend.commit(
                    hidden_indices=hidden_indices,
                    prev_kv_len=int(prev_kv_len),
                    decoded_tree_size=int(decoded_tree_size),
                    finished=bool(finished),
                    prune_tokens=int(prune_tokens),
                )

    backend.finalize(input_ids)
    return input_ids


# --------------------------------------------------------------------------- #
# SDPA / static-cache adapter. Target ops delegate to the shared base helpers.
# --------------------------------------------------------------------------- #
class SdpaClassicBackend(ClassicBackend):
    def __init__(self, gen):
        self.gen = gen
        self.past_key_values = None
        self.draft_past_key_values = None

    def begin(self, input_ids, past_key_values, draft_past_key_values):
        g = self.gen
        self.past_key_values = past_key_values
        self.draft_past_key_values = draft_past_key_values
        max_cache_len = getattr(past_key_values.cache, "max_cache_len", None)
        if draft_past_key_values is not None:
            g.draft_model.set_past_key_values(draft_past_key_values)
        g._init_tree_mask(g.draft_params.max_verify_tokens, max_cache_len, device=input_ids.device)
        outputs = g._chunked_prefill_forward(
            input_ids,
            past_key_values,
            prefill_chunk_size=g.prefill_chunk_size,
            use_position_ids=True,
        )
        return outputs.logits

    def finalize(self, input_ids):
        return None

    def speculate(self, input_ids):
        contiguous_input_ids = input_ids.clone(memory_format=torch.contiguous_format)
        return self.gen._speculate(contiguous_input_ids)

    def after_cap(self, tree_size_before, tree_size_after, input_ids, decoded_tree_size):
        # Original SDPA loop cropped the dynamic draft cache only *after* passing the
        # `decoded_tree_size <= 0` break, i.e. only when the round proceeds.
        if int(decoded_tree_size) > 0 and self.gen.cache_implementation == "dynamic":
            self.draft_past_key_values.crop(int(input_ids.shape[1]))

    def current_kv_len(self):
        return int(self.past_key_values.get_seq_length())

    def tree_forward(self, tree, *, position_offset, decoded_tree_size, device):
        cache_position = torch.arange(
            int(position_offset),
            int(position_offset) + int(decoded_tree_size),
            dtype=torch.long,
            device=device,
        )
        return self.gen._tree_decoding(
            tree,
            self.past_key_values,
            position_offset=position_offset,
            cache_position=cache_position,
            device=device,
        )

    def commit(self, *, hidden_indices, prev_kv_len, decoded_tree_size, finished, prune_tokens):
        pkv = self.past_key_values
        pkv.reorder_cache_with_offset(
            hidden_indices,
            offset=int(prev_kv_len),
            new_chunk_len=int(decoded_tree_size),
            dim=2,
        )
        pkv.seq_len += int(hidden_indices.shape[0])
        if finished:
            pkv.seq_len -= int(prune_tokens)


# --------------------------------------------------------------------------- #
# FlashInfer / paged adapter. Separate target + draft RequestKvCache; owns the
# FlashInfer tree forward + draft speculate.
# --------------------------------------------------------------------------- #
class FlashInferClassicBackend(ClassicBackend):
    def __init__(self, gen):
        self.gen = gen
        self.kv_cache_pool = None
        self.request_kv_cache = None
        self.draft_request_kv_cache = None

    # ----- backend-specific method (relocated from the FI classic generator) ----- #
    def _tree_decoding(self, tree, request_kv_cache, position_offset, cache_position, device):
        from ..utils.flashinfer.cache_manager import getKvCacheBatchPosition

        g = self.gen
        kv_cache_pool = request_kv_cache.kvCachePool
        tree_input_ids, tree_position_ids, tree_mask = g._prepare_tree_inputs_and_mask(
            tree,
            position_offset=position_offset,
            device=device,
            model_dtype=kv_cache_pool.cache_data[0].dtype,
            non_blocking=True,
            invert=False,
        )

        # Target model forward
        with nvtx.annotate("target_forward", color="red"):
            num_tokens = int(tree_input_ids.shape[0])
            if num_tokens == 0:
                return None
            kvCachePool = kv_cache_pool

            request_kv_cache.increment(num_tokens)

            batch_position = getKvCacheBatchPosition(
                request_kv_caches=[request_kv_cache],
                mode="tree",  # Set to False if you're doing incremental decoding
                device=device,
                treeTokens=num_tokens,
            )
            g.flashinferWrapper.prepareAttention(
                "tree",
                batch_position,
                kvCachePool.page_len,
                "NONE",  # POS_ENCODING_MODE.NONE,
                kvCachePool.cache_data[0].dtype,
                attention_mask=tree_mask,
            )
            # Check if the current instance has the attribute 'graph'
            if hasattr(g, "graph"):
                outputs = g.tree_decoding_step(
                    input_ids=tree_input_ids.unsqueeze(0),
                    position_ids=tree_position_ids.unsqueeze(0),
                    batch_position=batch_position,
                )
            else:
                outputs = g.target_model(
                    input_ids=tree_input_ids.unsqueeze(0),
                    past_key_values=None,
                    position_ids=tree_position_ids.unsqueeze(0),
                    output_hidden_states=True,
                    use_cache=False,
                    kvCachePool=kvCachePool,
                    batch_position=batch_position,
                    mode="tree",
                    flashinferWrapper=g.flashinferWrapper,
                )
        return outputs

    # ----- ClassicBackend hooks ----- #
    def begin(self, input_ids, past_key_values, draft_past_key_values):
        from ..utils.flashinfer.cache_manager import RequestKvCache
        from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper
        from ..utils.flashinfer.prefill import flashinfer_chunked_prefill

        g = self.gen
        max_cache_len = getattr(past_key_values, "max_cache_len", None)
        g._init_tree_mask(g.draft_params.max_verify_tokens, max_cache_len, device=input_ids.device)

        if not hasattr(g, "flashinferWrapper"):
            g.flashinferWrapper = FlashinferAttentionWrapper(
                g.target_model.config.num_attention_heads,
                g.target_model.config.num_key_value_heads,
                g.target_model.config.hidden_size,
                past_key_values.page_len,
                # Tree row count can vary across decode rounds; keep planning dynamic.
                tree_use_cuda_graph=False,
            )

        self.kv_cache_pool = past_key_values
        g.kvCachePool = past_key_values
        self.request_kv_cache = g._ensure_request_kv_cache(
            attr_name="_fi_request_kv_cache",
            request_cls=RequestKvCache,
            kv_cache_pool=self.kv_cache_pool,
            input_ids_len=int(input_ids.shape[1]),
            input_ids=input_ids,
            tokens_attr_name="_fi_request_tokens",
        )
        self.draft_request_kv_cache = g._ensure_request_kv_cache(
            attr_name="_fi_draft_request_kv_cache",
            request_cls=RequestKvCache,
            kv_cache_pool=draft_past_key_values,
            input_ids_len=int(input_ids.shape[1]),
            input_ids=input_ids,
            tokens_attr_name="_fi_draft_request_tokens",
        )
        outputs = flashinfer_chunked_prefill(
            target_model=g.target_model,
            flashinfer_wrapper=g.flashinferWrapper,
            input_ids=input_ids,
            kv_cache_pool=self.kv_cache_pool,
            request_kv_cache=self.request_kv_cache,
            prefill_chunk_size=g.prefill_chunk_size,
        )
        return outputs.logits

    def finalize(self, input_ids):
        g = self.gen
        g._remember_request_cache_tokens(
            tokens_attr_name="_fi_request_tokens",
            input_ids=input_ids,
        )
        g._remember_request_cache_tokens(
            tokens_attr_name="_fi_draft_request_tokens",
            input_ids=input_ids,
        )

    def decode_headroom(self):
        return self.gen._request_cache_headroom(self.draft_request_kv_cache)

    def speculate(self, input_ids):
        g = self.gen
        last_token_ids = input_ids[
            :, self.draft_request_kv_cache.get_seq_length() :
        ].clone(memory_format=torch.contiguous_format)
        return g.draft_model.speculate(
            last_token_ids,
            request_kv_cache=self.draft_request_kv_cache,
            flashinferWrapper=g.flashinferWrapper,
        )

    def after_cap(self, tree_size_before, tree_size_after, input_ids, decoded_tree_size):
        # Original FI loop synced (rolling back the speculative draft increment) before
        # the `decoded_tree_size <= 0` break, so this runs unconditionally.
        self.gen._sync_request_cache_after_tree_truncation(
            self.draft_request_kv_cache,
            tree_size_before=int(tree_size_before),
            tree_size_after=int(tree_size_after),
        )

    def current_kv_len(self):
        return int(self.request_kv_cache.get_seq_length()) + 1

    def tree_forward(self, tree, *, position_offset, decoded_tree_size, device):
        return self._tree_decoding(
            tree,
            self.request_kv_cache,
            position_offset=position_offset,
            cache_position=None,
            device=device,
        )

    def commit(self, *, hidden_indices, prev_kv_len, decoded_tree_size, finished, prune_tokens):
        req = self.request_kv_cache
        draft_req = self.draft_request_kv_cache
        num_new_tokens = int(decoded_tree_size)
        req.reorder_cache_with_offset(
            hidden_indices,
            offset=int(prev_kv_len),
            num_new_tokens=num_new_tokens,
        )
        draft_req.reorder_cache_with_offset(
            hidden_indices,
            offset=draft_req.get_seq_length(),
            num_new_tokens=num_new_tokens,
        )
        if finished and int(prune_tokens) > 0:
            req.decrement(int(prune_tokens))
            draft_req.decrement(int(prune_tokens))

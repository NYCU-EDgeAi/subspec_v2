"""Shared v2 SubSpec (post-verify) speculative-decoding loop, behind a backend seam.

generators/subspec_sd_v2.py (SDPA) and generators/subspec_sd_v2_fi.py (FlashInfer)
carried the same-shaped post-verify `_generate`; only KV-cache lifecycle, prefill, and
attention execution differed. `run_subspec_v2_generate` is that one loop; the two
`SubSpecV2Backend` adapters below wrap each generator's existing methods, so this is a
dedup, not a rewrite of the numerics.

The v2 loop is *not* the clean parallel the v1 pair was: v2_fi has control-flow-mutating
divergences the shared loop models through backend hooks — FI-only request-cache syncs at
several points (`sync_to_tree`), an FI-only commit-seed postspec step before post-verify
(`commit_seed`), a `_post_verify` that threads `cache_position` for SDPA but not FI
(`post_verify`), a budget clamp that mutates the tree for SDPA vs a truncation-sync for FI
(`after_cap`), and a `kv_reorder` tail that on the FI side reassigns loop state
(`reorder_tail`, which RETURNS the updated `(root_ind, is_prev_accepted,
hidden_indices_cache)`). SDPA's implementations of those hooks are the identity/no-op.

This mirrors the v1 seam (generators/subspec_sd_v1_loop.py). The adapters intentionally
hold a reference to their generator (`gen`) and delegate to its existing helpers; they are
internal collaborators, not standalone units. Its own ABC (not the v1
`SpecDecodeBackend`): the v2 hook set is wider and differently typed, so co-locating the
contract with its only consumer avoids stub methods on the v1 anchor.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import nvtx


# --------------------------------------------------------------------------- #
# The v2 backend seam. Wider than the v1 `SpecDecodeBackend` (post-verify has
# more backend-divergent points); every SDPA implementation of an FI-only hook
# is the identity/no-op.
# --------------------------------------------------------------------------- #
class SubSpecV2Backend(ABC):
    """Attention/KV-cache backend that the shared v2 post-verify loop drives."""

    #: Passed to `_remap_hidden_indices_after_tree_prune` for error messages.
    method_name: str = "subspec_sd_v2"

    @abstractmethod
    def begin(self, input_ids: torch.LongTensor, past_key_values: Any) -> torch.Tensor:
        """Set up backend state and run chunked prefill; return next-token logits."""

    @abstractmethod
    def finalize(self, input_ids: torch.LongTensor) -> None:
        """End-of-generate hook. FlashInfer records reuse tokens; SDPA is a no-op."""

    @abstractmethod
    def speculate(self, last_token_id: torch.LongTensor) -> Any:
        """Run the draft model to propose a fresh candidate tree."""

    def sync_to_tree(self, *, position_offset: int, tree_size: int) -> None:
        """Clamp request-cache metadata to the `[prefix + tree]` footprint.
        FlashInfer-only; SDPA is a no-op (the default)."""
        return None

    @abstractmethod
    def flush_headroom_cap(self) -> int | None:
        """Cache capacity used to decide whether a carry-over post-verify still fits,
        or ``None`` for an unbounded cache. SDPA: ``max_cache_len``; FI: request-cache
        capacity."""

    @abstractmethod
    def flush_deferred(
        self, hidden_indices_cache: Any, tree_size: int, *, input_len: int
    ) -> tuple[int, bool, Any]:
        """Flush the deferred tree cache and return the reset
        ``(root_ind, is_prev_accepted, hidden_indices_cache)`` loop state."""

    def commit_seed(self, tree: Any, *, position_offset: int) -> Any:
        """One deterministic postspec commit-seed step before carry-over post-verify.
        FlashInfer-only; SDPA returns ``tree`` unchanged (the default)."""
        return tree

    @abstractmethod
    def post_verify(
        self,
        tree: Any,
        root_ind: int,
        *,
        position_offset: int,
        skip_nodes: int,
        last_tree_depth: int,
        logits_processor: Any,
        device: Any,
    ) -> tuple[Any, Any]:
        """Run post-verify over the carried tree; return ``(tree, kept_old_indices)``.
        SDPA threads a computed ``cache_position``; FI does not."""

    @abstractmethod
    def after_cap(
        self,
        tree: Any,
        *,
        tree_size_before: int,
        tree_size_after: int,
        position_offset: int,
        skip_nodes: int,
        decoded_tree_size: int,
    ) -> int:
        """React to the budget cap and return the (possibly reduced) decoded tree size.
        SDPA clamps to ``max_cache_len`` (may truncate the tree and shrink the size);
        FI syncs request-cache metadata after the truncation and returns it unchanged."""

    @abstractmethod
    def tree_forward(
        self, tree: Any, *, position_offset: int, skip_nodes: int, decoded_tree_size: int, device: Any
    ) -> Any:
        """Run the target model over the capped tree and return its outputs."""

    @abstractmethod
    def reorder_tail(
        self,
        *,
        hidden_indices_cache: Any,
        tree_size: int,
        root_ind: int,
        is_prev_accepted: bool,
        finished: bool,
        prune_tokens: int,
        disable_post_verify: bool,
        input_len: int,
    ) -> tuple[int, bool, Any]:
        """End-of-round KV reorder. Returns the (possibly reassigned)
        ``(root_ind, is_prev_accepted, hidden_indices_cache)``. On FI a
        ``disable_post_verify`` accepted round flushes deferred cache and resets state;
        SDPA returns the triple unchanged."""


# --------------------------------------------------------------------------- #
# The one shared v2 loop. Drives any SubSpecV2Backend; no backend branching.
# --------------------------------------------------------------------------- #
def run_subspec_v2_generate(
    gen,
    backend: "SubSpecV2Backend",
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
        raise ValueError("past_key_values is not provided")
    past_key_values = model_kwargs["past_key_values"]

    stream_callback = model_kwargs.get("stream_callback", None)
    gen._init_step_trace()

    with nvtx.annotate("prefill_chunked", color="orange"):
        next_token_logits = backend.begin(input_ids, past_key_values)

    remaining = gen._remaining_token_budget(input_ids, stopping_criteria)
    if remaining is not None and int(remaining) <= 0:
        backend.finalize(input_ids)
        return input_ids

    with nvtx.annotate("sample"):
        sampled_tokens = gen._sample_token(next_token_logits, logits_processor, do_sample)

    with nvtx.annotate("state_update"):
        input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
        position_offset = int(input_ids.shape[1]) - 1
        gen._maybe_stream(stream_callback, sampled_tokens)

    with nvtx.annotate("decode_loop"):
        # `post_verify_count`: previous tree was fully accepted, so run post-verify
        # instead of re-speculating. `speculate_count`: ran a fresh speculation step.
        gen.post_verify_count = 0
        gen.speculate_count = 0
        disable_post_verify = bool(gen.generator_kwargs.get("disable_post_verify", False))

        finished = False
        is_prev_accepted = False
        hidden_indices_cache = None
        last_tree_size = 0
        last_tree_depth = 0
        root_ind = 0

        while not finished:
            remaining = gen._remaining_token_budget(input_ids, stopping_criteria)
            if remaining is not None and int(remaining) <= 0:
                break

            post_verify_used = False
            if is_prev_accepted:
                skip_nodes = int(last_tree_size)
                pending_post_tokens = int(tree.size()) - int(skip_nodes)
                cache_cap = backend.flush_headroom_cap()
                should_flush_deferred = bool(disable_post_verify) or int(pending_post_tokens) <= 0
                if (not should_flush_deferred) and (cache_cap is not None):
                    cache_headroom = max(
                        0,
                        int(cache_cap) - (int(position_offset) + int(skip_nodes)),
                    )
                    should_flush_deferred = int(cache_headroom) < int(pending_post_tokens)
                if should_flush_deferred:
                    root_ind, is_prev_accepted, hidden_indices_cache = backend.flush_deferred(
                        hidden_indices_cache, int(tree.size()), input_len=int(input_ids.shape[1]),
                    )
                    continue

                tree = backend.commit_seed(tree, position_offset=int(position_offset))
                post_verify_used = True
                with nvtx.annotate("post_verify", color="cyan"):
                    tree, kept_old_indices = backend.post_verify(
                        tree,
                        root_ind,
                        position_offset=int(position_offset),
                        skip_nodes=int(skip_nodes),
                        last_tree_depth=int(last_tree_depth),
                        logits_processor=logits_processor,
                        device=input_ids.device,
                    )
                hidden_indices_cache = gen._remap_hidden_indices_after_tree_prune(
                    hidden_indices_cache,
                    kept_old_indices,
                    method_name=backend.method_name,
                )
                if int(tree.size()) <= int(skip_nodes):
                    root_ind, is_prev_accepted, hidden_indices_cache = backend.flush_deferred(
                        hidden_indices_cache, int(tree.size()), input_len=int(input_ids.shape[1]),
                    )
                    continue

                last_tree_size = int(tree.size())
                last_tree_depth = int(tree.get_depth())

            else:
                gen.speculate_count += 1
                with nvtx.annotate("speculate", color="cyan"):
                    last_token_id = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
                    tree = backend.speculate(last_token_id)
                position_offset = int(input_ids.shape[1]) - 1
                backend.sync_to_tree(position_offset=int(position_offset), tree_size=int(tree.size()))
                last_tree_size = int(tree.size())
                last_tree_depth = int(tree.get_depth())
                skip_nodes = 0

            tree_size_before_cap = int(tree.size())
            decoded_tree_size = gen._cap_tree_to_budget(
                tree,
                input_ids,
                stopping_criteria,
                skip_nodes=int(skip_nodes),
            )
            tree_size_after_cap = int(tree.size())
            decoded_tree_size = backend.after_cap(
                tree,
                tree_size_before=int(tree_size_before_cap),
                tree_size_after=int(tree_size_after_cap),
                position_offset=int(position_offset),
                skip_nodes=int(skip_nodes),
                decoded_tree_size=int(decoded_tree_size),
            )
            if int(decoded_tree_size) <= 0:
                break
            last_tree_size = int(tree.size())

            with nvtx.annotate("target_decode", color="orange"):
                gen.draft_model.init_postspec()
                outputs = backend.tree_forward(
                    tree,
                    position_offset=int(position_offset),
                    skip_nodes=int(skip_nodes),
                    decoded_tree_size=int(decoded_tree_size),
                    device=input_ids.device,
                )
                next_token_logits = outputs.logits if outputs is not None else None

            with nvtx.annotate("postspec_update", color="cyan"):
                tree = gen.draft_model.update_tree_after_post()
                backend.sync_to_tree(position_offset=int(position_offset), tree_size=int(tree.size()))

            with nvtx.annotate("verify"):
                root_ind_in = int(root_ind) if is_prev_accepted else 0
                step_trace_extra = gen._build_verify_debug_trace(
                    tree=tree,
                    next_token_logits=next_token_logits,
                    skip_nodes=int(skip_nodes),
                )
                sampled_tokens, hidden_indices, (_, accept_len) = gen._verify(
                    tree,
                    root_ind_in,
                    next_token_logits,
                    logits_processor,
                    do_sample,
                    skip_nodes=int(skip_nodes),
                )
                sampled_tokens = sampled_tokens.to(input_ids.device)
                hidden_indices = hidden_indices.to(input_ids.device)

                last_accepted_ind = int(hidden_indices[-1].item())
                bonus_token = int(sampled_tokens[:, -1].item())

                if is_prev_accepted:
                    hidden_indices_cache = torch.cat([hidden_indices_cache, hidden_indices], dim=-1)
                else:
                    hidden_indices_cache = hidden_indices

            root_ind = int(tree.find_child_index(last_accepted_ind, bonus_token))
            root_ind_out = int(root_ind)
            is_prev_accepted = int(root_ind) >= 0
            gen._append_step_trace(
                is_prev_accepted=bool(is_prev_accepted),
                skip_nodes=int(skip_nodes),
                tree_size_before_cap=int(tree_size_before_cap),
                tree_size_after_cap=int(tree_size_after_cap),
                decoded_tree_size=int(decoded_tree_size),
                root_ind_in=int(root_ind_in),
                root_ind_out=int(root_ind_out),
                accept_len=int(accept_len),
                hidden_indices_len=int(hidden_indices.numel()),
                post_verify_used=bool(post_verify_used),
                extra_fields=step_trace_extra,
            )

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
                root_ind, is_prev_accepted, hidden_indices_cache = backend.reorder_tail(
                    hidden_indices_cache=hidden_indices_cache,
                    tree_size=int(tree.size()),
                    root_ind=root_ind,
                    is_prev_accepted=is_prev_accepted,
                    finished=bool(finished),
                    prune_tokens=int(prune_tokens),
                    disable_post_verify=bool(disable_post_verify),
                    input_len=int(input_ids.shape[1]),
                )

        # Normalize to plain ints for logging/consumers.
        gen.post_verify_count = int(gen.post_verify_count)
        gen.speculate_count = int(gen.speculate_count)

    backend.finalize(input_ids)
    return input_ids


# --------------------------------------------------------------------------- #
# SDPA / static-cache adapter. Unbounded-by-default; threads `cache_position`.
# --------------------------------------------------------------------------- #
class SdpaV2Backend(SubSpecV2Backend):
    method_name = "subspec_sd_v2"

    def __init__(self, gen):
        self.gen = gen
        self.past_key_values = None
        self.max_cache_len = None

    def begin(self, input_ids, past_key_values):
        g = self.gen
        self.past_key_values = past_key_values
        self.max_cache_len = getattr(past_key_values.cache, "max_cache_len", None)
        g.draft_model.set_past_key_values(past_key_values)
        g._init_tree_mask(
            g.draft_params.max_verify_tokens * 2,
            self.max_cache_len,
            device=input_ids.device,
        )
        outputs = g._chunked_prefill_forward(
            input_ids,
            past_key_values,
            prefill_chunk_size=g.prefill_chunk_size,
            use_position_ids=True,
        )
        return outputs.logits

    def finalize(self, input_ids):
        return None

    def speculate(self, last_token_id):
        return self.gen._speculate(last_token_id)

    def flush_headroom_cap(self):
        return self.max_cache_len

    def flush_deferred(self, hidden_indices_cache, tree_size, *, input_len):
        return self.gen._flush_deferred_tree_cache(
            self.past_key_values,
            hidden_indices_cache,
            int(tree_size),
            input_len=int(input_len),
        )

    def post_verify(self, tree, root_ind, *, position_offset, skip_nodes, last_tree_depth, logits_processor, device):
        cache_position = torch.arange(
            int(position_offset) + int(skip_nodes),
            int(position_offset) + int(tree.size()),
            dtype=torch.long,
            device=device,
        )
        return self.gen._post_verify(
            tree,
            root_ind,
            self.past_key_values,
            position_offset,
            cache_position,
            last_tree_depth,
            skip_nodes,
            logits_processor,
            device,
        )

    def after_cap(self, tree, *, tree_size_before, tree_size_after, position_offset, skip_nodes, decoded_tree_size):
        if self.max_cache_len is not None:
            cache_decode_budget = max(
                0,
                int(self.max_cache_len) - (int(position_offset) + int(skip_nodes)),
            )
            if int(decoded_tree_size) > int(cache_decode_budget):
                max_tree_nodes = int(skip_nodes) + int(cache_decode_budget)
                if hasattr(tree, "truncate_prefix"):
                    tree.truncate_prefix(max_tree_nodes)
                else:
                    tree.prune_to_top_n(max_tree_nodes)
                decoded_tree_size = int(cache_decode_budget)
        return int(decoded_tree_size)

    def tree_forward(self, tree, *, position_offset, skip_nodes, decoded_tree_size, device):
        cache_position = torch.arange(
            int(position_offset) + int(skip_nodes),
            int(position_offset) + int(skip_nodes) + int(decoded_tree_size),
            dtype=torch.long,
            device=device,
        )
        return self.gen._tree_decoding(
            tree,
            self.past_key_values,
            position_offset=position_offset,
            cache_position=cache_position,
            skip_nodes=skip_nodes,
            device=device,
        )

    def reorder_tail(self, *, hidden_indices_cache, tree_size, root_ind, is_prev_accepted, finished, prune_tokens, disable_post_verify, input_len):
        if (not is_prev_accepted) or finished:
            self.gen._reorder_pending_tree_cache(
                self.past_key_values,
                hidden_indices_cache,
                int(tree_size),
                input_len=int(input_len),
            )
            if finished:
                self.past_key_values.seq_len -= prune_tokens
        return root_ind, is_prev_accepted, hidden_indices_cache


# --------------------------------------------------------------------------- #
# FlashInfer / paged adapter. Request-cache syncs + commit-seed; capacity-bounded.
# --------------------------------------------------------------------------- #
class FlashInferV2Backend(SubSpecV2Backend):
    method_name = "subspec_sd_v2_fi"

    def __init__(self, gen):
        self.gen = gen
        self.kv_cache_pool = None
        self.request_kv_cache = None
        self.max_cache_len = None
        self._prompt_len = 0

    def begin(self, input_ids, past_key_values):
        from ..utils.flashinfer.cache_manager import RequestKvCache
        from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper
        from ..utils.flashinfer.prefill import flashinfer_chunked_prefill

        g = self.gen
        self._prompt_len = int(input_ids.shape[1])
        self.kv_cache_pool = past_key_values
        self.max_cache_len = getattr(past_key_values, "max_cache_len", None)

        if not hasattr(g, "flashinferWrapper"):
            g.flashinferWrapper = FlashinferAttentionWrapper(
                g.target_model.config.num_attention_heads,
                g.target_model.config.num_key_value_heads,
                g.target_model.config.hidden_size,
                past_key_values.page_len,
                # SubSpec v2 overlap has variable tree row counts; keep FlashInfer
                # tree planning dynamic instead of pinning to the first row count.
                tree_use_cuda_graph=False,
            )
        g._init_tree_mask(
            int(g.draft_params.max_verify_tokens) * 2,
            self.max_cache_len,
            device=input_ids.device,
        )
        self.request_kv_cache = g._ensure_request_kv_cache(
            attr_name="_fi_v2_request_kv_cache",
            request_cls=RequestKvCache,
            kv_cache_pool=self.kv_cache_pool,
            input_ids_len=int(input_ids.shape[1]),
            input_ids=input_ids,
            tokens_attr_name="_fi_v2_request_tokens",
            reuse_len_attr_name="_fi_v2_request_reuse_len",
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
            tokens_attr_name="_fi_v2_request_tokens",
            input_ids=input_ids,
        )
        setattr(g, "_fi_v2_request_reuse_len", int(self._prompt_len))

    def speculate(self, last_token_id):
        return self.gen._speculate(last_token_id, self.request_kv_cache)

    def sync_to_tree(self, *, position_offset, tree_size):
        self.gen._sync_request_cache_to_tree(
            self.request_kv_cache,
            position_offset=int(position_offset),
            tree_size=int(tree_size),
        )

    def flush_headroom_cap(self):
        return self.gen._request_cache_capacity(self.request_kv_cache)

    def flush_deferred(self, hidden_indices_cache, tree_size, *, input_len):
        return self.gen._flush_deferred_tree_cache(
            self.request_kv_cache,
            hidden_indices_cache,
            int(tree_size),
        )

    def commit_seed(self, tree, *, position_offset):
        return self.gen._commit_seed_postspec_before_post_verify(
            tree=tree,
            request_kv_cache=self.request_kv_cache,
            position_offset=int(position_offset),
        )

    def post_verify(self, tree, root_ind, *, position_offset, skip_nodes, last_tree_depth, logits_processor, device):
        return self.gen._post_verify(
            tree,
            int(root_ind),
            self.request_kv_cache,
            position_offset,
            int(last_tree_depth),
            int(skip_nodes),
            logits_processor,
            device,
        )

    def after_cap(self, tree, *, tree_size_before, tree_size_after, position_offset, skip_nodes, decoded_tree_size):
        self.gen._sync_request_cache_after_tree_truncation(
            self.request_kv_cache,
            tree_size_before=int(tree_size_before),
            tree_size_after=int(tree_size_after),
        )
        return int(decoded_tree_size)

    def tree_forward(self, tree, *, position_offset, skip_nodes, decoded_tree_size, device):
        g = self.gen
        g._sync_request_cache_to_tree(
            self.request_kv_cache,
            position_offset=int(position_offset),
            tree_size=int(tree.size()),
        )
        outputs = g._tree_decoding(
            tree,
            self.request_kv_cache,
            position_offset=int(position_offset),
            skip_nodes=int(skip_nodes),
            device=device,
        )
        next_token_logits = outputs.logits if outputs is not None else None
        if next_token_logits is not None and int(next_token_logits.shape[1]) != int(decoded_tree_size):
            raise RuntimeError(
                "FI target tree logits length mismatch: "
                f"logits_len={int(next_token_logits.shape[1])}, "
                f"decoded_tree_size={int(decoded_tree_size)}, "
                f"skip_nodes={int(skip_nodes)}, tree_size={int(tree.size())}"
            )
        return outputs

    def reorder_tail(self, *, hidden_indices_cache, tree_size, root_ind, is_prev_accepted, finished, prune_tokens, disable_post_verify, input_len):
        if disable_post_verify and bool(is_prev_accepted) and (not bool(finished)):
            return self.gen._flush_deferred_tree_cache(
                self.request_kv_cache,
                hidden_indices_cache,
                int(tree_size),
            )
        elif (not is_prev_accepted) or finished:
            self.gen._reorder_pending_tree_cache(
                self.request_kv_cache,
                hidden_indices_cache,
                int(tree_size),
            )
            if finished and int(prune_tokens) > 0:
                self.request_kv_cache.decrement(int(prune_tokens))
        return root_ind, is_prev_accepted, hidden_indices_cache

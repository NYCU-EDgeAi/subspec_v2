"""Shared v2 SubSpec (post-verify) speculative-decoding loop, behind a backend seam.

One generator class (`generators/subspec_sd_v2.py::SubSpecSDGenerator`) drives this loop
for both attention backends; the backend is chosen by the `backend:` config field. Each
`SubSpecV2Backend` adapter below **owns** the backend-specific KV-cache lifecycle, prefill,
and attention execution (SDPA static/dynamic `Cache` vs FlashInfer paged `RequestKvCache`).
The generator holds only the shared algorithm helpers (`_verify`,
`_prepare_tree_inputs_and_mask`, `_cap_tree_to_budget`, step-trace, ...); the adapters call
back into it via `self.gen` for those.

The v2 loop is *not* the clean parallel the v1 pair was: v2_fi has control-flow-mutating
divergences the shared loop models through backend hooks — FI-only request-cache syncs at
several points (`sync_to_tree`), an FI-only commit-seed postspec step before post-verify
(`commit_seed`), a `_post_verify` that threads `cache_position` for SDPA but not FI
(`post_verify`), a budget clamp that mutates the tree for SDPA vs a truncation-sync for FI
(`after_cap`), and a `kv_reorder` tail that on the FI side reassigns loop state
(`reorder_tail`, which RETURNS the updated `(root_ind, is_prev_accepted,
hidden_indices_cache)`). SDPA's implementations of those hooks are the identity/no-op.

The adapters' backend methods (`_tree_decoding`, `_post_verify`, `_draft_tree_decoding`, ...)
keep explicit cache arguments so they stay unit-testable in isolation. Its own ABC (not the
v1 `SpecDecodeBackend`): the v2 hook set is wider and differently typed, so co-locating the
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
# Owns the SDPA KV-cache + attention methods (were on the SDPA generator).
# --------------------------------------------------------------------------- #
class SdpaV2Backend(SubSpecV2Backend):
    method_name = "subspec_sd_v2"

    def __init__(self, gen):
        self.gen = gen
        self.past_key_values = None
        self.max_cache_len = None

    # ----- backend-specific methods (relocated from the SDPA generator) ----- #
    def _draft_tree_decoding(self, tree, past_key_values, position_offset, cache_position, skip_nodes, device):
        g = self.gen
        tree_input_ids, tree_position_ids, tree_mask = g._prepare_tree_inputs_and_mask(
            tree,
            position_offset=position_offset,
            device=device,
            model_dtype=g.draft_model.model.dtype,
            skip_nodes=skip_nodes,
            invert=True,
        )
        with nvtx.annotate("draft_forward", color="red"):
            next_token_logits = g.draft_model(
                tree_input_ids.unsqueeze(0),
                past_key_values=past_key_values.cache,
                attention_mask=tree_mask,
                position_ids=tree_position_ids.unsqueeze(0),
                cache_position=cache_position,
            )
        return next_token_logits

    def _post_verify(self, tree, root_ind, past_key_values, position_offset, cache_position, last_tree_depth, skip_nodes, logits_processor, device):
        g = self.gen
        verify_tokens_expected = max(0, int(tree.size()) - int(skip_nodes))
        if int(verify_tokens_expected) <= 0:
            return tree, None

        next_token_logits = self._draft_tree_decoding(
            tree,
            past_key_values,
            position_offset=position_offset,
            cache_position=cache_position,
            skip_nodes=skip_nodes,
            device=device,
        )
        _, _, (_, accept_len) = g._verify(
            tree,
            root_ind,
            next_token_logits,
            logits_processor,
            False,
            skip_nodes=skip_nodes,
        )

        accept_len = int(accept_len)
        kept_old_indices = tree.prune_to_depth(int(last_tree_depth) + accept_len)

        # Speculate to refill the tree.
        refill_steps = int(g.draft_params.max_depth) - accept_len
        if refill_steps > 0:
            with nvtx.annotate("postspec_refill", color="cyan"):
                g.draft_model.init_postspec()
                for _ in range(refill_steps):
                    if not g.draft_model.postspec():
                        break
            tree = g.draft_model.update_tree_after_post()

        g.post_verify_count += 1
        return tree, kept_old_indices

    def _tree_decoding(self, tree, past_key_values, position_offset, cache_position, skip_nodes, device):
        g = self.gen
        # Disable draft profiling during target forward
        if g.profiling:
            g.profile_draft_time = False

        tree_input_ids, tree_position_ids, tree_mask = g._prepare_tree_inputs_and_mask(
            tree,
            position_offset=position_offset,
            device=device,
            model_dtype=g.target_model.model.dtype,
            skip_nodes=skip_nodes,
            invert=True,
        )

        with nvtx.annotate("target_forward", color="red"):
            outputs = g.target_model(
                tree_input_ids.unsqueeze(0),
                past_key_values=past_key_values.cache,
                attention_mask=tree_mask,
                position_ids=tree_position_ids.unsqueeze(0),
                cache_position=cache_position,
            )

        if g.profiling:
            g.profile_draft_time = True
        return outputs

    def _reorder_pending_tree_cache(self, past_key_values, hidden_indices, pending_tree_size: int, *, input_len: int) -> None:
        g = self.gen
        if hidden_indices is None or int(hidden_indices.numel()) <= 0:
            raise RuntimeError("Deferred reorder expected non-empty hidden_indices_cache in subspec_sd_v2.")

        pending_tree_size = g._resolve_pending_chunk_size(
            hidden_indices,
            int(pending_tree_size),
        )
        max_hidden_idx = int(hidden_indices.max().item())
        if max_hidden_idx >= int(pending_tree_size):
            raise RuntimeError(
                "Invalid deferred-reorder indices in subspec_sd_v2: "
                f"max_hidden_idx={max_hidden_idx}, pending_tree_size={int(pending_tree_size)}, "
                f"seq_len={int(past_key_values.get_seq_length())}, input_len={int(input_len)}"
            )

        max_cache_len = getattr(past_key_values.cache, "max_cache_len", None)
        if max_cache_len is not None:
            base_offset = int(past_key_values.get_seq_length())
            src_max = base_offset + max_hidden_idx
            dest_max = base_offset + int(hidden_indices.size(0)) - 1
            if max(src_max, dest_max) >= int(max_cache_len):
                raise RuntimeError(
                    "Deferred reorder would read/write beyond static cache in subspec_sd_v2: "
                    f"src_max={src_max}, dest_max={dest_max}, max_cache_len={int(max_cache_len)}, "
                    f"seq_len={int(past_key_values.get_seq_length())}, max_hidden_idx={max_hidden_idx}"
                )

        past_key_values.reorder_cache_with_offset(
            hidden_indices,
            offset=past_key_values.get_seq_length(),
            new_chunk_len=pending_tree_size,
            dim=2,
        )
        past_key_values.seq_len += hidden_indices.shape[0]

    def _flush_deferred_tree_cache(
        self, past_key_values, hidden_indices_cache, tree_size: int, *, input_len: int
    ) -> tuple[int, bool, torch.Tensor | None]:
        with nvtx.annotate("kv_reorder"):
            self._reorder_pending_tree_cache(
                past_key_values,
                hidden_indices_cache,
                int(tree_size),
                input_len=int(input_len),
            )
        return 0, False, None

    # ----- SubSpecV2Backend hooks ----- #
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
        return self._flush_deferred_tree_cache(
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
        return self._post_verify(
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
        return self._tree_decoding(
            tree,
            self.past_key_values,
            position_offset=position_offset,
            cache_position=cache_position,
            skip_nodes=skip_nodes,
            device=device,
        )

    def reorder_tail(self, *, hidden_indices_cache, tree_size, root_ind, is_prev_accepted, finished, prune_tokens, disable_post_verify, input_len):
        if (not is_prev_accepted) or finished:
            self._reorder_pending_tree_cache(
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
# Owns the FlashInfer KV-cache + attention methods (were on the FI generator).
# --------------------------------------------------------------------------- #
class FlashInferV2Backend(SubSpecV2Backend):
    method_name = "subspec_sd_v2_fi"

    def __init__(self, gen):
        self.gen = gen
        self.kv_cache_pool = None
        self.request_kv_cache = None
        self.max_cache_len = None
        self._prompt_len = 0

    # ----- backend-specific methods (relocated from the FI generator) ----- #
    def _build_rewrite_batch_position(self, request_kv_cache, *, num_tokens: int, device):
        from ..utils.flashinfer.cache_manager import KvCacheBatchPosition

        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive for rewrite, got {num_tokens}")
        seq_len = int(request_kv_cache.get_seq_length())
        rewrite_start = int(seq_len) - int(num_tokens)
        if rewrite_start < 0:
            raise RuntimeError(
                "Invalid FI rewrite window for post-verify: "
                f"seq_len={seq_len}, num_tokens={num_tokens}, rewrite_start={rewrite_start}"
            )

        kv_page_indices = torch.tensor(
            request_kv_cache.kv_page_indices,
            dtype=torch.int32,
            device=device,
        )
        kv_page_indptr = torch.tensor(
            [0, int(kv_page_indices.numel())],
            dtype=torch.int32,
            device=device,
        )
        kv_last_page_len = torch.tensor(
            [int(request_kv_cache.kv_last_page_len)],
            dtype=torch.int32,
            device=device,
        )
        seq_indptr = torch.tensor([0, int(num_tokens)], dtype=torch.int32, device=device)
        batch_indices = torch.zeros((int(num_tokens),), dtype=torch.int32, device=device)
        positions = torch.arange(
            int(rewrite_start),
            int(seq_len),
            dtype=torch.int32,
            device=device,
        )

        return KvCacheBatchPosition(
            seq_indptr=seq_indptr,
            kv_page_indptr=kv_page_indptr,
            kv_page_indices=kv_page_indices,
            kv_last_page_len=kv_last_page_len,
            batch_indices=batch_indices,
            positions=positions,
        )

    def _draft_tree_decoding(self, tree, request_kv_cache, position_offset, skip_nodes, device, *, append_tokens: bool = True):
        from ..utils.flashinfer.cache_manager import getKvCacheBatchPosition

        g = self.gen
        kv_cache_pool = request_kv_cache.kvCachePool
        tree_input_ids, tree_position_ids, tree_mask = g._prepare_tree_inputs_and_mask(
            tree,
            position_offset=position_offset,
            device=device,
            model_dtype=kv_cache_pool.cache_data[0].dtype,
            non_blocking=True,
            skip_nodes=skip_nodes,
            invert=False,
        )

        num_tokens = int(tree_input_ids.shape[0])
        if num_tokens == 0:
            return None, 0

        with nvtx.annotate("draft_forward", color="red"):
            # For rewrite phases (post-verify), use overwrite semantics by
            # writing into the trailing `[seq_len - num_tokens, seq_len)` window.
            if append_tokens:
                request_kv_cache.increment(num_tokens)
                batch_position = getKvCacheBatchPosition(
                    request_kv_caches=[request_kv_cache],
                    mode="tree",
                    device=device,
                    treeTokens=num_tokens,
                )
            else:
                batch_position = self._build_rewrite_batch_position(
                    request_kv_cache,
                    num_tokens=int(num_tokens),
                    device=device,
                )
            g.flashinferWrapper.prepareAttention(
                "tree",
                batch_position,
                kv_cache_pool.page_len,
                "NONE",
                kv_cache_pool.cache_data[0].dtype,
                attention_mask=tree_mask,
            )
            logits = g.draft_model(
                tree_input_ids.unsqueeze(0),
                with_softmax=False,
                past_key_values=None,
                position_ids=tree_position_ids.unsqueeze(0),
                use_cache=False,
                kvCachePool=kv_cache_pool,
                batch_position=batch_position,
                mode="tree",
                flashinferWrapper=g.flashinferWrapper,
            )
        return logits, int(num_tokens)

    def _post_verify(self, tree, root_ind, request_kv_cache, position_offset, last_tree_depth, skip_nodes, logits_processor, device):
        g = self.gen
        debug_enabled = bool(
            getattr(g, "step_trace_enabled", False)
            and getattr(g, "step_trace_debug_verify", False)
        )
        debug_extra: dict = {}

        verify_tokens_expected = max(0, int(tree.size()) - int(skip_nodes))
        if int(verify_tokens_expected) <= 0:
            if debug_enabled:
                g._last_post_verify_debug = debug_extra
            return tree, None

        # Ensure synchronous post-verify refill mutates the live request cache,
        # not an overlap-clone snapshot from the previous round.
        g.draft_model.request_kv_cache = request_kv_cache
        if debug_enabled:
            debug_extra["post_verify_rewrite_req_len_before"] = int(
                request_kv_cache.get_seq_length()
            )
        expected_window_end = int(position_offset) + int(tree.size())
        # Post-verify should rewrite the existing suffix window, not append a
        # second copy after it.
        self._sync_request_cache_to_len(
            request_kv_cache,
            expected_len=int(expected_window_end),
        )
        req_len_after_sync = int(request_kv_cache.get_seq_length())
        rewrite_start = int(req_len_after_sync) - int(verify_tokens_expected)
        if rewrite_start < 0:
            raise RuntimeError(
                "Invalid FI post-verify rewrite start: "
                f"req_len_after_sync={req_len_after_sync}, "
                f"verify_tokens_expected={int(verify_tokens_expected)}, "
                f"rewrite_start={int(rewrite_start)}"
            )
        if debug_enabled:
            debug_extra["post_verify_rewrite_req_len_after_sync"] = int(req_len_after_sync)
            debug_extra["post_verify_rewrite_window_start"] = int(rewrite_start)
            debug_extra["post_verify_rewrite_window_end"] = int(req_len_after_sync - 1)
        next_token_logits, decoded_tokens = self._draft_tree_decoding(
            tree,
            request_kv_cache,
            position_offset=position_offset,
            skip_nodes=skip_nodes,
            device=device,
            append_tokens=False,
        )
        if debug_enabled:
            debug_extra["post_verify_rewrite_req_len_after_decode"] = int(
                request_kv_cache.get_seq_length()
            )
        if next_token_logits is None:
            if debug_enabled:
                g._last_post_verify_debug = debug_extra
            return tree, None
        if int(decoded_tokens) != int(verify_tokens_expected):
            raise RuntimeError(
                "FI post-verify token count mismatch after rewrite: "
                f"decoded_tokens={int(decoded_tokens)}, expected={int(verify_tokens_expected)}"
            )
        if int(next_token_logits.shape[1]) != int(decoded_tokens):
            raise RuntimeError(
                "FI post-verify draft logits length mismatch: "
                f"logits_len={int(next_token_logits.shape[1])}, decoded_tokens={int(decoded_tokens)}"
            )

        _, _, (_, accept_len) = g._verify(
            tree,
            root_ind,
            next_token_logits,
            logits_processor,
            False,
            skip_nodes=skip_nodes,
        )

        accept_len = int(accept_len)
        kept_old_indices = tree.prune_to_depth(int(last_tree_depth) + accept_len)
        if debug_enabled:
            debug_extra["post_verify_accept_len"] = int(accept_len)
            debug_extra["post_verify_kept_old_len"] = int(kept_old_indices.numel())
        # Post-verify prune changes the live tree window. Clamp request-cache
        # metadata before postspec refill so rebuilt frontier positions match
        # the pruned tree geometry.
        self._sync_request_cache_to_len(
            request_kv_cache,
            expected_len=int(position_offset) + int(tree.size()),
        )

        refill_steps = max(0, int(g.draft_params.max_depth) - accept_len)
        if refill_steps > 0:
            with nvtx.annotate("postspec_refill", color="cyan"):
                g.draft_model.init_postspec()
                for _ in range(refill_steps):
                    if not g.draft_model.postspec():
                        break
            tree = g.draft_model.update_tree_after_post()
        if debug_enabled:
            debug_extra["post_verify_tree_token_hash"] = int(
                g._tree_token_hash(tree=tree, skip_nodes=0)
            )
            g._last_post_verify_debug = debug_extra

        g.post_verify_count += 1
        return tree, kept_old_indices

    def _tree_decoding(self, tree, request_kv_cache, position_offset, skip_nodes, device):
        g = self.gen
        kv_cache_pool = request_kv_cache.kvCachePool
        tree_input_ids, tree_position_ids, tree_mask = g._prepare_tree_inputs_and_mask(
            tree,
            position_offset=position_offset,
            device=device,
            model_dtype=kv_cache_pool.cache_data[0].dtype,
            non_blocking=True,
            skip_nodes=skip_nodes,
            invert=False,
        )

        num_tokens = int(tree_input_ids.shape[0])
        if num_tokens == 0:
            return None
        batch_position = self._build_rewrite_batch_position(
            request_kv_cache,
            num_tokens=int(num_tokens),
            device=device,
        )

        with nvtx.annotate("target_forward", color="red"):
            g.flashinferWrapper.prepareAttention(
                "tree",
                batch_position,
                kv_cache_pool.page_len,
                "NONE",
                kv_cache_pool.cache_data[0].dtype,
                attention_mask=tree_mask,
            )
            outputs = g.target_model(
                input_ids=tree_input_ids.unsqueeze(0),
                past_key_values=None,
                position_ids=tree_position_ids.unsqueeze(0),
                output_hidden_states=True,
                use_cache=False,
                kvCachePool=kv_cache_pool,
                batch_position=batch_position,
                mode="tree",
                flashinferWrapper=g.flashinferWrapper,
            )
        return outputs

    def _reorder_pending_tree_cache(self, request_kv_cache, hidden_indices, pending_tree_size: int) -> None:
        g = self.gen
        if hidden_indices is None or int(hidden_indices.numel()) <= 0:
            raise RuntimeError(
                "Deferred reorder expected non-empty hidden_indices_cache in subspec_sd_v2_fi."
            )

        pending_tree_size = g._resolve_pending_chunk_size(
            hidden_indices,
            int(pending_tree_size),
        )
        max_hidden_idx = int(hidden_indices.max().item())
        if max_hidden_idx >= int(pending_tree_size):
            raise RuntimeError(
                "Invalid deferred-reorder indices in subspec_sd_v2_fi: "
                f"max_hidden_idx={max_hidden_idx}, pending_tree_size={int(pending_tree_size)}"
            )

        seq_len = int(request_kv_cache.get_seq_length())
        base_offset = int(seq_len) - int(pending_tree_size) + 1
        if base_offset <= 0:
            raise RuntimeError(
                "Invalid deferred-reorder offset in subspec_sd_v2_fi: "
                f"seq_len={seq_len}, pending_tree_size={int(pending_tree_size)}, "
                f"computed_offset={base_offset}"
            )

        request_kv_cache.reorder_cache_with_offset(
            hidden_indices,
            offset=base_offset,
            num_new_tokens=int(pending_tree_size),
        )

    def _flush_deferred_tree_cache(
        self, request_kv_cache, hidden_indices_cache, tree_size: int
    ) -> tuple[int, bool, torch.Tensor | None]:
        with nvtx.annotate("kv_reorder"):
            self._reorder_pending_tree_cache(
                request_kv_cache,
                hidden_indices_cache,
                int(tree_size),
            )
        return 0, False, None

    def _sync_request_cache_to_len(self, request_kv_cache, *, expected_len: int) -> None:
        g = self.gen
        expected_len = int(expected_len)
        cache_capacity = g._request_cache_capacity(request_kv_cache)
        if cache_capacity is not None:
            expected_len = min(int(expected_len), int(cache_capacity))
        current_len = int(request_kv_cache.get_seq_length())
        if current_len > expected_len:
            request_kv_cache.decrement(int(current_len - expected_len))
            return
        if current_len < expected_len:
            # Budget-capped rounds can carry a tree whose full window was not
            # materialized in the request cache metadata yet; grow to target.
            request_kv_cache.increment(int(expected_len - current_len))

    def _sync_request_cache_to_tree(self, request_kv_cache, *, position_offset: int, tree_size: int) -> None:
        """Clamp request metadata to the current `[prefix + tree]` footprint."""
        self._sync_request_cache_to_len(
            request_kv_cache,
            expected_len=int(position_offset) + int(tree_size),
        )

    def _commit_seed_postspec_before_post_verify(self, *, tree, request_kv_cache, position_offset: int):
        g = self.gen
        # Commit-seed one deterministic postspec step from the current
        # committed `[prefix + tree]` boundary before carry-over post-verify.
        self._sync_request_cache_to_tree(
            request_kv_cache,
            position_offset=int(position_offset),
            tree_size=int(tree.size()),
        )
        g.draft_model.request_kv_cache = request_kv_cache
        g.draft_model.init_postspec(rebuild_frontier=True)
        with nvtx.annotate("postspec_commit_seed", color="cyan"):
            g.draft_model.postspec()
        tree = g.draft_model.update_tree_after_post()
        self._sync_request_cache_to_tree(
            request_kv_cache,
            position_offset=int(position_offset),
            tree_size=int(tree.size()),
        )
        return tree

    # ----- SubSpecV2Backend hooks ----- #
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
        g = self.gen
        return g.draft_model.speculate(
            last_token_id,
            request_kv_cache=self.request_kv_cache,
            flashinferWrapper=g.flashinferWrapper,
        )

    def sync_to_tree(self, *, position_offset, tree_size):
        self._sync_request_cache_to_tree(
            self.request_kv_cache,
            position_offset=int(position_offset),
            tree_size=int(tree_size),
        )

    def flush_headroom_cap(self):
        return self.gen._request_cache_capacity(self.request_kv_cache)

    def flush_deferred(self, hidden_indices_cache, tree_size, *, input_len):
        return self._flush_deferred_tree_cache(
            self.request_kv_cache,
            hidden_indices_cache,
            int(tree_size),
        )

    def commit_seed(self, tree, *, position_offset):
        return self._commit_seed_postspec_before_post_verify(
            tree=tree,
            request_kv_cache=self.request_kv_cache,
            position_offset=int(position_offset),
        )

    def post_verify(self, tree, root_ind, *, position_offset, skip_nodes, last_tree_depth, logits_processor, device):
        return self._post_verify(
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
        self._sync_request_cache_to_tree(
            self.request_kv_cache,
            position_offset=int(position_offset),
            tree_size=int(tree.size()),
        )
        outputs = self._tree_decoding(
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
            return self._flush_deferred_tree_cache(
                self.request_kv_cache,
                hidden_indices_cache,
                int(tree_size),
            )
        elif (not is_prev_accepted) or finished:
            self._reorder_pending_tree_cache(
                self.request_kv_cache,
                hidden_indices_cache,
                int(tree_size),
            )
            if finished and int(prune_tokens) > 0:
                self.request_kv_cache.decrement(int(prune_tokens))
        return root_ind, is_prev_accepted, hidden_indices_cache

"""Shared v1 SubSpec speculative-decoding loop, driven through a backend seam.

One generator class (`generators/subspec_sd.py::SubSpecSDGenerator`) drives this loop for
both attention backends; the backend is chosen by the `backend:` config field. Each
`SpecDecodeBackend` adapter below **owns** the KV-cache lifecycle + attention execution
that differs between SDPA (static/dynamic `Cache`) and FlashInfer (paged `RequestKvCache`):
`SdpaV1Backend` owns the target-cache reorder, `FlashInferV1Backend` owns the FI tree
forward + draft speculate. Shared, backend-agnostic helpers (`_verify`,
`_prepare_tree_inputs_and_mask`, the SDPA `_tree_decoding`/`_speculate` on
`ClassicSDGeneratorBase`, the FI request-cache mixin) stay on the generator; the adapters
call back into it via `self.gen`.
"""
from __future__ import annotations

import torch
import nvtx

from .spec_decode_backend import SpecDecodeBackend


# --------------------------------------------------------------------------- #
# The one shared v1 loop. Drives any SpecDecodeBackend; no backend branching.
# --------------------------------------------------------------------------- #
def run_subspec_v1_generate(
    gen,
    backend: "SpecDecodeBackend",
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
                last_token_id = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
                prev_kv_len = backend.current_kv_len()
                tree = backend.speculate(last_token_id)
                tree_size_before_cap = int(tree.size())
                decoded_tree_size = gen._cap_tree_to_budget(tree, input_ids, stopping_criteria)
                tree_size_after_cap = int(tree.size())
                backend.after_cap(tree_size_before_cap, tree_size_after_cap)
                if decoded_tree_size <= 0:
                    break

            with nvtx.annotate("target_decode", color="orange"):
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
                sampled_tokens, hidden_indices, (_, accept_len) = gen._verify(
                    tree, 0, next_token_logits, logits_processor, do_sample,
                )
                sampled_tokens = sampled_tokens.to(input_ids.device)
                del next_token_logits
                gen._append_step_trace(
                    is_prev_accepted=False,
                    skip_nodes=0,
                    tree_size_before_cap=int(tree_size_before_cap),
                    tree_size_after_cap=int(tree_size_after_cap),
                    decoded_tree_size=int(decoded_tree_size),
                    root_ind_in=0,
                    root_ind_out=-1,
                    accept_len=int(accept_len),
                    hidden_indices_len=int(hidden_indices.numel()),
                    post_verify_used=False,
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
# SDPA / static-cache adapter. Unbounded headroom; seq_len bookkeeping.
# --------------------------------------------------------------------------- #
class SdpaV1Backend(SpecDecodeBackend):
    def __init__(self, gen):
        self.gen = gen
        self.past_key_values = None

    def begin(self, input_ids, past_key_values):
        g = self.gen
        self.past_key_values = past_key_values
        max_cache_len = getattr(past_key_values.cache, "max_cache_len", None)
        g.draft_model.set_past_key_values(past_key_values)
        g._init_tree_mask(g.draft_params.max_verify_tokens, max_cache_len, device=input_ids.device)
        outputs = g._chunked_prefill_forward(
            input_ids, past_key_values,
            prefill_chunk_size=g.prefill_chunk_size, use_position_ids=True,
        )
        return outputs.logits

    def current_kv_len(self):
        return int(self.past_key_values.get_seq_length())

    def decode_headroom(self):
        return None

    def speculate(self, last_token_id):
        return self.gen._speculate(last_token_id)

    def after_cap(self, tree_size_before, tree_size_after):
        return None

    def tree_forward(self, tree, *, position_offset, decoded_tree_size, device):
        g = self.gen
        pkv = self.past_key_values
        if g.cache_implementation == "dynamic":
            pkv.crop(pkv.get_seq_length())
        cache_position = torch.arange(
            position_offset, position_offset + int(decoded_tree_size),
            dtype=torch.long, device=device,
        )
        return g._tree_decoding(
            tree, pkv, position_offset=position_offset,
            cache_position=cache_position, device=device,
        )

    def _commit_target_cache_reorder(
        self,
        past_key_values,
        *,
        hidden_indices: torch.Tensor,
        prev_kv_len: int,
        decoded_tree_size: int,
        finished: bool,
        prune_tokens: int,
    ) -> None:
        past_key_values.reorder_cache_with_offset(
            hidden_indices,
            offset=int(prev_kv_len),
            new_chunk_len=int(decoded_tree_size),
            dim=2,
        )
        past_key_values.seq_len += int(hidden_indices.shape[0])
        if finished:
            past_key_values.seq_len -= int(prune_tokens)

    def commit(self, *, hidden_indices, prev_kv_len, decoded_tree_size, finished, prune_tokens):
        self._commit_target_cache_reorder(
            self.past_key_values,
            hidden_indices=hidden_indices,
            prev_kv_len=int(prev_kv_len),
            decoded_tree_size=int(decoded_tree_size),
            finished=bool(finished),
            prune_tokens=int(prune_tokens),
        )

    def finalize(self, input_ids):
        return None


# --------------------------------------------------------------------------- #
# FlashInfer / paged adapter. Bounded headroom; RequestKvCache lifecycle.
# --------------------------------------------------------------------------- #
class FlashInferV1Backend(SpecDecodeBackend):
    def __init__(self, gen):
        self.gen = gen
        self.request_kv_cache = None
        self._prompt_len = 0

    def begin(self, input_ids, past_key_values):
        from ..utils.flashinfer.cache_manager import RequestKvCache
        from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper
        from ..utils.flashinfer.prefill import flashinfer_chunked_prefill

        g = self.gen
        self._prompt_len = int(input_ids.shape[1])
        max_cache_len = getattr(past_key_values, "max_cache_len", None)
        g._init_tree_mask(g.draft_params.max_verify_tokens, max_cache_len, device=input_ids.device)

        if not hasattr(g, "flashinferWrapper"):
            g.flashinferWrapper = FlashinferAttentionWrapper(
                g.target_model.config.num_attention_heads,
                g.target_model.config.num_key_value_heads,
                g.target_model.config.hidden_size,
                past_key_values.page_len,
                tree_use_cuda_graph=False,
            )
        g.kvCachePool = past_key_values
        self.request_kv_cache = g._ensure_request_kv_cache(
            attr_name="_fi_request_kv_cache",
            request_cls=RequestKvCache,
            kv_cache_pool=g.kvCachePool,
            input_ids_len=int(input_ids.shape[1]),
            input_ids=input_ids,
            tokens_attr_name="_fi_request_tokens",
            reuse_len_attr_name="_fi_request_reuse_len",
        )
        outputs = flashinfer_chunked_prefill(
            target_model=g.target_model,
            flashinfer_wrapper=g.flashinferWrapper,
            input_ids=input_ids,
            kv_cache_pool=g.kvCachePool,
            request_kv_cache=self.request_kv_cache,
            prefill_chunk_size=g.prefill_chunk_size,
        )
        return outputs.logits

    def current_kv_len(self):
        return int(self.request_kv_cache.get_seq_length()) + 1

    def decode_headroom(self):
        return self.gen._request_cache_headroom(self.request_kv_cache)

    def speculate(self, last_token_id):
        g = self.gen
        return g.draft_model.speculate(
            last_token_id,
            request_kv_cache=self.request_kv_cache,
            flashinferWrapper=g.flashinferWrapper,
        )

    def after_cap(self, tree_size_before, tree_size_after):
        self.gen._sync_request_cache_after_tree_truncation(
            self.request_kv_cache,
            tree_size_before=int(tree_size_before),
            tree_size_after=int(tree_size_after),
        )

    def _tree_decoding(self, tree, request_kv_cache, position_offset, cache_position, device):
        from ..utils.flashinfer.cache_manager import getKvCacheBatchPosition

        g = self.gen
        tree_input_ids, tree_position_ids, tree_mask = g._prepare_tree_inputs_and_mask(
            tree,
            position_offset=position_offset,
            device=device,
            model_dtype=g.target_model.model.dtype,
            non_blocking=True,
            invert=False,
        )

        # Target model forward
        with nvtx.annotate("target_forward", color="red"):
            num_tokens = int(tree_input_ids.shape[0])
            if num_tokens == 0:
                return None
            kvCachePool = request_kv_cache.kvCachePool

            # The draft appends KV only for the levels it forwards, so kv_len may be
            # short of the full tree footprint. Grow it so the target rewrite window
            # [kv_len - num_tokens, kv_len) lands exactly on [position_offset, +tree)
            # instead of eating into committed prefix KV.
            _target_len = int(position_offset) + int(num_tokens)
            _cur_len = int(request_kv_cache.get_seq_length())
            if _cur_len < _target_len:
                request_kv_cache.increment(_target_len - _cur_len)
            elif _cur_len > _target_len:
                request_kv_cache.decrement(_cur_len - _target_len)

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

    def tree_forward(self, tree, *, position_offset, decoded_tree_size, device):
        return self._tree_decoding(
            tree, self.request_kv_cache,
            position_offset=position_offset, cache_position=None, device=device,
        )

    def commit(self, *, hidden_indices, prev_kv_len, decoded_tree_size, finished, prune_tokens):
        self.request_kv_cache.reorder_cache_with_offset(
            hidden_indices, offset=int(prev_kv_len), num_new_tokens=int(decoded_tree_size),
        )
        if finished and int(prune_tokens) > 0:
            self.request_kv_cache.decrement(int(prune_tokens))

    def finalize(self, input_ids):
        g = self.gen
        g._remember_request_cache_tokens(tokens_attr_name="_fi_request_tokens", input_ids=input_ids)
        setattr(g, "_fi_request_reuse_len", int(self._prompt_len))

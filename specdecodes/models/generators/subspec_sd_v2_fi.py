import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import nvtx

from .classic_sd import ClassicSDGeneratorBase
from .flashinfer_cache_mixin import FlashInferCacheMixin
from .subspec_sd_v2_loop import run_subspec_v2_generate, FlashInferV2Backend
from ..utils.mixin import SDProfilingMixin
from ..utils.flashinfer.cache_manager import (
    KvCacheBatchPosition,
    getKvCacheBatchPosition,
)


class SubSpecSDGeneratorBase(FlashInferCacheMixin, ClassicSDGeneratorBase):
    def _build_rewrite_batch_position(
        self,
        request_kv_cache,
        *,
        num_tokens: int,
        device,
    ) -> KvCacheBatchPosition:
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

    def init_cuda_graph_runner(self, device, kvCachePool=None):
        if hasattr(self.draft_model, "init_cuda_graph_runner") and callable(
            self.draft_model.init_cuda_graph_runner
        ):
            self.draft_model.init_cuda_graph_runner(device=device)

    def _draft_tree_decoding(
        self,
        tree,
        request_kv_cache,
        position_offset,
        skip_nodes,
        device,
        *,
        append_tokens: bool = True,
    ):
        kv_cache_pool = request_kv_cache.kvCachePool
        tree_input_ids, tree_position_ids, tree_mask = self._prepare_tree_inputs_and_mask(
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
            self.flashinferWrapper.prepareAttention(
                "tree",
                batch_position,
                kv_cache_pool.page_len,
                "NONE",
                kv_cache_pool.cache_data[0].dtype,
                attention_mask=tree_mask,
            )
            logits = self.draft_model(
                tree_input_ids.unsqueeze(0),
                with_softmax=False,
                past_key_values=None,
                position_ids=tree_position_ids.unsqueeze(0),
                use_cache=False,
                kvCachePool=kv_cache_pool,
                batch_position=batch_position,
                mode="tree",
                flashinferWrapper=self.flashinferWrapper,
            )
        return logits, int(num_tokens)

    def _post_verify(
        self,
        tree,
        root_ind,
        request_kv_cache,
        position_offset,
        last_tree_depth,
        skip_nodes,
        logits_processor,
        device,
    ):
        debug_enabled = bool(
            getattr(self, "step_trace_enabled", False)
            and getattr(self, "step_trace_debug_verify", False)
        )
        debug_extra: dict = {}

        verify_tokens_expected = max(0, int(tree.size()) - int(skip_nodes))
        if int(verify_tokens_expected) <= 0:
            if debug_enabled:
                self._last_post_verify_debug = debug_extra
            return tree, None

        # Ensure synchronous post-verify refill mutates the live request cache,
        # not an overlap-clone snapshot from the previous round.
        self.draft_model.request_kv_cache = request_kv_cache
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
                self._last_post_verify_debug = debug_extra
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

        _, _, (_, accept_len) = self._verify(
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

        refill_steps = max(0, int(self.draft_params.max_depth) - accept_len)
        if refill_steps > 0:
            with nvtx.annotate("postspec_refill", color="cyan"):
                self.draft_model.init_postspec()
                for _ in range(refill_steps):
                    if not self.draft_model.postspec():
                        break
            tree = self.draft_model.update_tree_after_post()
        if debug_enabled:
            debug_extra["post_verify_tree_token_hash"] = int(
                self._tree_token_hash(tree=tree, skip_nodes=0)
            )
            self._last_post_verify_debug = debug_extra

        self.post_verify_count += 1
        return tree, kept_old_indices

    def _tree_decoding(
        self,
        tree,
        request_kv_cache,
        position_offset,
        skip_nodes,
        device,
    ):
        kv_cache_pool = request_kv_cache.kvCachePool
        tree_input_ids, tree_position_ids, tree_mask = self._prepare_tree_inputs_and_mask(
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
            self.flashinferWrapper.prepareAttention(
                "tree",
                batch_position,
                kv_cache_pool.page_len,
                "NONE",
                kv_cache_pool.cache_data[0].dtype,
                attention_mask=tree_mask,
            )
            outputs = self.target_model(
                input_ids=tree_input_ids.unsqueeze(0),
                past_key_values=None,
                position_ids=tree_position_ids.unsqueeze(0),
                output_hidden_states=True,
                use_cache=False,
                kvCachePool=kv_cache_pool,
                batch_position=batch_position,
                mode="tree",
                flashinferWrapper=self.flashinferWrapper,
            )
        return outputs

    def _reorder_pending_tree_cache(
        self,
        request_kv_cache,
        hidden_indices,
        pending_tree_size: int,
    ) -> None:
        if hidden_indices is None or int(hidden_indices.numel()) <= 0:
            raise RuntimeError(
                "Deferred reorder expected non-empty hidden_indices_cache in subspec_sd_v2_fi."
            )

        pending_tree_size = self._resolve_pending_chunk_size(
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
        self,
        request_kv_cache,
        hidden_indices_cache,
        tree_size: int,
    ) -> tuple[int, bool, torch.Tensor | None]:
        with nvtx.annotate("kv_reorder"):
            self._reorder_pending_tree_cache(
                request_kv_cache,
                hidden_indices_cache,
                int(tree_size),
            )
        return 0, False, None

    def _sync_request_cache_to_len(
        self,
        request_kv_cache,
        *,
        expected_len: int,
    ) -> None:
        expected_len = int(expected_len)
        cache_capacity = self._request_cache_capacity(request_kv_cache)
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

    def _sync_request_cache_to_tree(
        self,
        request_kv_cache,
        *,
        position_offset: int,
        tree_size: int,
    ) -> None:
        """Clamp request metadata to the current `[prefix + tree]` footprint."""
        self._sync_request_cache_to_len(
            request_kv_cache,
            expected_len=int(position_offset) + int(tree_size),
        )

    def _speculate(self, input_ids, request_kv_cache):
        return self.draft_model.speculate(
            input_ids,
            request_kv_cache=request_kv_cache,
            flashinferWrapper=self.flashinferWrapper,
        )

    def _commit_seed_postspec_before_post_verify(
        self,
        *,
        tree,
        request_kv_cache,
        position_offset: int,
    ):
        # Commit-seed one deterministic postspec step from the current
        # committed `[prefix + tree]` boundary before carry-over post-verify.
        self._sync_request_cache_to_tree(
            request_kv_cache,
            position_offset=int(position_offset),
            tree_size=int(tree.size()),
        )
        self.draft_model.request_kv_cache = request_kv_cache
        self.draft_model.init_postspec(rebuild_frontier=True)
        with nvtx.annotate("postspec_commit_seed", color="cyan"):
            self.draft_model.postspec()
        tree = self.draft_model.update_tree_after_post()
        self._sync_request_cache_to_tree(
            request_kv_cache,
            position_offset=int(position_offset),
            tree_size=int(tree.size()),
        )
        return tree

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """Generate a token sequence with SubSpec v2 (post-verify) speculative decoding.

        The loop itself is shared with the SDPA variant; see
        `subspec_sd_v2_loop.run_subspec_v2_generate`. This backend drives the paged
        `RequestKvCache` + FlashInfer attention-wrapper path (request-cache syncs, the
        commit-seed postspec step, and capacity-bounded flush decisions).
        """
        return run_subspec_v2_generate(
            self,
            FlashInferV2Backend(self),
            input_ids,
            stopping_criteria,
            logits_processor,
            do_sample,
            **model_kwargs,
        )


class SubSpecSDGenerator(SDProfilingMixin, SubSpecSDGeneratorBase):
    pass

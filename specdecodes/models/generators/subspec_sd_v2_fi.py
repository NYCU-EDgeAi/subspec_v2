import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import nvtx

from .classic_sd import ClassicSDGeneratorBase
from .flashinfer_cache_mixin import FlashInferCacheMixin
from ..utils.mixin import SDProfilingMixin
from ..utils.flashinfer.cache_manager import (
    KvCacheBatchPosition,
    RequestKvCache,
    getKvCacheBatchPosition,
)
from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper
from ..utils.flashinfer.prefill import flashinfer_chunked_prefill


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
        assert self.target_model is not None, "target_model must be provided"
        assert self.draft_model is not None, "draft_model must be provided"
        assert self.tokenizer is not None, "tokenizer must be provided"

        input_ids = input_ids.clone()
        batch_size, _ = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."
        prompt_input_len = int(input_ids.shape[1])

        if stopping_criteria.max_length is None and self.cache_implementation == "static":
            raise ValueError(
                "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
            )

        if model_kwargs.get("past_key_values") is None:
            raise ValueError("past_key_values should be provided")

        kv_cache_pool = model_kwargs["past_key_values"]
        max_cache_len = getattr(kv_cache_pool, "max_cache_len", None)
        stream_callback = model_kwargs.get("stream_callback", None)
        self._init_step_trace()

        if not hasattr(self, "flashinferWrapper"):
            self.flashinferWrapper = FlashinferAttentionWrapper(
                self.target_model.config.num_attention_heads,
                self.target_model.config.num_key_value_heads,
                self.target_model.config.hidden_size,
                kv_cache_pool.page_len,
                # SubSpec v2 overlap has variable tree row counts; keep FlashInfer
                # tree planning dynamic instead of pinning to the first row count.
                tree_use_cuda_graph=False,
            )

        with nvtx.annotate("prefill_chunked", color="orange"):
            self._init_tree_mask(
                int(self.draft_params.max_verify_tokens) * 2,
                max_cache_len,
                device=input_ids.device,
            )
            request_kv_cache = self._ensure_request_kv_cache(
                attr_name="_fi_v2_request_kv_cache",
                request_cls=RequestKvCache,
                kv_cache_pool=kv_cache_pool,
                input_ids_len=int(input_ids.shape[1]),
                input_ids=input_ids,
                tokens_attr_name="_fi_v2_request_tokens",
                reuse_len_attr_name="_fi_v2_request_reuse_len",
            )
            outputs = flashinfer_chunked_prefill(
                target_model=self.target_model,
                flashinfer_wrapper=self.flashinferWrapper,
                input_ids=input_ids,
                kv_cache_pool=kv_cache_pool,
                request_kv_cache=request_kv_cache,
                prefill_chunk_size=self.prefill_chunk_size,
            )
            next_token_logits = outputs.logits
            del outputs

        remaining = self._remaining_token_budget(input_ids, stopping_criteria)
        if remaining is not None and int(remaining) <= 0:
            self._remember_request_cache_tokens(
                tokens_attr_name="_fi_v2_request_tokens",
                input_ids=input_ids,
            )
            setattr(self, "_fi_v2_request_reuse_len", int(prompt_input_len))
            return input_ids

        with nvtx.annotate("sample"):
            sampled_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

        with nvtx.annotate("state_update"):
            input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
            self._maybe_stream(stream_callback, sampled_tokens)

        with nvtx.annotate("decode_loop"):
            self.post_verify_count = 0
            self.speculate_count = 0
            disable_post_verify = bool(self.generator_kwargs.get("disable_post_verify", False))

            finished = False
            is_prev_accepted = False
            hidden_indices_cache = None
            last_tree_size = 0
            last_tree_depth = 0
            root_ind = 0
            position_offset = int(input_ids.shape[1]) - 1

            while not finished:
                remaining = self._remaining_token_budget(input_ids, stopping_criteria)
                if remaining is not None and int(remaining) <= 0:
                    break

                post_verify_used = False
                if is_prev_accepted:
                    skip_nodes = int(last_tree_size)

                    pending_post_tokens = int(tree.size()) - int(skip_nodes)
                    cache_capacity = self._request_cache_capacity(request_kv_cache)
                    should_flush_deferred = bool(disable_post_verify) or int(pending_post_tokens) <= 0
                    if (not should_flush_deferred) and (cache_capacity is not None):
                        cache_headroom = max(
                            0,
                            int(cache_capacity) - (int(position_offset) + int(skip_nodes)),
                        )
                        should_flush_deferred = int(cache_headroom) < int(pending_post_tokens)
                    if should_flush_deferred:
                        root_ind, is_prev_accepted, hidden_indices_cache = self._flush_deferred_tree_cache(
                            request_kv_cache,
                            hidden_indices_cache,
                            int(tree.size()),
                        )
                        continue

                    tree = self._commit_seed_postspec_before_post_verify(
                        tree=tree,
                        request_kv_cache=request_kv_cache,
                        position_offset=int(position_offset),
                    )
                    post_verify_used = True
                    with nvtx.annotate("post_verify", color="cyan"):
                        tree, kept_old_indices = self._post_verify(
                            tree,
                            int(root_ind),
                            request_kv_cache,
                            position_offset,
                            int(last_tree_depth),
                            int(skip_nodes),
                            logits_processor,
                            input_ids.device,
                        )
                    hidden_indices_cache = self._remap_hidden_indices_after_tree_prune(
                        hidden_indices_cache,
                        kept_old_indices,
                        method_name="subspec_sd_v2_fi",
                    )
                    if int(tree.size()) <= int(skip_nodes):
                        root_ind, is_prev_accepted, hidden_indices_cache = self._flush_deferred_tree_cache(
                            request_kv_cache,
                            hidden_indices_cache,
                            int(tree.size()),
                        )
                        continue

                    last_tree_size = int(tree.size())
                    last_tree_depth = int(tree.get_depth())

                else:
                    self.speculate_count += 1
                    with nvtx.annotate("speculate", color="cyan"):
                        last_token_id = sampled_tokens[:, -1:].clone(
                            memory_format=torch.contiguous_format
                        )
                        tree = self._speculate(last_token_id, request_kv_cache)

                    position_offset = int(input_ids.shape[1]) - 1
                    self._sync_request_cache_to_tree(
                        request_kv_cache,
                        position_offset=int(position_offset),
                        tree_size=int(tree.size()),
                    )
                    last_tree_size = int(tree.size())
                    last_tree_depth = int(tree.get_depth())
                    skip_nodes = 0

                tree_size_before_cap = int(tree.size())
                decoded_tree_size = self._cap_tree_to_budget(
                    tree,
                    input_ids,
                    stopping_criteria,
                    skip_nodes=int(skip_nodes),
                )
                tree_size_after_cap = int(tree.size())
                self._sync_request_cache_after_tree_truncation(
                    request_kv_cache,
                    tree_size_before=tree_size_before_cap,
                    tree_size_after=tree_size_after_cap,
                )
                if int(decoded_tree_size) <= 0:
                    break
                last_tree_size = int(tree.size())

                with nvtx.annotate("target_decode", color="orange"):
                    self.draft_model.init_postspec()
                    self._sync_request_cache_to_tree(
                        request_kv_cache,
                        position_offset=int(position_offset),
                        tree_size=int(tree.size()),
                    )
                    outputs = self._tree_decoding(
                        tree,
                        request_kv_cache,
                        position_offset=int(position_offset),
                        skip_nodes=int(skip_nodes),
                        device=input_ids.device,
                    )
                    next_token_logits = outputs.logits if outputs is not None else None
                    if next_token_logits is not None and int(next_token_logits.shape[1]) != int(decoded_tree_size):
                        raise RuntimeError(
                            "FI target tree logits length mismatch: "
                            f"logits_len={int(next_token_logits.shape[1])}, "
                            f"decoded_tree_size={int(decoded_tree_size)}, "
                            f"skip_nodes={int(skip_nodes)}, tree_size={int(tree.size())}"
                        )
                    del outputs

                with nvtx.annotate("postspec_update", color="cyan"):
                    tree = self.draft_model.update_tree_after_post()
                    self._sync_request_cache_to_tree(
                        request_kv_cache,
                        position_offset=int(position_offset),
                        tree_size=int(tree.size()),
                    )

                with nvtx.annotate("verify"):
                    root_ind_in = int(root_ind) if is_prev_accepted else 0
                    step_trace_extra = self._build_verify_debug_trace(
                        tree=tree,
                        next_token_logits=next_token_logits,
                        skip_nodes=int(skip_nodes),
                    )
                    sampled_tokens, hidden_indices, (_, accept_len) = self._verify(
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
                self._append_step_trace(
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
                    finished, input_ids, kept, prune_tokens = self._apply_tokenwise_stopping_criteria(
                        input_ids=input_ids,
                        sampled_tokens=sampled_tokens,
                        stopping_criteria=stopping_criteria,
                    )
                if kept.numel() > 0:
                    self._maybe_stream(stream_callback, kept)

                with nvtx.annotate("kv_reorder"):
                    if disable_post_verify and bool(is_prev_accepted) and (not bool(finished)):
                        root_ind, is_prev_accepted, hidden_indices_cache = self._flush_deferred_tree_cache(
                            request_kv_cache,
                            hidden_indices_cache,
                            int(tree.size()),
                        )
                    elif (not is_prev_accepted) or finished:
                        self._reorder_pending_tree_cache(
                            request_kv_cache,
                            hidden_indices_cache,
                            int(tree.size()),
                        )
                        if finished and int(prune_tokens) > 0:
                            request_kv_cache.decrement(int(prune_tokens))

            self.post_verify_count = int(self.post_verify_count)
            self.speculate_count = int(self.speculate_count)

        self._remember_request_cache_tokens(
            tokens_attr_name="_fi_v2_request_tokens",
            input_ids=input_ids,
        )
        setattr(self, "_fi_v2_request_reuse_len", int(prompt_input_len))
        return input_ids


class SubSpecSDGenerator(SDProfilingMixin, SubSpecSDGeneratorBase):
    pass

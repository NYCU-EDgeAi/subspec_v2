import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import nvtx

from .classic_sd import ClassicSDGeneratorBase
from ..utils.mixin import SDProfilingMixin


class SubSpecSDGeneratorBase(ClassicSDGeneratorBase):
    def _draft_tree_decoding(
        self,
        tree,
        past_key_values,
        position_offset,
        cache_position,
        skip_nodes,
        device,
    ):
        tree_input_ids, tree_position_ids, tree_mask = self._prepare_tree_inputs_and_mask(
            tree,
            position_offset=position_offset,
            device=device,
            model_dtype=self.draft_model.model.dtype,
            skip_nodes=skip_nodes,
            invert=True,
        )
        
        # Draft model forward
        with nvtx.annotate("draft_forward", color="red"):
            next_token_logits = self.draft_model(
                tree_input_ids.unsqueeze(0),
                past_key_values=past_key_values.cache,
                attention_mask=tree_mask,
                position_ids=tree_position_ids.unsqueeze(0),
                cache_position=cache_position,
            )
        return next_token_logits

    def _post_verify(
        self,
        tree,
        root_ind,
        past_key_values,
        position_offset,
        cache_position,
        last_tree_depth,
        skip_nodes,
        logits_processor,
        device,
    ):
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

        # Speculate to refill the tree.
        refill_steps = int(self.draft_params.max_depth) - accept_len
        if refill_steps > 0:
            with nvtx.annotate("postspec_refill", color="cyan"):
                self.draft_model.init_postspec()
                for _ in range(refill_steps):
                    if not self.draft_model.postspec():
                        break
            tree = self.draft_model.update_tree_after_post()

        self.post_verify_count += 1
        return tree, kept_old_indices

    def _tree_decoding(
        self,
        tree,
        past_key_values,
        position_offset,
        cache_position,
        skip_nodes,
        device,
    ):
        # Disable draft profiling during target forward
        if self.profiling:
            self.profile_draft_time = False

        tree_input_ids, tree_position_ids, tree_mask = self._prepare_tree_inputs_and_mask(
            tree,
            position_offset=position_offset,
            device=device,
            model_dtype=self.target_model.model.dtype,
            skip_nodes=skip_nodes,
            invert=True,
        )
        
        # Target model forward
        with nvtx.annotate("target_forward", color="red"):
            outputs = self.target_model(
                tree_input_ids.unsqueeze(0),
                past_key_values=past_key_values.cache,
                attention_mask=tree_mask,
                position_ids=tree_position_ids.unsqueeze(0),
                cache_position=cache_position,
            )

        if self.profiling:
            self.profile_draft_time = True
        return outputs

    def _reorder_pending_tree_cache(
        self,
        past_key_values,
        hidden_indices,
        pending_tree_size: int,
        *,
        input_len: int,
    ) -> None:
        if hidden_indices is None or int(hidden_indices.numel()) <= 0:
            raise RuntimeError("Deferred reorder expected non-empty hidden_indices_cache in subspec_sd_v2.")

        pending_tree_size = self._resolve_pending_chunk_size(
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
        self,
        past_key_values,
        hidden_indices_cache,
        tree_size: int,
        *,
        input_len: int,
    ) -> tuple[int, bool, torch.Tensor | None]:
        with nvtx.annotate("kv_reorder"):
            self._reorder_pending_tree_cache(
                past_key_values,
                hidden_indices_cache,
                int(tree_size),
                input_len=int(input_len),
            )
        return 0, False, None

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """Generate a token sequence with speculative decoding.

        Stages:
        - Prefill: run the target model on the prompt, then sample the next token.
        - Decode loop:
            1) Draft proposes candidate tokens (tree form).
            2) Target scores candidates in one forward.
            3) Verify and accept a prefix, then update KV/state.

        Args:
            input_ids (torch.LongTensor): The input token IDs.
            stopping_criteria (StoppingCriteria): The criteria to stop the generation.
            logits_processor (LogitsProcessor): The processor to modify the logits.
            do_sample (bool): Whether to sample tokens during generation. If False, the generation will be deterministic.

        Returns:
            input_ids (torch.LongTensor): The generated token IDs.
        """
        assert self.target_model is not None, "target_model must be provided"
        assert self.draft_model is not None, "draft_model must be provided"
        assert self.tokenizer is not None, "tokenizer must be provided"

        input_ids = input_ids.clone()
        batch_size, _ = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."

        # Raise error if max_length not set while using static cache
        if stopping_criteria.max_length is None:
            if self.cache_implementation == "static":
                raise ValueError(
                    "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
                )

        if model_kwargs.get("past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
            max_cache_len = getattr(past_key_values.cache, "max_cache_len", None)

            self.draft_model.set_past_key_values(past_key_values)
        else:
            raise ValueError("past_key_values is not provided")

        stream_callback = model_kwargs.get("stream_callback", None)
        self._init_step_trace()

        with nvtx.annotate("prefill_chunked", color="orange"):
            self._init_tree_mask(
                self.draft_params.max_verify_tokens * 2,
                max_cache_len,
                device=input_ids.device,
            )
            outputs = self._chunked_prefill_forward(
                input_ids,
                past_key_values,
                prefill_chunk_size=self.prefill_chunk_size,
                use_position_ids=True,
            )
            next_token_logits = outputs.logits
            del outputs

        remaining = self._remaining_token_budget(input_ids, stopping_criteria)
        if remaining is not None and int(remaining) <= 0:
            return input_ids

        with nvtx.annotate("sample"):
            sampled_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

        with nvtx.annotate("state_update"):
            input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
            position_offset = input_ids.shape[1] - 1
            self._maybe_stream(stream_callback, sampled_tokens)

        with nvtx.annotate("decode_loop"):
            # Better naming:
            # - `post_verify_count`: previous speculative tree was fully accepted, so we run `post_verify` instead of re-speculating.
            # - `speculate_count`: we had to run a fresh speculation step.
            self.post_verify_count = 0
            self.speculate_count = 0
            disable_post_verify = bool(self.generator_kwargs.get("disable_post_verify", False))

            finished = False
            is_prev_accepted = False
            hidden_indices_cache = None
            last_tree_size = 0
            last_tree_depth = 0

            while not finished:
                remaining = self._remaining_token_budget(input_ids, stopping_criteria)
                if remaining is not None and int(remaining) <= 0:
                    break

                post_verify_used = False
                if is_prev_accepted:
                    skip_nodes = last_tree_size
                    pending_post_tokens = int(tree.size()) - int(skip_nodes)
                    should_flush_deferred = bool(disable_post_verify) or int(pending_post_tokens) <= 0
                    if (not should_flush_deferred) and (max_cache_len is not None):
                        cache_headroom = max(
                            0,
                            int(max_cache_len) - (int(position_offset) + int(skip_nodes)),
                        )
                        should_flush_deferred = int(cache_headroom) < int(pending_post_tokens)
                    if should_flush_deferred:
                        root_ind, is_prev_accepted, hidden_indices_cache = self._flush_deferred_tree_cache(
                            past_key_values,
                            hidden_indices_cache,
                            int(tree.size()),
                            input_len=int(input_ids.shape[1]),
                        )
                        continue

                    cache_position = torch.arange(
                        position_offset + last_tree_size,
                        position_offset + tree.size(),
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                    post_verify_used = True
                    with nvtx.annotate("post_verify", color="cyan"):
                        tree, kept_old_indices = self._post_verify(
                            tree,
                            root_ind,
                            past_key_values,
                            position_offset,
                            cache_position,
                            last_tree_depth,
                            skip_nodes,
                            logits_processor,
                            input_ids.device,
                        )
                    hidden_indices_cache = self._remap_hidden_indices_after_tree_prune(
                        hidden_indices_cache,
                        kept_old_indices,
                        method_name="subspec_sd_v2",
                    )
                    if int(tree.size()) <= int(skip_nodes):
                        root_ind, is_prev_accepted, hidden_indices_cache = self._flush_deferred_tree_cache(
                            past_key_values,
                            hidden_indices_cache,
                            int(tree.size()),
                            input_len=int(input_ids.shape[1]),
                        )
                        continue

                    # NOTE: `_post_verify` can prune/refill the tree (post-spec), changing `tree.size()`.
                    # `cache_position` must be recomputed to match the *updated* tree slice we will decode.
                    cache_position = torch.arange(
                        position_offset + skip_nodes,
                        position_offset + tree.size(),
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                    last_tree_size = tree.size()
                    last_tree_depth = tree.get_depth()

                else:
                    self.speculate_count += 1
                    last_token_id = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
                    with nvtx.annotate("speculate", color="cyan"):
                        tree = self._speculate(last_token_id)
                    last_tree_size = tree.size()
                    last_tree_depth = tree.get_depth()

                    skip_nodes = 0
                    position_offset = input_ids.shape[1] - 1
                    cache_position = torch.arange(
                        position_offset,
                        position_offset + tree.size(),
                        dtype=torch.long,
                        device=input_ids.device,
                    )

                tree_size_before_cap = int(tree.size())
                decoded_tree_size = self._cap_tree_to_budget(
                    tree,
                    input_ids,
                    stopping_criteria,
                    skip_nodes=skip_nodes,
                )
                tree_size_after_cap = int(tree.size())
                if max_cache_len is not None:
                    cache_decode_budget = max(
                        0,
                        int(max_cache_len) - (int(position_offset) + int(skip_nodes)),
                    )
                    if int(decoded_tree_size) > int(cache_decode_budget):
                        max_tree_nodes = int(skip_nodes) + int(cache_decode_budget)
                        if hasattr(tree, "truncate_prefix"):
                            tree.truncate_prefix(max_tree_nodes)
                        else:
                            tree.prune_to_top_n(max_tree_nodes)
                        decoded_tree_size = int(cache_decode_budget)
                if decoded_tree_size <= 0:
                    break
                last_tree_size = tree.size()
                cache_position = torch.arange(
                    position_offset + skip_nodes,
                    position_offset + skip_nodes + decoded_tree_size,
                    dtype=torch.long,
                    device=input_ids.device,
                )

                with nvtx.annotate("target_decode", color="orange"):
                    self.draft_model.init_postspec()
                    outputs = self._tree_decoding(
                        tree,
                        past_key_values,
                        position_offset=position_offset,
                        cache_position=cache_position,
                        skip_nodes=skip_nodes,
                        device=input_ids.device,
                    )
                    next_token_logits = outputs.logits

                with nvtx.annotate("postspec_update", color="cyan"):
                    tree = self.draft_model.update_tree_after_post()

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
                        skip_nodes=skip_nodes,
                    )

                    last_accepted_ind = hidden_indices[-1]
                    bonus_token = sampled_tokens[:, -1].item()
                    sampled_tokens = sampled_tokens.to(input_ids.device)
                    hidden_indices = hidden_indices.to(input_ids.device)

                    if is_prev_accepted:
                        hidden_indices_cache = torch.cat([hidden_indices_cache, hidden_indices], dim=-1)
                    else:
                        hidden_indices_cache = hidden_indices

                root_ind = tree.find_child_index(last_accepted_ind, bonus_token)
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
                    if not is_prev_accepted or finished:
                        self._reorder_pending_tree_cache(
                            past_key_values,
                            hidden_indices_cache,
                            int(tree.size()),
                            input_len=int(input_ids.shape[1]),
                        )
                        if finished:
                            past_key_values.seq_len -= prune_tokens

            # Normalize to plain ints for logging/consumers.
            self.post_verify_count = int(self.post_verify_count)
            self.speculate_count = int(self.speculate_count)

        return input_ids


class SubSpecSDGenerator(SDProfilingMixin, SubSpecSDGeneratorBase):
    pass

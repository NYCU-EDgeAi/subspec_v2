import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import nvtx

from .classic_sd import ClassicSDGeneratorBase
from ..utils.mixin import SDProfilingMixin
from ..utils.flashinfer.cache_manager import (
    RequestKvCache,
    getKvCacheBatchPosition,
)
from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper
from ..utils.flashinfer.prefill import flashinfer_chunked_prefill


class SubSpecSDGeneratorBase(ClassicSDGeneratorBase):
    def __init__(self, generator_kwargs, *model_args, **kwargs):
        super().__init__(generator_kwargs, *model_args, **kwargs)

    def init_cuda_graph_runner(self, device, kvCachePool=None):
        """
        Initialize the draft model CUDA-graph runner (FlashInfer path only).
        """
        if hasattr(self.draft_model, "init_cuda_graph_runner") and callable(
            self.draft_model.init_cuda_graph_runner
        ):
            self.draft_model.init_cuda_graph_runner(device=device)

    def _tree_decoding(self, tree, request_kv_cache, position_offset, cache_position, device):
        tree_input_ids, tree_position_ids, tree_mask = self._prepare_tree_inputs_and_mask(
            tree,
            position_offset=position_offset,
            device=device,
            model_dtype=self.target_model.model.dtype,
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
            # batch_position.print_info()
            self.flashinferWrapper.prepareAttention(
                "tree",
                batch_position,
                kvCachePool.page_len,
                "NONE",  # POS_ENCODING_MODE.NONE,
                kvCachePool.cache_data[0].dtype,
                attention_mask=tree_mask,
            )
            # Check if the current instance has the attribute 'graph'
            if hasattr(self, "graph"):
                outputs = self.tree_decoding_step(
                    input_ids=tree_input_ids.unsqueeze(0),
                    position_ids=tree_position_ids.unsqueeze(0),
                    batch_position=batch_position,
                )
            else:
                outputs = self.target_model(
                    input_ids=tree_input_ids.unsqueeze(0),
                    past_key_values=None,
                    position_ids=tree_position_ids.unsqueeze(0),
                    output_hidden_states=True,
                    use_cache=False,
                    kvCachePool=kvCachePool,
                    batch_position=batch_position,
                    mode="tree",
                    flashinferWrapper=self.flashinferWrapper,
                )
        return outputs

    def _speculate(self, input_ids, request_kv_cache):
        return self.draft_model.speculate(
            input_ids,
            request_kv_cache=request_kv_cache,
            flashinferWrapper=self.flashinferWrapper,
        )

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """Generate a token sequence with speculative decoding (FlashInfer-backed).

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
        prompt_input_len = int(input_ids.shape[1])

        # Raise error if max_length not set while using static cache
        if stopping_criteria.max_length is None:
            if self.cache_implementation == "static":
                raise ValueError(
                    "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
                )

        if model_kwargs.get("past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
            max_cache_len = getattr(past_key_values, "max_cache_len", None)
        else:
            raise ValueError("past_key_values is not provided")

        stream_callback = model_kwargs.get("stream_callback", None)
        self._init_step_trace()

        with nvtx.annotate("prefill_chunked", color="orange"):
            self._init_tree_mask(
                self.draft_params.max_verify_tokens,
                max_cache_len,
                device=input_ids.device,
            )
            if not hasattr(self, "flashinferWrapper"):
                self.flashinferWrapper = FlashinferAttentionWrapper(
                    self.target_model.config.num_attention_heads,
                    self.target_model.config.num_key_value_heads,
                    self.target_model.config.hidden_size,
                    past_key_values.page_len,
                    # Tree row count can vary across decode rounds; keep planning dynamic.
                    tree_use_cuda_graph=False,
                )
            self.kvCachePool = past_key_values
            request_kv_cache = self._ensure_request_kv_cache(
                attr_name="_fi_request_kv_cache",
                request_cls=RequestKvCache,
                kv_cache_pool=self.kvCachePool,
                input_ids_len=int(input_ids.shape[1]),
                input_ids=input_ids,
                tokens_attr_name="_fi_request_tokens",
                reuse_len_attr_name="_fi_request_reuse_len",
            )
            outputs = flashinfer_chunked_prefill(
                target_model=self.target_model,
                flashinfer_wrapper=self.flashinferWrapper,
                input_ids=input_ids,
                kv_cache_pool=self.kvCachePool,
                request_kv_cache=request_kv_cache,
                prefill_chunk_size=self.prefill_chunk_size,
            )
            next_token_logits = outputs.logits
            del outputs

        remaining = self._remaining_token_budget(input_ids, stopping_criteria)
        if remaining is not None and int(remaining) <= 0:
            self._remember_request_cache_tokens(
                tokens_attr_name="_fi_request_tokens",
                input_ids=input_ids,
            )
            setattr(self, "_fi_request_reuse_len", int(prompt_input_len))
            return input_ids

        with nvtx.annotate("sample"):
            sampled_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

        with nvtx.annotate("state_update"):
            input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
            self._maybe_stream(stream_callback, sampled_tokens)

        with nvtx.annotate("decode_loop"):
            finished = False
            while not finished:
                remaining = self._remaining_token_budget(input_ids, stopping_criteria)
                if remaining is not None and int(remaining) <= 0:
                    break
                cache_headroom = self._request_cache_headroom(request_kv_cache)
                if cache_headroom is not None and int(cache_headroom) <= 0:
                    break

                with nvtx.annotate("speculate", color="cyan"):
                    last_token_id = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
                    prev_kv_len = request_kv_cache.get_seq_length() + 1
                    tree = self._speculate(last_token_id, request_kv_cache)
                    tree_size_before_cap = int(tree.size())
                    decoded_tree_size = self._cap_tree_to_budget(
                        tree,
                        input_ids,
                        stopping_criteria,
                    )
                    tree_size_after_cap = int(tree.size())
                    self._sync_request_cache_after_tree_truncation(
                        request_kv_cache,
                        tree_size_before=tree_size_before_cap,
                        tree_size_after=tree_size_after_cap,
                    )
                    if decoded_tree_size <= 0:
                        break

                with nvtx.annotate("target_decode", color="orange"):
                    outputs = self._tree_decoding(
                        tree,
                        request_kv_cache,
                        position_offset=input_ids.shape[1] - 1,
                        cache_position=None,
                        device=input_ids.device,
                    )
                    next_token_logits = outputs.logits if outputs is not None else None
                    del outputs

                with nvtx.annotate("verify"):
                    root_ind = 0
                    sampled_tokens, hidden_indices, (_, accept_len) = self._verify(
                        tree,
                        root_ind,
                        next_token_logits,
                        logits_processor,
                        do_sample,
                    )
                    sampled_tokens = sampled_tokens.to(input_ids.device)
                    del next_token_logits
                    self._append_step_trace(
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

                with nvtx.annotate("kv_reorder"):
                    num_new_tokens = int(decoded_tree_size)
                    request_kv_cache.reorder_cache_with_offset(
                        hidden_indices,
                        offset=prev_kv_len,
                        num_new_tokens=num_new_tokens,
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
                if finished and int(prune_tokens) > 0:
                    request_kv_cache.decrement(int(prune_tokens))
        self._remember_request_cache_tokens(
            tokens_attr_name="_fi_request_tokens",
            input_ids=input_ids,
        )
        setattr(self, "_fi_request_reuse_len", int(prompt_input_len))
        return input_ids


class SubSpecSDGenerator(SDProfilingMixin, SubSpecSDGeneratorBase):
    pass

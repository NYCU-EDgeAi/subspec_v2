import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx

from .base import GeneratorBase
from ..utils.mixin import SDProfilingMixin
from ..utils.utils import invert_mask
from ..utils.tree_verify import verify_tree


class ClassicSDGeneratorBase(GeneratorBase):
    def __init__(self, generator_kwargs, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.generator_kwargs = generator_kwargs or {}
        self.prefill_chunk_size = self.generator_kwargs.get("prefill_chunk_size", None)
        self.step_trace_enabled = bool(self.generator_kwargs.get("step_trace", False))
        self.step_trace_debug_verify = bool(
            self.generator_kwargs.get("step_trace_debug_verify", False)
        )
        self._step_trace = None
        self._step_trace_step = 0
        self._is_prev_accepted_count = 0
        self._is_prev_accepted_steps = 0

    def _init_step_trace(self) -> None:
        self._is_prev_accepted_count = 0
        self._is_prev_accepted_steps = 0
        if not self.step_trace_enabled:
            self._step_trace = None
            self._step_trace_step = 0
            return
        self._step_trace = []
        self._step_trace_step = 0

    def _append_step_trace(
        self,
        *,
        is_prev_accepted: bool,
        skip_nodes: int,
        tree_size_before_cap: int,
        tree_size_after_cap: int,
        decoded_tree_size: int,
        root_ind_in: int,
        root_ind_out: int,
        accept_len: int,
        hidden_indices_len: int,
        post_verify_used: bool,
        extra_fields: dict | None = None,
    ) -> None:
        self._is_prev_accepted_steps += 1
        if bool(is_prev_accepted):
            self._is_prev_accepted_count += 1

        if not self.step_trace_enabled or self._step_trace is None:
            return

        row = {
            "step": int(self._step_trace_step),
            "is_prev_accepted": bool(is_prev_accepted),
            "skip_nodes": int(skip_nodes),
            "tree_size_before_cap": int(tree_size_before_cap),
            "tree_size_after_cap": int(tree_size_after_cap),
            "decoded_tree_size": int(decoded_tree_size),
            "root_ind_in": int(root_ind_in),
            "root_ind_out": int(root_ind_out),
            "accept_len": int(accept_len),
            "hidden_indices_len": int(hidden_indices_len),
            "post_verify_used": bool(post_verify_used),
        }
        if extra_fields:
            for key, value in extra_fields.items():
                if isinstance(value, bool):
                    row[str(key)] = bool(value)
                elif isinstance(value, int):
                    row[str(key)] = int(value)
                elif isinstance(value, torch.Tensor) and int(value.numel()) == 1:
                    scalar = value.item()
                    if isinstance(scalar, bool):
                        row[str(key)] = bool(scalar)
                    else:
                        row[str(key)] = int(scalar)
                else:
                    row[str(key)] = value
        self._step_trace.append(row)
        self._step_trace_step += 1

    def _export_step_trace(self):
        if not self.step_trace_enabled:
            return None
        if self._step_trace is None:
            return []
        return list(self._step_trace)

    def _export_is_prev_accepted_stats(self) -> dict:
        total_steps = int(self._is_prev_accepted_steps)
        accepted_count = int(self._is_prev_accepted_count)
        return {
            "is_prev_accepted_count": accepted_count,
            "is_prev_accepted_steps": total_steps,
            "is_prev_accepted_rate": (float(accepted_count) / float(total_steps))
            if total_steps > 0
            else 0.0,
        }

    @staticmethod
    def _weighted_tensor_hash(values: torch.Tensor) -> int:
        if values is None:
            return 0
        flat = values.detach().to(dtype=torch.int64).view(-1).cpu()
        if int(flat.numel()) == 0:
            return 0
        weights = torch.arange(
            1,
            int(flat.numel()) + 1,
            dtype=torch.int64,
            device=flat.device,
        )
        return int((flat * weights).sum().item())

    def _build_verify_debug_trace(
        self,
        *,
        tree,
        next_token_logits: torch.Tensor | None,
        skip_nodes: int,
    ) -> dict:
        if not (self.step_trace_enabled and self.step_trace_debug_verify):
            return {}

        node_data = tree.get_tree_data(int(skip_nodes))
        tree_token_ids = node_data["token_ids"]
        debug = {
            "verify_tree_token_count": int(tree_token_ids.numel()),
            "verify_tree_token_hash": int(self._weighted_tensor_hash(tree_token_ids)),
        }

        if next_token_logits is None:
            debug["verify_argmax_len"] = 0
            debug["verify_argmax_hash"] = 0
            debug["verify_argmax_last"] = -1
            return debug

        logits = next_token_logits
        if int(logits.dim()) == 3:
            logits = logits[0]
        argmax_ids = torch.argmax(logits, dim=-1).to(torch.int64)
        debug["verify_argmax_len"] = int(argmax_ids.numel())
        debug["verify_argmax_hash"] = int(self._weighted_tensor_hash(argmax_ids))
        debug["verify_argmax_last"] = (
            int(argmax_ids[-1].item()) if int(argmax_ids.numel()) > 0 else -1
        )
        return debug

    def _tree_token_hash(
        self,
        *,
        tree,
        skip_nodes: int = 0,
    ) -> int:
        node_data = tree.get_tree_data(int(skip_nodes))
        return int(self._weighted_tensor_hash(node_data["token_ids"]))
        
    def _speculate(self, input_ids):
        return self.draft_model.speculate(input_ids)
        
    def _init_tree_mask(self, max_verify_tokens, max_cache_len=None, device="cpu"):
        if not hasattr(self, "tree_mask_update_method"):
            self.tree_mask_update_method = "static" if max_cache_len is not None else "dynamic"
            logging.debug(
                "'max_cache_len' is %s tree_mask.",
                "set, uses static" if max_cache_len else "not set, uses dynamic",
            )

        tree_mask = (
            torch.zeros((1, 1, max_verify_tokens, max_cache_len), device=device, dtype=torch.bool)
            if max_cache_len is not None else None
        )
        self.base_tree_mask = tree_mask
        return tree_mask

    def _get_tree_mask(self, tree_mask_partial):
        if self.tree_mask_update_method == "static":
            # Avoid prints in hot path; use logging if needed.
            _, _, K, D = tree_mask_partial.shape

            # If the preallocated buffer is missing or too small, fall back to the
            # partial mask (dynamic behavior). This prevents intermittent shape
            # mismatch errors when the cache reports an unexpected small length.
            if (
                self.base_tree_mask is None
                or self.base_tree_mask.shape[2] < K
                or self.base_tree_mask.shape[3] < D
            ):
                return tree_mask_partial

            # Slice to the same shape as the partial input
            tree_mask_view = self.base_tree_mask[:, :, :K, :].clone()
            tree_mask_view[:, :, :K, :D] = tree_mask_partial

            # Return view with the correct shape
            return tree_mask_view
        else:
            return tree_mask_partial

    def _prepare_tree_inputs_and_mask(
        self,
        tree,
        *,
        position_offset: int,
        device: torch.device,
        model_dtype: torch.dtype,
        skip_nodes: int = 0,
        non_blocking: bool = False,
        invert: bool = True,
    ):
        """Prepare (input_ids, position_ids, attention_mask) for a tree decode forward.

        This centralizes the repeated tree batching logic across Classic/SubSpec/Eagle generators.
        """
        with nvtx.annotate("attn_mask/build"):
            node_data = tree.get_tree_data(skip_nodes)
            tree_input_ids = node_data["token_ids"]
            tree_position_ids = node_data["depths"] + position_offset

            tree_mask_partial = tree.create_attention_mask(position_offset, skip_nodes)

        with nvtx.annotate("attn_mask/to_device"):
            tree_input_ids = tree_input_ids.to(device, non_blocking=non_blocking)
            tree_position_ids = tree_position_ids.to(device, non_blocking=non_blocking)
            tree_mask_partial = tree_mask_partial.to(device)

        with nvtx.annotate("attn_mask/prepare"):
            tree_mask = self._get_tree_mask(tree_mask_partial)
            if invert:
                tree_mask = invert_mask(tree_mask, dtype=model_dtype)

        return tree_input_ids, tree_position_ids, tree_mask

    def _tree_decoding(self, tree, past_key_values, position_offset, cache_position, device):
        tree_input_ids, tree_position_ids, tree_mask = self._prepare_tree_inputs_and_mask(
            tree,
            position_offset=position_offset,
            device=device,
            model_dtype=self.target_model.model.dtype,
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
        return outputs

    def _verify_step(self, p, token_ids, logits_processor, do_sample):
        sampled_token_id = p.argmax() if not do_sample else p.multinomial(1).squeeze(-1)
        if torch.any(sampled_token_id == token_ids):
            return sampled_token_id, None
        else:
            return None, sampled_token_id

    def _verify(self, tree, root_ind, logits, logits_processor, do_sample, skip_nodes=0):
        verify_method = str(self.generator_kwargs.get("verify_method", "exact") or "exact").strip().lower()
        verify_kwargs = dict(self.generator_kwargs.get("verify_kwargs") or {})

        return verify_tree(
            tree=tree,
            root_ind=int(root_ind),
            logits=logits,
            sample_token_fn=self._sample_token,
            verify_step_fn=self._verify_step,
            eos_token_id=getattr(self.draft_model, "eos_token_id", None),
            logits_processor=logits_processor,
            do_sample=do_sample,
            skip_nodes=int(skip_nodes),
            verify_method=verify_method,
            verify_kwargs=verify_kwargs,
        )

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

            if model_kwargs.get("draft_past_key_values") is not None:
                draft_past_key_values = model_kwargs["draft_past_key_values"]
                self.draft_model.set_past_key_values(draft_past_key_values)
        else:
            raise ValueError("past_key_values and draft_past_key_values should both be provided")

        stream_callback = model_kwargs.get("stream_callback", None)
        
        with nvtx.annotate("prefill_chunked", color="orange"):
            self._init_tree_mask(
                self.draft_params.max_verify_tokens, max_cache_len, device=input_ids.device
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
            self._maybe_stream(stream_callback, sampled_tokens)

        with nvtx.annotate("decode_loop"):
            finished = False
            while not finished:
                remaining = self._remaining_token_budget(input_ids, stopping_criteria)
                if remaining is not None and int(remaining) <= 0:
                    break

                with nvtx.annotate("speculate", color="cyan"):
                    input_ids = input_ids.clone(memory_format=torch.contiguous_format)
                    tree = self._speculate(input_ids)
                    decoded_tree_size = self._cap_tree_to_budget(
                        tree,
                        input_ids,
                        stopping_criteria,
                    )
                    if decoded_tree_size <= 0:
                        break
                    if self.cache_implementation == "dynamic":
                        _, input_len = input_ids.shape
                        draft_past_key_values.crop(input_len)

                with nvtx.annotate("target_decode", color="orange"):
                    prev_kv_len = past_key_values.get_seq_length()
                    position_offset = int(input_ids.shape[1]) - 1
                    cache_position = torch.arange(
                        position_offset,
                        position_offset + int(decoded_tree_size),
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                    outputs = self._tree_decoding(
                        tree,
                        past_key_values,
                        position_offset=position_offset,
                        cache_position=cache_position,
                        device=input_ids.device,
                    )
                    next_token_logits = outputs.logits
                    del outputs

                with nvtx.annotate("verify"):
                    root_ind = 0
                    sampled_tokens, hidden_indices, _ = self._verify(
                        tree,
                        root_ind,
                        next_token_logits,
                        logits_processor,
                        do_sample,
                    )
                    
                    sampled_tokens = sampled_tokens.to(input_ids.device)
                    del next_token_logits

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
                    past_key_values.reorder_cache_with_offset(
                        hidden_indices,
                        offset=prev_kv_len,
                        new_chunk_len=int(decoded_tree_size),
                        dim=2,
                    )
                    past_key_values.seq_len += hidden_indices.shape[0]
                    if finished:
                        past_key_values.seq_len -= prune_tokens

        return input_ids


class ClassicSDGenerator(SDProfilingMixin, ClassicSDGeneratorBase):
    pass

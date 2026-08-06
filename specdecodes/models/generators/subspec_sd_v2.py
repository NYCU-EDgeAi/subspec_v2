import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import nvtx

from .classic_sd import ClassicSDGeneratorBase
from .subspec_sd_v2_loop import run_subspec_v2_generate, SdpaV2Backend
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
        """Generate a token sequence with SubSpec v2 (post-verify) speculative decoding.

        The loop itself is shared with the FlashInfer variant; see
        `subspec_sd_v2_loop.run_subspec_v2_generate`. This backend drives the static/
        dynamic `Cache` path (threading `cache_position` and clamping to `max_cache_len`).
        """
        return run_subspec_v2_generate(
            self,
            SdpaV2Backend(self),
            input_ids,
            stopping_criteria,
            logits_processor,
            do_sample,
            **model_kwargs,
        )


class SubSpecSDGenerator(SDProfilingMixin, SubSpecSDGeneratorBase):
    pass

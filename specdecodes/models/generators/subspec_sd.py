import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria

from .classic_sd import ClassicSDGeneratorBase
from .subspec_sd_v1_loop import run_subspec_v1_generate, SdpaV1Backend
from ..utils.mixin import SDProfilingMixin


class SubSpecSDGeneratorBase(ClassicSDGeneratorBase):
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

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """Generate a token sequence with speculative decoding (SDPA backend).

        The loop itself is shared with the FlashInfer variant; see
        `subspec_sd_v1_loop.run_subspec_v1_generate`. This backend drives the static/
        dynamic `Cache` path.
        """
        return run_subspec_v1_generate(
            self,
            SdpaV1Backend(self),
            input_ids,
            stopping_criteria,
            logits_processor,
            do_sample,
            **model_kwargs,
        )


class SubSpecSDGenerator(SDProfilingMixin, SubSpecSDGeneratorBase):
    pass

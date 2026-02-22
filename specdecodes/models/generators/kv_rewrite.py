import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import nvtx

from .classic_seq_sd import ClassicSDGeneratorBase
from ..utils.mixin import SDProfilingMixin

class KVRewriteGeneratorBase(ClassicSDGeneratorBase):
    def _verify(self, draft_ids, root_ind, logits, logits_processor, do_sample, skip_nodes: int = 0):
        # KV rewrite accepts the entire drafted block.
        accept_len = draft_ids.shape[1] - 1
        total_len = accept_len + 1
        sampled_tokens = draft_ids[:, :total_len]
        return sampled_tokens, None, (total_len, accept_len)
    
    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """
        Generate a token sequence with sequential speculative decoding.

        Stages:
        - Prefill: run the target model once on the prompt, then sample the next token.
        - Decode loop:
            1) Draft proposes a block of candidate tokens.
            2) Target scores the candidate block in one forward.
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
            
            self.draft_model.set_past_key_values(past_key_values)
        else:
            raise ValueError("past_key_values is not provided")

        stream_callback = model_kwargs.get("stream_callback", None)

        with nvtx.annotate("prefill_chunked", color="orange"):
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

        # Early stop guard after prefill sampling.
        done = stopping_criteria(input_ids, None)
        done = bool(done.item()) if hasattr(done, "item") else bool(done)
        if done:
            return input_ids

        
        with nvtx.annotate("speculate", color="cyan"):
            last_token_id = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
            draft_ids = self._speculate(last_token_id)

        with nvtx.annotate("decode_loop"):
            finished = False
            while not finished:
                remaining = self._remaining_token_budget(input_ids, stopping_criteria)
                if remaining is not None and int(remaining) <= 0:
                    break
                draft_ids = self._cap_draft_ids_to_budget(
                    draft_ids,
                    input_ids,
                    stopping_criteria,
                )
                
                with nvtx.annotate("target_decode", color="orange"):
                    # Recompute decode positions from the current committed prefix and
                    # current draft block so postspec length changes stay valid.
                    position_offset = input_ids.shape[1] - 1
                    cache_position = torch.arange(
                        position_offset,
                        position_offset + draft_ids.shape[1],
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                    _ = self._tree_decoding(draft_ids, cache_position, past_key_values)
                    
                with nvtx.annotate("verify"):
                    _ = self._verify(draft_ids, None, None, None, None)
                    
                sampled_tokens = draft_ids[:, 1:]
                self.draft_model.init_postspec()
                self.draft_model.sim_run_postspec()
                draft_ids = self.draft_model.get_draft_ids_after_post()
                    
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
                                
                with nvtx.annotate("kv_update"):
                    # Keep committed cache length exactly aligned with generated prefix.
                    past_key_values.seq_len = input_ids.shape[1]
                
        return input_ids
    
class KVRewriteGenerator(SDProfilingMixin, KVRewriteGeneratorBase):
    pass

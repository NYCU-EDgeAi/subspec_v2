import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import nvtx

from .classic_sd import ClassicSDGeneratorBase
from .flashinfer_cache_mixin import FlashInferCacheMixin
from .subspec_sd_v1_loop import run_subspec_v1_generate, FlashInferV1Backend
from ..utils.mixin import SDProfilingMixin
from ..utils.flashinfer.cache_manager import getKvCacheBatchPosition


class SubSpecSDGeneratorBase(FlashInferCacheMixin, ClassicSDGeneratorBase):
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
        """Generate a token sequence with speculative decoding (FlashInfer backend).

        The loop itself is shared with the SDPA variant; see
        `subspec_sd_v1_loop.run_subspec_v1_generate`. This backend drives the paged
        `RequestKvCache` + attention-wrapper path.
        """
        return run_subspec_v1_generate(
            self,
            FlashInferV1Backend(self),
            input_ids,
            stopping_criteria,
            logits_processor,
            do_sample,
            **model_kwargs,
        )


class SubSpecSDGenerator(SDProfilingMixin, SubSpecSDGeneratorBase):
    pass

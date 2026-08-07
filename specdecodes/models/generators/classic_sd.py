import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx

from .base import GeneratorBase
from .flashinfer_cache_mixin import FlashInferCacheMixin
from .classic_sd_loop import (
    run_classic_generate,
    SdpaClassicBackend,
    FlashInferClassicBackend,
)
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

    #: backend name -> ClassicBackend adapter class.
    _CLASSIC_BACKENDS = {
        "sdpa": SdpaClassicBackend,
        "flashinfer": FlashInferClassicBackend,
    }

    def init_cuda_graph_runner(self, device, kvCachePool=None):
        """Initialize the draft model's CUDA-graph runner (FlashInfer path only).

        A no-op on SDPA: the SDPA draft model does not expose this hook.
        """
        if hasattr(self.draft_model, "init_cuda_graph_runner") and callable(
            self.draft_model.init_cuda_graph_runner
        ):
            self.draft_model.init_cuda_graph_runner(device=device)

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """Generate a token sequence with classic speculative decoding.

        The decode loop is shared across attention backends; see
        `classic_sd_loop.run_classic_generate`. The `backend:` config field
        (`self.backend`) selects the `ClassicBackend` adapter.
        """
        backend_cls = self._CLASSIC_BACKENDS.get(str(self.backend))
        if backend_cls is None:
            raise ValueError(
                f"Unknown backend {self.backend!r} for classic_sd; "
                f"expected one of {sorted(self._CLASSIC_BACKENDS)}."
            )
        return run_classic_generate(
            self,
            backend_cls(self),
            input_ids,
            stopping_criteria,
            logits_processor,
            do_sample,
            **model_kwargs,
        )


class ClassicSDGenerator(SDProfilingMixin, FlashInferCacheMixin, ClassicSDGeneratorBase):
    pass

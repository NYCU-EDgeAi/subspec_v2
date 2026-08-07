import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria

from .classic_sd import ClassicSDGeneratorBase
from .flashinfer_cache_mixin import FlashInferCacheMixin
from .subspec_sd_v1_loop import (
    run_subspec_v1_generate,
    SdpaV1Backend,
    FlashInferV1Backend,
)
from ..utils.mixin import SDProfilingMixin


class SubSpecSDGeneratorBase(FlashInferCacheMixin, ClassicSDGeneratorBase):
    """SubSpec v1 generator — one class, backend chosen by config.

    The decode loop lives in `subspec_sd_v1_loop.run_subspec_v1_generate`; the `backend:`
    config field (`self.backend`) selects which `SpecDecodeBackend` adapter owns the
    KV-cache lifecycle, prefill, and attention execution. `FlashInferCacheMixin` provides
    the paged request-cache helpers the FlashInfer adapter calls back into; they are inert
    on the SDPA path.
    """

    #: backend name -> SpecDecodeBackend adapter class.
    _V1_BACKENDS = {
        "sdpa": SdpaV1Backend,
        "flashinfer": FlashInferV1Backend,
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
        """Generate a token sequence with SubSpec v1 speculative decoding.

        Delegates to the shared loop with the backend adapter selected by `self.backend`.
        """
        backend_cls = self._V1_BACKENDS.get(str(self.backend))
        if backend_cls is None:
            raise ValueError(
                f"Unknown backend {self.backend!r} for subspec_sd; "
                f"expected one of {sorted(self._V1_BACKENDS)}."
            )
        return run_subspec_v1_generate(
            self,
            backend_cls(self),
            input_ids,
            stopping_criteria,
            logits_processor,
            do_sample,
            **model_kwargs,
        )


class SubSpecSDGenerator(SDProfilingMixin, SubSpecSDGeneratorBase):
    pass

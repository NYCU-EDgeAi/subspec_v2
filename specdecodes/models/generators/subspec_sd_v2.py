import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria

from .classic_sd import ClassicSDGeneratorBase
from .flashinfer_cache_mixin import FlashInferCacheMixin
from .subspec_sd_v2_loop import (
    run_subspec_v2_generate,
    SdpaV2Backend,
    FlashInferV2Backend,
)
from ..utils.mixin import SDProfilingMixin


class SubSpecSDGeneratorBase(FlashInferCacheMixin, ClassicSDGeneratorBase):
    """SubSpec v2 (post-verify) generator — one class, backend chosen by config.

    The decode loop lives in `subspec_sd_v2_loop.run_subspec_v2_generate`; the
    `backend:` config field (`self.backend`) selects which `SubSpecV2Backend` adapter
    owns the KV-cache lifecycle, prefill, and attention execution. `FlashInferCacheMixin`
    provides the paged request-cache helpers the FlashInfer adapter calls back into;
    they are inert on the SDPA path.
    """

    #: backend name -> SubSpecV2Backend adapter class.
    _V2_BACKENDS = {
        "sdpa": SdpaV2Backend,
        "flashinfer": FlashInferV2Backend,
    }

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """Generate a token sequence with SubSpec v2 (post-verify) speculative decoding.

        Delegates to the shared loop with the backend adapter selected by `self.backend`.
        """
        backend_cls = self._V2_BACKENDS.get(str(self.backend))
        if backend_cls is None:
            raise ValueError(
                f"Unknown backend {self.backend!r} for subspec_sd_v2; "
                f"expected one of {sorted(self._V2_BACKENDS)}."
            )
        return run_subspec_v2_generate(
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

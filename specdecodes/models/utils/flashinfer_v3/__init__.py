"""FlashInfer v3 utilities for SubSpec SD.

Clean implementation with best design patterns:
- CUDAGraphRunner: Separate class for CUDA graph lifecycle management
- Clear state management and defensive programming
- FlashInfer methods require fullgraph=False in torch.compile
"""

from .cache_manager import (
    KvCachePool,
    KvCacheBatchPosition,
    RequestKvCache,
    getKvCacheBatchPosition,
    FlashInferCache,
)
from .attention_wrapper import (
    FlashinferAttentionWrapper,
    POS_ENCODING_MODE,
    AttentionRotaryParams,
)
from .prefill import flashinfer_chunked_prefill
from .monkey_patch import apply_flashinfer_kernel_to_llama
from .cuda_graph_runner import CUDAGraphRunner, CUDAGraphConfig

__all__ = [
    "KvCachePool", "KvCacheBatchPosition", "RequestKvCache",
    "getKvCacheBatchPosition", "FlashInferCache",
    "FlashinferAttentionWrapper", "POS_ENCODING_MODE", "AttentionRotaryParams",
    "flashinfer_chunked_prefill", "apply_flashinfer_kernel_to_llama",
    "CUDAGraphRunner", "CUDAGraphConfig",
]

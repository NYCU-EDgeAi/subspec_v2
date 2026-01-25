"""CUDA Graph Runner for FlashInfer v3 (tree mode only).

Design decision: Only tree mode (D>1) uses CUDA graph.
Root step (D=1) runs without CUDA graph - the overhead for a single token
is minimal compared to the complexity of maintaining dual CUDA graphs.
"""

from dataclasses import dataclass
from typing import Callable
import torch
from .cache_manager import KvCacheBatchPosition


@dataclass
class CUDAGraphConfig:
    """Configuration for CUDA graph."""
    enabled: bool = True
    warmup_iterations: int = 2
    batch_size: int = 1


class CUDAGraphRunner:
    """Manages single CUDA graph for tree mode (D>1)."""

    def __init__(self, config: CUDAGraphConfig, topk_len: int, max_pages: int):
        self.config = config
        self.topk_len = topk_len
        self.max_pages = max_pages

        # Tree graph
        self._graph: torch.cuda.CUDAGraph = None
        self._captured = False

        # Tree buffers (seq_len=topk_len)
        self._input_ids = None
        self._position_ids = None
        self._output = None

        # KV position buffers
        self._seq_indptr = None
        self._kv_page_indptr = None
        self._kv_page_indices = None
        self._kv_last_page_len = None
        self._batch_indices = None
        self._positions = None

        # Dependencies
        self._forward_fn: Callable = None
        self._pool = None
        self._wrapper = None

    @property
    def is_captured(self) -> bool:
        return self._captured

    def set_dependencies(self, forward_fn: Callable, kv_cache_pool, flashinfer_wrapper):
        self._forward_fn = forward_fn
        self._pool = kv_cache_pool
        self._wrapper = flashinfer_wrapper

    def _alloc_buffers(self, device: torch.device):
        B, L = self.config.batch_size, self.topk_len

        # Tree buffers (seq_len=topk_len)
        self._input_ids = torch.zeros((B, L), dtype=torch.long, device=device)
        self._position_ids = torch.zeros((B, L), dtype=torch.long, device=device)

        # KV position buffers
        self._seq_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        self._kv_page_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        self._kv_page_indices = torch.zeros((self.max_pages,), dtype=torch.int32, device=device)
        self._kv_last_page_len = torch.zeros((B,), dtype=torch.int32, device=device)
        self._batch_indices = torch.zeros((L,), dtype=torch.int32, device=device)
        self._positions = torch.zeros((L,), dtype=torch.int32, device=device)

    def _get_batch_position(self) -> KvCacheBatchPosition:
        return KvCacheBatchPosition(
            seq_indptr=self._seq_indptr,
            kv_page_indptr=self._kv_page_indptr,
            kv_page_indices=self._kv_page_indices,
            kv_last_page_len=self._kv_last_page_len,
            batch_indices=self._batch_indices,
            positions=self._positions,
        )

    def capture(self, device: torch.device):
        """Capture tree CUDA graph."""
        if not self.config.enabled or self._captured:
            return

        print("[CUDAGraphRunner] Capturing tree CUDA graph...")
        self._alloc_buffers(device)
        batch_pos = self._get_batch_position()

        # NOTE: Don't call prepareAttention here!
        # The FlashInfer wrapper's state is preserved from the last speculation loop.
        # Calling prepareAttention with different data could corrupt the internal state.
        # This matches v1's behavior which does not call prepareAttention during capture.

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            # Warmup
            for _ in range(self.config.warmup_iterations):
                self._forward_fn(
                    self._input_ids, position_ids=self._position_ids,
                    kvCachePool=self._pool, batch_position=batch_pos,
                    mode='tree', flashinferWrapper=self._wrapper
                )

            torch.cuda.current_stream().wait_stream(stream)

            # Capture
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=stream):
                self._output = self._forward_fn(
                    self._input_ids, position_ids=self._position_ids,
                    kvCachePool=self._pool, batch_position=batch_pos,
                    mode='tree', flashinferWrapper=self._wrapper
                )

        self._captured = True
        print("[CUDAGraphRunner] Tree CUDA graph captured successfully")

    def _copy_kv_position(self, batch_position: KvCacheBatchPosition):
        """Copy KV position data to buffers."""
        L = self.topk_len
        self._seq_indptr.copy_(batch_position.seq_indptr)
        self._kv_page_indptr.copy_(batch_position.kv_page_indptr)
        self._kv_last_page_len.copy_(batch_position.kv_last_page_len)
        self._batch_indices[:L].copy_(batch_position.batch_indices[:L])
        self._positions[:L].copy_(batch_position.positions[:L])

        n_pages = batch_position.kv_page_indptr[1].item()
        self._kv_page_indices[:n_pages].copy_(batch_position.kv_page_indices[:n_pages])

    def replay(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
               batch_position: KvCacheBatchPosition) -> torch.Tensor:
        """Replay tree graph.

        Note: prepareAttention("tree", ..., attention_mask=...) must be called
        externally before this method.
        """
        L = input_ids.shape[1]
        self._input_ids[:, :L].copy_(input_ids)
        self._position_ids[:, :L].copy_(position_ids)
        self._copy_kv_position(batch_position)
        self._graph.replay()
        return self._output

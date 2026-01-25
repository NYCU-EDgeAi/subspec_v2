"""KV Cache management for FlashInfer v3.

Direct copy of v1 implementation (known working).
"""

from typing import List
import logging
import math
import os
import torch
import nvtx
import flashinfer


class KvCacheBatchPosition:
    """Batch position information for FlashInfer attention."""

    def __init__(
        self,
        seq_indptr: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
    ):
        # for append kv cache
        self.batch_indices = batch_indices
        self.kv_page_indices = kv_page_indices
        self.positions = positions
        self.kv_page_indptr = kv_page_indptr
        self.kv_last_page_len = kv_last_page_len
        self.seq_indptr = seq_indptr  # for begin forward

    def print_info(self):
        print(f"  q_indptr:         {self.seq_indptr}")
        print(f"  kv_page_indptr:   {self.kv_page_indptr}")
        print(f"  kv_page_indices:  {self.kv_page_indices}")
        print(f"  kv_last_page_len: {self.kv_last_page_len}")
        print(f"  batch_indices:    {self.batch_indices}")
        print(f"  positions:        {self.positions}")


class KvCachePool:
    """Paged KV cache pool for FlashInfer."""

    def __init__(
        self,
        max_pages: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_len: int,
        dtype: torch.dtype,
        device: torch.device,
        max_cache_len: int = None,
    ):
        self.cache_data = torch.zeros(
            num_layers, max_pages, 2, page_len, num_heads, head_dim,
            dtype=dtype, device=device
        )

        self.num_layers = num_layers
        self.device = device
        self.max_pages = max_pages
        self.page_len = page_len
        self.max_cache_len = max_cache_len if max_cache_len is not None else max_pages * page_len
        self.free_page_mask = torch.ones(max_pages, dtype=torch.bool, device="cpu")
        self.num_heads = num_heads
        self.head_dims = head_dim
        self.dtype = dtype

    def reset(self):
        self.cache_data.zero_()

    def num_free_pages(self):
        return self.free_page_mask.sum()

    def allocate(self, num_pages: int):
        free_page_indices = self.free_page_mask.nonzero()
        assert (
            len(free_page_indices) >= num_pages
        ), f"Out of available cache pages: asked {num_pages}, only {len(free_page_indices)} free pages"

        allocated_indices = free_page_indices[:num_pages]
        self.free_page_mask[allocated_indices] = False
        return allocated_indices.squeeze(1).tolist()

    def deallocate(self, kv_page_indices: List[int]):
        self.free_page_mask[kv_page_indices] = True

    def crop(self, seq_len: int) -> None:
        """Zero-out all KV cache entries after `seq_len` tokens."""
        if not (0 <= seq_len <= self.max_pages * self.page_len):
            raise ValueError(
                f"seq_len={seq_len} is outside the [0, {self.max_pages * self.page_len}] range."
            )

        full_pages = seq_len // self.page_len
        remainder = seq_len % self.page_len
        keep_pages = full_pages + (1 if remainder else 0)

        if remainder:
            self.cache_data[:, full_pages, :, remainder:, ...].zero_()

        if keep_pages < self.max_pages:
            self.cache_data[:, keep_pages:, ...].zero_()

        self.free_page_mask.zero_()
        if keep_pages < self.max_pages:
            self.free_page_mask[keep_pages:] = True

    def reorder_cache_with_offset(self, beam_idx: torch.LongTensor, offset=0, num_new_tokens=0):
        """Reorders the cache for speculative decoding."""
        with nvtx.annotate("to device", color="green"):
            beam_idx = beam_idx.to(self.device)
            beam_size = beam_idx.size(0)

        old_indices = beam_idx + offset
        new_indices = torch.arange(offset, offset + beam_size, device=self.device, dtype=torch.long)

        page_len = self.page_len

        def to_flat_idx(idx: torch.Tensor):
            page_indices = idx // page_len
            token_indices = idx % page_len
            return page_indices, token_indices

        with nvtx.annotate("compute idx", color="blue"):
            old_page_indices, old_token_indices = to_flat_idx(old_indices)
            new_page_indices, new_token_indices = to_flat_idx(new_indices)

            old_flat = old_page_indices * page_len + old_token_indices
            new_flat = new_page_indices * page_len + new_token_indices

            total_tokens = offset + num_new_tokens
            total_pages = (total_tokens + page_len - 1) // page_len
            max_flat_len = total_pages * page_len

        with nvtx.annotate("stack cache", color="red"):
            cache_stacked = self.cache_data
            L, max_pages, _, page_len_, num_heads, head_dim = cache_stacked.shape
            if page_len_ != page_len:
                raise ValueError(f"Expected page_len={page_len}, found {page_len_}")
            if total_pages > max_pages:
                raise ValueError(
                    f"Cache does not have enough pages ({max_pages}) for total tokens ({total_tokens})."
                )

        with nvtx.annotate("split k/v", color="green"):
            k_cat = cache_stacked[:, :, 0, :, :, :].clone()
            v_cat = cache_stacked[:, :, 1, :, :, :].clone()

        with nvtx.annotate("flatten", color="blue"):
            k_cat = k_cat.view(L, max_pages * page_len, num_heads, head_dim)
            v_cat = v_cat.view(L, max_pages * page_len, num_heads, head_dim)

        with nvtx.annotate("reorder", color="yellow"):
            k_cat.index_copy_(1, new_flat, k_cat.index_select(1, old_flat))
            v_cat.index_copy_(1, new_flat, v_cat.index_select(1, old_flat))

        with nvtx.annotate("unflatten", color="green"):
            k_cat = k_cat.view(L, max_pages, page_len, num_heads, head_dim)
            v_cat = v_cat.view(L, max_pages, page_len, num_heads, head_dim)

        with nvtx.annotate("assign", color="purple"):
            self.cache_data[:, :, 0, :, :, :].copy_(k_cat, non_blocking=True)
            self.cache_data[:, :, 1, :, :, :].copy_(v_cat, non_blocking=True)


class RequestKvCache:
    """Per-request KV cache state tracking."""

    def __init__(self, kvCachePool: KvCachePool, page_len: int, seq_init_len: int):
        self.kvCachePool = kvCachePool
        self.page_len = page_len
        # IMPORTANT: Match v1's exact initialization logic
        # When seq_init_len=0: init_num_pages=0, kv_last_page_len=page_len
        # This ensures increment() allocates a page on first call
        init_num_pages = math.ceil(seq_init_len / self.page_len)
        self.kv_last_page_len = seq_init_len - (init_num_pages - 1) * self.page_len
        self.kv_page_indices = kvCachePool.allocate(init_num_pages)
        self.kv_len = seq_init_len
        self.is_released = False

    def get_seq_length(self):
        return self.kv_len

    def increment(self, num_tokens: int = 1):
        self.kv_len += num_tokens
        self.kv_last_page_len += num_tokens
        while self.kv_last_page_len > self.page_len:
            self.kv_last_page_len -= self.page_len
            new_indices = self.kvCachePool.allocate(1)
            self.kv_page_indices.extend(new_indices)

    def release(self):
        self.kvCachePool.deallocate(self.kv_page_indices)
        self.is_released = True

    def decrement(self, num_tokens: int = 1):
        """Remove tokens from the end of this request's cache usage."""
        if num_tokens <= 0:
            return

        if num_tokens > self.kv_len:
            num_tokens = self.kv_len

        self.kv_len -= num_tokens
        needed_pages = (self.kv_len + self.page_len - 1) // self.page_len if self.kv_len > 0 else 0

        while len(self.kv_page_indices) > needed_pages:
            last_page = self.kv_page_indices.pop()
            self.kvCachePool.deallocate([last_page])

        if self.kv_len == 0:
            self.kv_last_page_len = 0
        else:
            self.kv_last_page_len = (self.kv_len - 1) % self.page_len + 1

    def crop(self, start: int, end=None, dim=0):
        """Crop the past key/values up to a new length."""
        if end is None:
            end = self.get_seq_length()

        if start < 0:
            start = end - abs(start)
        if end <= start:
            return

        self.kv_len = start

        if self.kv_len == 0:
            self.kv_last_page_len = 0
        else:
            self.kv_last_page_len = (self.kv_len - 1) % self.page_len + 1

        num_pages_needed = (self.kv_len + self.page_len - 1) // self.page_len
        current_num_pages = len(self.kv_page_indices)

        if current_num_pages > num_pages_needed:
            extra_pages = self.kv_page_indices[num_pages_needed:]
            self.kvCachePool.deallocate(extra_pages)
            self.kv_page_indices = self.kv_page_indices[:num_pages_needed]
        elif current_num_pages < num_pages_needed:
            additional_pages_needed = num_pages_needed - current_num_pages
            new_indices = self.kvCachePool.allocate(additional_pages_needed)
            self.kv_page_indices.extend(new_indices)
            raise ValueError("need to allocate new pages in crop, should not happen")

    def reorder_cache_with_offset(self, beam_idx: torch.LongTensor, offset=0, num_new_tokens=0):
        """Reorders the cache for beam search."""
        # IMPORTANT: v1's offset adjustment - this is critical!
        if offset != 0:
            offset -= 1

        self.kvCachePool.reorder_cache_with_offset(beam_idx, offset, num_new_tokens)

        self.kv_len = offset + beam_idx.size(0)

        if self.kv_len == 0:
            self.kv_last_page_len = 0
        else:
            self.kv_last_page_len = (self.kv_len - 1) % self.page_len + 1

        num_pages_needed = (self.kv_len + self.page_len - 1) // self.page_len
        current_num_pages = len(self.kv_page_indices)

        if current_num_pages > num_pages_needed:
            extra_pages = self.kv_page_indices[num_pages_needed:]
            self.kvCachePool.deallocate(extra_pages)
            self.kv_page_indices = self.kv_page_indices[:num_pages_needed]
        elif current_num_pages < num_pages_needed:
            additional_pages_needed = num_pages_needed - current_num_pages
            new_indices = self.kvCachePool.allocate(additional_pages_needed)
            self.kv_page_indices.extend(new_indices)
            raise ValueError("need to allocate new pages in reorder cache, should not happen")


def getKvCacheBatchPosition(
    request_kv_caches: List[RequestKvCache],
    mode: str,
    device: torch.device,
    treeTokens: int = 0,
) -> KvCacheBatchPosition:
    """Compute batch position for FlashInfer attention."""
    kv_page_indices_list = []
    kv_page_indptr_list = []
    seq_indptr_list = []
    kv_last_page_len_list = []
    seq_lens_list = []
    cum_pages = 0
    cum_seq_len = 0

    for request_kv_cache in request_kv_caches:
        kv_page_indices_list.extend(request_kv_cache.kv_page_indices)
        kv_page_indptr_list.append(cum_pages)
        seq_indptr_list.append(cum_seq_len)
        kv_last_page_len_list.append(request_kv_cache.kv_last_page_len)
        seq_lens_list.append(request_kv_cache.kv_len)
        cum_pages += len(request_kv_cache.kv_page_indices)

        if mode == 'prefill':
            cum_seq_len += request_kv_cache.kv_len
        elif mode == 'decode':
            cum_seq_len += 1
        elif mode == 'tree':
            cum_seq_len += treeTokens
        else:
            raise ValueError('invalid mode')

    kv_page_indptr_list.append(cum_pages)
    seq_indptr_list.append(cum_seq_len)

    kv_page_indices = torch.tensor(kv_page_indices_list, dtype=torch.int32, device=device)
    kv_page_indptr = torch.tensor(kv_page_indptr_list, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(kv_last_page_len_list, dtype=torch.int32, device=device)
    seq_indptr = torch.tensor(seq_indptr_list, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    kv_append_length = torch.tensor([cum_seq_len], dtype=torch.int32, device=device)
    kv_append_indptr = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.cumsum(kv_append_length, dim=0)
    ])

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr,
        seq_lens,
        cum_seq_len
    )

    return KvCacheBatchPosition(
        seq_indptr=seq_indptr,
        kv_page_indptr=kv_page_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_len=kv_last_page_len,
        batch_indices=batch_indices,
        positions=positions,
    )


class FlashInferCache:
    """FlashInfer cache wrapper for compatibility with transformers."""

    def __init__(self, config, max_tokens: int = None, PAGE_LEN: int = 16) -> None:
        currentDevice = torch.device(f'cuda:{torch.cuda.current_device()}')
        dtype_size = torch.tensor([], dtype=torch.float16).element_size()
        self.config = config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        MEMORY_FRACTION = float(os.getenv("CUDA_MEMORY_FRACTION", "1.0"))
        self.max_cache_len = max_tokens
        self.page_len = PAGE_LEN

        cache_page_size = (
            2 * PAGE_LEN
            * config.num_hidden_layers
            * config.num_key_value_heads
            * head_dim
            * dtype_size
        )

        total_free_memory, _ = torch.cuda.mem_get_info(currentDevice)
        total_gpu_memory = torch.cuda.get_device_properties(currentDevice).total_memory
        free_memory = max(0, total_free_memory - (1 - MEMORY_FRACTION) * total_gpu_memory)

        if free_memory < cache_page_size:
            raise RuntimeError(
                f"Not enough GPU memory to allocate even a single cache page "
                f"({cache_page_size / (1024**2):.2f} MiB required, "
                f"{free_memory / (1024**2):.2f} MiB available)."
            )

        # Use page_len=16 for efficient paging (v1 approach)
        max_pages = (max_tokens + PAGE_LEN - 1) // PAGE_LEN if max_tokens else 256

        self.kvCachePool = KvCachePool(
            max_pages=max_pages,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_key_value_heads,
            head_dim=head_dim,
            page_len=PAGE_LEN,
            dtype=torch.float16,
            device=currentDevice,
            max_cache_len=max_tokens,
        )

    def reset(self):
        self.kvCachePool.reset()

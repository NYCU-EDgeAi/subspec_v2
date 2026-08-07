"""FlashInfer paged request-cache lifecycle, extracted from GeneratorBase.

These helpers manage the per-session `RequestKvCache` over a paged `kvCachePool` and
are used ONLY on the FlashInfer backend (the `FlashInfer*Backend` adapters call back
into them, and `classic_sd_fi`). They live in a mixin so the shared `GeneratorBase`
stays free of FlashInfer concepts and so the unified SubSpec generators can inherit
them inertly on the SDPA path. Behaviour is identical to the previous GeneratorBase
methods; this is a pure relocation.
"""
from __future__ import annotations

import torch


class FlashInferCacheMixin:
    """Request-cache lifecycle for FlashInfer-backed generators.

    Mixed in ahead of the shared generator base so `self` still resolves the shared
    methods; the mixin only adds the FlashInfer-specific request-cache helpers.
    """

    def _request_cache_capacity(self, request_kv_cache):
        """Return total token capacity for a RequestKvCache-backed pool, if known."""
        if request_kv_cache is None:
            return None

        pool = getattr(request_kv_cache, "kvCachePool", None)
        if pool is None:
            return None

        max_cache_len = getattr(pool, "max_cache_len", None)
        if max_cache_len is not None:
            return int(max_cache_len)

        max_pages = getattr(pool, "max_pages", None)
        page_len = getattr(pool, "page_len", None)
        if max_pages is None or page_len is None:
            return None
        return int(max_pages) * int(page_len)

    def _request_cache_headroom(self, request_kv_cache):
        """Return remaining appendable tokens before hitting cache capacity."""
        capacity = self._request_cache_capacity(request_kv_cache)
        if capacity is None:
            return None
        return max(0, int(capacity) - int(request_kv_cache.get_seq_length()))

    def _is_cache_pool_reset(self, kv_cache_pool) -> bool:
        """Best-effort detection that a paged KV pool was externally reset."""
        if kv_cache_pool is None:
            return False

        num_free_pages_fn = getattr(kv_cache_pool, "num_free_pages", None)
        max_pages = getattr(kv_cache_pool, "max_pages", None)
        if not callable(num_free_pages_fn) or max_pages is None:
            return False

        try:
            return int(num_free_pages_fn()) == int(max_pages)
        except Exception:
            return False

    def _ensure_request_kv_cache(
        self,
        *,
        attr_name: str,
        request_cls,
        kv_cache_pool,
        input_ids_len: int,
        input_ids: torch.LongTensor | None = None,
        tokens_attr_name: str | None = None,
        reuse_len_attr_name: str | None = None,
    ):
        """Get/create a reusable per-session RequestKvCache for FlashInfer generators."""
        request_kv_cache = getattr(self, attr_name, None)
        recreate = request_kv_cache is None

        if not recreate:
            pool_reset_counter = getattr(kv_cache_pool, "reset_counter", None)
            cache_reset_counter = getattr(request_kv_cache, "_pool_reset_counter", None)
            if getattr(request_kv_cache, "is_released", False):
                recreate = True
            elif getattr(request_kv_cache, "kvCachePool", None) is not kv_cache_pool:
                recreate = True
            else:
                if pool_reset_counter is not None and cache_reset_counter is not None:
                    if int(pool_reset_counter) != int(cache_reset_counter):
                        recreate = True

                cached_len = int(request_kv_cache.get_seq_length())
                if reuse_len_attr_name is not None:
                    reusable_len = getattr(self, reuse_len_attr_name, None)
                    if reusable_len is not None:
                        reusable_len = max(0, min(int(reusable_len), int(cached_len)))
                        if int(reusable_len) < int(cached_len):
                            try:
                                request_kv_cache.decrement(int(cached_len) - int(reusable_len))
                                cached_len = int(reusable_len)
                            except Exception:
                                recreate = True

                # If prompt did not strictly grow, treat this as a new session boundary.
                if not recreate and cached_len >= int(input_ids_len):
                    recreate = True
                # If external reset cleared the pool, cached request metadata is stale.
                elif (
                    not recreate
                    and cached_len > 0
                    and self._is_cache_pool_reset(kv_cache_pool)
                ):
                    recreate = True

                if (
                    not recreate
                    and input_ids is not None
                    and tokens_attr_name is not None
                    and cached_len > 0
                ):
                    cached_tokens = getattr(self, tokens_attr_name, None)
                    if (
                        not isinstance(cached_tokens, torch.Tensor)
                        or int(cached_tokens.dim()) != 2
                        or int(cached_tokens.size(0)) != 1
                        or int(cached_tokens.size(1)) < int(cached_len)
                        or int(input_ids.size(1)) < int(cached_len)
                    ):
                        recreate = True
                    else:
                        cached_prefix = cached_tokens[:, : int(cached_len)]
                        input_prefix = input_ids[:, : int(cached_len)].detach()
                        if cached_prefix.device != input_prefix.device:
                            cached_prefix = cached_prefix.to(
                                input_prefix.device,
                                non_blocking=True,
                            )
                        if not torch.equal(cached_prefix, input_prefix):
                            recreate = True

        if recreate:
            if request_kv_cache is not None and not getattr(request_kv_cache, "is_released", False):
                try:
                    request_kv_cache.release()
                except Exception:
                    pass
            request_kv_cache = request_cls(
                kvCachePool=kv_cache_pool,
                page_len=kv_cache_pool.page_len,
                seq_init_len=0,
            )
            setattr(
                request_kv_cache,
                "_pool_reset_counter",
                int(getattr(kv_cache_pool, "reset_counter", 0)),
            )
            setattr(self, attr_name, request_kv_cache)
            if tokens_attr_name is not None:
                setattr(self, tokens_attr_name, None)
            if reuse_len_attr_name is not None:
                setattr(self, reuse_len_attr_name, None)
        else:
            setattr(
                request_kv_cache,
                "_pool_reset_counter",
                int(getattr(kv_cache_pool, "reset_counter", 0)),
            )

        return request_kv_cache

    def _remember_request_cache_tokens(
        self,
        *,
        tokens_attr_name: str,
        input_ids: torch.LongTensor,
    ) -> None:
        setattr(
            self,
            tokens_attr_name,
            input_ids.detach().clone(memory_format=torch.contiguous_format),
        )

    def _sync_request_cache_after_tree_truncation(
        self,
        request_kv_cache,
        *,
        tree_size_before: int,
        tree_size_after: int,
    ) -> None:
        """Rollback speculative KV writes for nodes removed by tree truncation."""
        removed = int(tree_size_before) - int(tree_size_after)
        if removed <= 0:
            return
        request_kv_cache.decrement(int(removed))

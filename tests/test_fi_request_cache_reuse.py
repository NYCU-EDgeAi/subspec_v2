from types import SimpleNamespace

import torch

from specdecodes.models.generators.base import GeneratorBase
from specdecodes.models.generators.flashinfer_cache_mixin import FlashInferCacheMixin


class _DummyTargetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=4096)
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def forward(self, *_args, **_kwargs):
        return None


class _DummyTokenizer:
    eos_token_id = 2


class _DummyGenerator(FlashInferCacheMixin, GeneratorBase):
    def _generate(self, *_args, **_kwargs):  # pragma: no cover - not used here
        raise NotImplementedError


class _DummyPool:
    def __init__(self):
        self.page_len = 16
        self.max_pages = 8
        self.reset_counter = 0

    def num_free_pages(self):
        return 0

    def reset(self):
        self.reset_counter += 1


class _DummyRequestCache:
    def __init__(self, kvCachePool, page_len, seq_init_len):
        self.kvCachePool = kvCachePool
        self.page_len = page_len
        self.kv_len = int(seq_init_len)
        self.is_released = False
        self.released_count = 0

    def get_seq_length(self):
        return self.kv_len

    def release(self):
        self.is_released = True
        self.released_count += 1

    def decrement(self, num_tokens: int = 1):
        self.kv_len = max(0, int(self.kv_len) - int(num_tokens))


def test_ensure_request_kv_cache_reuses_for_growing_prompt():
    generator = _DummyGenerator(
        target_model=_DummyTargetModel(),
        tokenizer=_DummyTokenizer(),
    )
    pool = _DummyPool()

    req1 = generator._ensure_request_kv_cache(
        attr_name="_fi_request_kv_cache",
        request_cls=_DummyRequestCache,
        kv_cache_pool=pool,
        input_ids_len=10,
    )
    req1.kv_len = 8

    req2 = generator._ensure_request_kv_cache(
        attr_name="_fi_request_kv_cache",
        request_cls=_DummyRequestCache,
        kv_cache_pool=pool,
        input_ids_len=12,
    )

    assert req2 is req1
    assert req2.is_released is False


def test_ensure_request_kv_cache_recreates_on_session_boundary():
    generator = _DummyGenerator(
        target_model=_DummyTargetModel(),
        tokenizer=_DummyTokenizer(),
    )
    pool = _DummyPool()

    req1 = generator._ensure_request_kv_cache(
        attr_name="_fi_request_kv_cache",
        request_cls=_DummyRequestCache,
        kv_cache_pool=pool,
        input_ids_len=10,
    )
    req1.kv_len = 10

    req2 = generator._ensure_request_kv_cache(
        attr_name="_fi_request_kv_cache",
        request_cls=_DummyRequestCache,
        kv_cache_pool=pool,
        input_ids_len=10,
    )

    assert req2 is not req1
    assert req1.is_released is True


def test_ensure_request_kv_cache_recreates_after_pool_reset():
    generator = _DummyGenerator(
        target_model=_DummyTargetModel(),
        tokenizer=_DummyTokenizer(),
    )
    pool = _DummyPool()

    req1 = generator._ensure_request_kv_cache(
        attr_name="_fi_request_kv_cache",
        request_cls=_DummyRequestCache,
        kv_cache_pool=pool,
        input_ids_len=10,
    )
    req1.kv_len = 8

    pool.reset()

    req2 = generator._ensure_request_kv_cache(
        attr_name="_fi_request_kv_cache",
        request_cls=_DummyRequestCache,
        kv_cache_pool=pool,
        input_ids_len=12,
    )

    assert req2 is not req1
    assert req1.is_released is True


def test_ensure_request_kv_cache_reuses_when_prefix_matches():
    generator = _DummyGenerator(
        target_model=_DummyTargetModel(),
        tokenizer=_DummyTokenizer(),
    )
    pool = _DummyPool()

    input_ids_1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    req1 = generator._ensure_request_kv_cache(
        attr_name="_fi_request_kv_cache",
        request_cls=_DummyRequestCache,
        kv_cache_pool=pool,
        input_ids_len=int(input_ids_1.shape[1]),
        input_ids=input_ids_1,
        tokens_attr_name="_fi_request_tokens",
    )
    req1.kv_len = 8
    generator._remember_request_cache_tokens(
        tokens_attr_name="_fi_request_tokens",
        input_ids=input_ids_1,
    )

    input_ids_2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14]], dtype=torch.long)
    req2 = generator._ensure_request_kv_cache(
        attr_name="_fi_request_kv_cache",
        request_cls=_DummyRequestCache,
        kv_cache_pool=pool,
        input_ids_len=int(input_ids_2.shape[1]),
        input_ids=input_ids_2,
        tokens_attr_name="_fi_request_tokens",
    )

    assert req2 is req1
    assert req1.is_released is False


def test_ensure_request_kv_cache_recreates_when_prefix_mismatches():
    generator = _DummyGenerator(
        target_model=_DummyTargetModel(),
        tokenizer=_DummyTokenizer(),
    )
    pool = _DummyPool()

    input_ids_1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    req1 = generator._ensure_request_kv_cache(
        attr_name="_fi_request_kv_cache",
        request_cls=_DummyRequestCache,
        kv_cache_pool=pool,
        input_ids_len=int(input_ids_1.shape[1]),
        input_ids=input_ids_1,
        tokens_attr_name="_fi_request_tokens",
    )
    req1.kv_len = 8
    generator._remember_request_cache_tokens(
        tokens_attr_name="_fi_request_tokens",
        input_ids=input_ids_1,
    )

    # Prefix diverges before cached_len (8), so cache must be recreated.
    input_ids_2 = torch.tensor([[1, 2, 30, 4, 5, 6, 7, 8, 11, 12, 13, 14]], dtype=torch.long)
    req2 = generator._ensure_request_kv_cache(
        attr_name="_fi_request_kv_cache",
        request_cls=_DummyRequestCache,
        kv_cache_pool=pool,
        input_ids_len=int(input_ids_2.shape[1]),
        input_ids=input_ids_2,
        tokens_attr_name="_fi_request_tokens",
    )

    assert req2 is not req1
    assert req1.is_released is True


def test_ensure_request_kv_cache_honors_reuse_len_cap():
    generator = _DummyGenerator(
        target_model=_DummyTargetModel(),
        tokenizer=_DummyTokenizer(),
    )
    pool = _DummyPool()

    input_ids_1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    req1 = generator._ensure_request_kv_cache(
        attr_name="_fi_request_kv_cache",
        request_cls=_DummyRequestCache,
        kv_cache_pool=pool,
        input_ids_len=int(input_ids_1.shape[1]),
        input_ids=input_ids_1,
        tokens_attr_name="_fi_request_tokens",
        reuse_len_attr_name="_fi_request_reuse_len",
    )
    req1.kv_len = 10
    generator._remember_request_cache_tokens(
        tokens_attr_name="_fi_request_tokens",
        input_ids=input_ids_1,
    )
    setattr(generator, "_fi_request_reuse_len", 6)

    input_ids_2 = torch.tensor([[1, 2, 3, 4, 5, 6, 11, 12]], dtype=torch.long)
    req2 = generator._ensure_request_kv_cache(
        attr_name="_fi_request_kv_cache",
        request_cls=_DummyRequestCache,
        kv_cache_pool=pool,
        input_ids_len=int(input_ids_2.shape[1]),
        input_ids=input_ids_2,
        tokens_attr_name="_fi_request_tokens",
        reuse_len_attr_name="_fi_request_reuse_len",
    )

    assert req2 is req1
    assert int(req2.get_seq_length()) == 6

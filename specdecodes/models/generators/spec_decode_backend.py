"""Backend seam for speculative-decoding generators.

The per-algorithm `_generate` loop (v1 plain, v2 post-verify) is identical in shape
across attention backends; everything that differs between SDPA and FlashInfer is
KV-cache lifecycle + attention execution + prefill. `SpecDecodeBackend` is that seam:
a small interface the shared loop drives, with two adapters behind it.

Design validated by the throwaway prototype on branch `prototype/backend-seam`
(one shared loop drives both adapters, zero branching). The interface mirrors the
call sequence of the current generators/subspec_sd{,_fi}.py `_generate` loops:

    logits = backend.begin(input_ids, past_key_values)     # setup + chunked prefill
    loop:
        if backend.decode_headroom() == 0: break           # None => unbounded (SDPA)
        prev = backend.current_kv_len()
        tree = backend.speculate(last_token)               # draft forward
        cap the tree (shared, backend-agnostic)
        logits = backend.tree_forward(tree, position_offset, device)   # target forward
        verify (shared)
        backend.commit(hidden_indices=..., prev_kv_len=prev, decoded_tree_size=...,
                       finished=..., prune_tokens=...)      # cache bookkeeping
    backend.finalize(input_ids)

Two adapters => a real seam. `SdpaBackend` wraps the static/dynamic `Cache` path;
`FlashInferBackend` wraps the paged `RequestKvCache` + attention-wrapper path and owns
the FlashInfer state that currently leaks into generators/base.py (the request-cache
lifecycle, headroom, tree-truncation sync, and reuse-token memory).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class SpecDecodeBackend(ABC):
    """Attention/KV-cache backend that the shared spec-decode loop drives.

    Implementations own all state that differs between attention backends; the loop
    holds only algorithm state (input_ids, sampled tokens, stopping criteria).
    """

    @abstractmethod
    def begin(self, input_ids: torch.LongTensor, past_key_values: Any) -> torch.Tensor:
        """Set up backend state (wrapper / cuda-graph / request cache) and run chunked
        prefill over ``input_ids``. Return the next-token logits after the prompt."""

    @abstractmethod
    def current_kv_len(self) -> int:
        """Committed KV length — the ``prev_kv_len`` source passed back to ``commit``."""

    @abstractmethod
    def decode_headroom(self) -> int | None:
        """Appendable tokens before capacity, or ``None`` for an unbounded cache (SDPA).
        The loop stops a round early when this is ``<= 0``."""

    @abstractmethod
    def speculate(self, last_token_id: torch.LongTensor) -> Any:
        """Run the draft model to propose a candidate tree from ``last_token_id``."""

    @abstractmethod
    def tree_forward(self, tree: Any, *, position_offset: int, device: Any) -> Any:
        """Run the target model over the (already budget-capped) ``tree`` and return
        its outputs (``.logits`` used by verify). Backends reconcile any draft/tree
        cache-footprint mismatch internally before forwarding."""

    @abstractmethod
    def commit(
        self,
        *,
        hidden_indices: torch.Tensor,
        prev_kv_len: int,
        decoded_tree_size: int,
        finished: bool,
        prune_tokens: int,
    ) -> None:
        """Keep the accepted prefix in the KV cache, drop the rest, prune on finish."""

    @abstractmethod
    def finalize(self, input_ids: torch.LongTensor) -> None:
        """End-of-generate hook. FlashInfer records reuse tokens; SDPA is a no-op."""

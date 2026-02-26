import logging
import torch
import torch.nn as nn
from typing import Any
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, LogitNormalization
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, MaxLengthCriteria, MaxTimeCriteria, EosTokenCriteria, StopStringCriteria
from specdecodes.models.utils.cache_utils import TreeDynamicCache, TreeStaticCache


# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
# Several functions are simplified from GenerationMixin class.
class GeneratorBase(nn.Module):
    def __init__(self, target_model, tokenizer, draft_model=None, draft_params=None, cache_implementation="dynamic", **generator_kwargs):
        super().__init__()
        self.target_model = target_model
        self.tokenizer = tokenizer

        if draft_model is not None:
            self.draft_model = draft_model
            self.draft_params = draft_params
            self.draft_model.draft_params = draft_params
        else:
            self.draft_model = None

        self.cache_implementation = cache_implementation
        
        # Set prefill function same as forward so torch.compile() forward will not execute on prefill phase)
        self.target_model.prefill_forward = self.target_model.forward

    @property
    def config(self):
        return self.target_model.config
    
    @property
    def dtype(self):
        return self.target_model.dtype
    
    @property
    def device(self):
        return self.target_model.device
        
    def _get_logits_processor(
        self,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ):
        """
        Simplified HuggingFace's `LogitsProcessorList` for multinomial sampling.
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`] instances
        used for multinomial sampling.
        Visit https://github.com/huggingface/transformers/pull/5420/files for more details.
        """
        # Instantiate warpers list
        warpers = LogitsProcessorList()
        
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p))
        
        return warpers
    
    def _get_stopping_criteria(
        self,
        input_ids_length: torch.LongTensor = None,
        max_new_tokens: int = None,
        max_length: int = None,
        max_time: float = None,
        eos_token_tensor: torch.LongTensor = None,
        stop_strings: list[str] = None,
    ):
        criteria = StoppingCriteriaList()
        if max_new_tokens is not None:
            candidate_max_len = int(input_ids_length) + int(max_new_tokens)
            if max_length is None:
                max_length = candidate_max_len
            else:
                # Keep `max_length` as a hard upper bound even when max_new_tokens is provided.
                max_length = min(int(max_length), candidate_max_len)
            
        if max_length is not None:
            max_position_embeddings = getattr(self.target_model.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=max_time))
        if stop_strings is not None:
            if self.tokenizer is None:
                raise ValueError(
                    "There are one or more stop strings, either in the arguments to `generate` or in the "
                    "model's generation config, but we could not locate a tokenizer. When generating with "
                    "stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`."
                )
            criteria.append(StopStringCriteria(stop_strings=stop_strings, tokenizer=self.tokenizer))
        if eos_token_tensor is not None:
            # EosTokenCriteria only checks last input token,
            # make sure not token is appended after eos_token_tensor during generation
            criteria.append(EosTokenCriteria(eos_token_id=eos_token_tensor))
        
        return criteria
    
    def _sample_token(
        self,
        logits: torch.FloatTensor,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        return_probs: bool = False,
    ):
        if do_sample:
            batch, seq_len, vocab_size = logits.shape
            
            # Flatten logits for sampling
            logits = logits.view(-1, vocab_size)
            
            # Apply logits warper
            next_token_scores = logits_processor(None, logits)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(next_token_scores, dim=-1)
            
            if return_probs: # return sample prob
                return probs.view(batch, seq_len, vocab_size) # preserve shape
            else: # return sampled token
                token = torch.multinomial(probs, num_samples=1)
                return token.view(batch, seq_len) # preserve shape

        else:
            
            if return_probs: # return sample prob
                return torch.softmax(logits, dim=-1)
            else: # return sampled token
                return torch.argmax(logits, dim=-1)

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessor,
        do_sample: bool,
        *args,
        **kwargs,
    ):
        r"""
        This method is expected to be implemented by subclasses.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=None,
        max_length=None,
        do_sample=True,
        stop_strings=None,
        stream_callback=None,
        **model_kwargs,
    ):        
        # 1. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            input_ids_length=input_ids.shape[1],
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            eos_token_tensor=self.tokenizer.eos_token_id,
            stop_strings=stop_strings
        )
        
        # 2. prepare logits processor (if `do_sample` is `True`)
        logits_processor = (
            self._get_logits_processor(
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k,
            ) if do_sample else None
        )
        
        # 3. generate
        if stream_callback is not None:
            model_kwargs["stream_callback"] = stream_callback
        results = self._generate(
            input_ids=input_ids,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            do_sample=do_sample,
            **model_kwargs,
        )
        return results

    def _maybe_stream(self, stream_callback, token_ids: torch.LongTensor):
        if stream_callback is None:
            return
        stream_callback(token_ids)

    def _chunked_prefill_forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any,
        *,
        prefill_chunk_size: int | None,
        use_position_ids: bool = True,
        model_forward_kwargs: dict[str, Any] | None = None,
        prefill_forward_kwargs: dict[str, Any] | None = None,
    ):
        """Run prefill in chunks to reduce peak memory, returning the last forward outputs.

        This helper only performs forward passes and updates `past_key_values.seq_len`.
        Callers typically read `outputs.logits` from the returned object.
        """
        model_forward_kwargs = model_forward_kwargs or {}
        prefill_forward_kwargs = prefill_forward_kwargs or {}

        current_kv_len = past_key_values.get_seq_length()
        prefill_tokens = input_ids[:, current_kv_len:]
        prefill_length = prefill_tokens.size(1)

        chunk_size = (
            prefill_length
            if prefill_chunk_size is None
            else min(prefill_length, int(prefill_chunk_size))
        )

        outputs = None
        for start in range(0, prefill_length, chunk_size):
            chunk = prefill_tokens[:, start : start + chunk_size]
            current_kv_len = past_key_values.get_seq_length()
            cache_position = torch.arange(
                current_kv_len,
                current_kv_len + chunk.size(1),
                dtype=torch.long,
                device=input_ids.device,
            )

            forward_common_kwargs: dict[str, Any] = {
                "past_key_values": past_key_values.cache,
                "cache_position": cache_position,
            }
            if use_position_ids:
                forward_common_kwargs["position_ids"] = cache_position.unsqueeze(0)

            # Last chunk returns logits (and optionally other outputs); earlier chunks only update KV.
            if start + chunk_size < prefill_length:
                self.target_model.model(chunk, **forward_common_kwargs, **model_forward_kwargs)
            else:
                outputs = self.target_model.prefill_forward(
                    chunk,
                    **forward_common_kwargs,
                    logits_to_keep=1,
                    **prefill_forward_kwargs,
                )

            past_key_values.seq_len += chunk.size(1)

        return outputs

    def _apply_tokenwise_stopping_criteria(
        self,
        input_ids: torch.LongTensor,
        sampled_tokens: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
    ):
        """Apply stopping criteria token-by-token over a generated token block.

        Returns: (finished, updated_input_ids, kept_sampled_tokens, prune_tokens)
        where `prune_tokens` counts tokens removed from the tail after stop.
        """
        finished = False
        prune_tokens = 0

        # `stopping_criteria` (e.g., MaxLengthCriteria) expects to see the full
        # generated sequence. `input_ids` already includes `sampled_tokens` when
        # this helper is called, so we simulate the incremental growth.
        base_len = int(input_ids.shape[1] - sampled_tokens.shape[1])

        for k in range(sampled_tokens.shape[1]):
            cur_len = base_len + k + 1
            res = stopping_criteria(input_ids[:, :cur_len], None)
            finished = bool(res.item()) if hasattr(res, "item") else bool(res)
            if finished:
                prune_tokens = sampled_tokens.shape[1] - k - 1
                if prune_tokens > 0:
                    input_ids = input_ids[:, :-prune_tokens]
                break

        kept = (
            sampled_tokens
            if prune_tokens == 0
            else sampled_tokens[:, : sampled_tokens.shape[1] - prune_tokens]
        )
        return finished, input_ids, kept, prune_tokens

    def _remaining_token_budget(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
    ):
        """Return remaining new-token budget until `max_length` (None if unbounded)."""
        max_length = getattr(stopping_criteria, "max_length", None)
        if max_length is None:
            return None
        return max(0, int(max_length) - int(input_ids.shape[1]))

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

    def _candidate_decode_budget(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
    ):
        """Max candidate nodes/timesteps we may decode this step without KV overflow.

        Candidate decode includes one boundary/root context token, so budget is
        `remaining_new_tokens + 1` when `max_length` is bounded.
        """
        remaining = self._remaining_token_budget(input_ids, stopping_criteria)
        if remaining is None:
            return None
        return max(1, int(remaining) + 1)

    def _cap_draft_ids_to_budget(
        self,
        draft_ids: torch.LongTensor,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
    ):
        budget = self._candidate_decode_budget(input_ids, stopping_criteria)
        if budget is None:
            return draft_ids
        if int(draft_ids.shape[1]) <= int(budget):
            return draft_ids
        return draft_ids[:, : int(budget)]

    def _cap_tree_to_budget(
        self,
        tree,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        *,
        skip_nodes: int = 0,
    ) -> int:
        """Truncate tree to fit this step's decode budget and return decoded node count."""
        budget = self._candidate_decode_budget(input_ids, stopping_criteria)
        if budget is None:
            return max(0, int(tree.size()) - int(skip_nodes))

        max_tree_nodes = int(skip_nodes) + int(budget)
        if int(tree.size()) > max_tree_nodes:
            if hasattr(tree, "truncate_prefix"):
                tree.truncate_prefix(max_tree_nodes)
            else:
                tree.prune_to_top_n(max_tree_nodes)

        return max(0, int(tree.size()) - int(skip_nodes))

    def _resolve_pending_chunk_size(
        self,
        hidden_indices: torch.Tensor | None,
        fallback_size: int,
    ) -> int:
        """Return a safe pending-chunk length covering all referenced indices."""
        size = max(0, int(fallback_size))
        if hidden_indices is None:
            return size
        if int(hidden_indices.numel()) <= 0:
            return size

        max_index = int(hidden_indices.max().item()) + 1
        if max_index > size:
            size = max_index
        return size

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

    def _remap_hidden_indices_after_tree_prune(
        self,
        hidden_indices: torch.Tensor | None,
        kept_old_indices: torch.Tensor | None,
        *,
        method_name: str,
    ) -> torch.Tensor | None:
        """Remap cached tree indices after `Tree.prune_to_depth` reindexes nodes."""
        if hidden_indices is None or kept_old_indices is None:
            return hidden_indices
        if int(hidden_indices.numel()) == 0:
            return hidden_indices
        if int(kept_old_indices.numel()) == 0:
            raise RuntimeError(
                f"Cannot remap hidden indices after prune in {method_name}: "
                "kept_old_indices is empty while hidden_indices is non-empty."
            )

        kept = kept_old_indices.to(device=hidden_indices.device, dtype=torch.long)
        max_old_idx = int(kept.max().item())
        old_to_new = torch.full(
            (max_old_idx + 1,),
            -1,
            dtype=torch.long,
            device=hidden_indices.device,
        )
        old_to_new[kept] = torch.arange(
            int(kept.numel()),
            dtype=torch.long,
            device=hidden_indices.device,
        )

        if torch.any(hidden_indices < 0) or torch.any(hidden_indices > max_old_idx):
            raise RuntimeError(
                f"Hidden index out of prune remap range in {method_name}: "
                f"max_hidden={int(hidden_indices.max().item())}, max_kept_old={max_old_idx}"
            )

        remapped = old_to_new[hidden_indices.long()]
        if torch.any(remapped < 0):
            raise RuntimeError(
                f"Deferred hidden index dropped by prune in {method_name}."
            )
        return remapped
    
    def create_kv_cache(
        self,
        cache_implementation,
        max_cache_len=None,
        max_batch_size=None,
        config=None,
        device=None,
        dtype=None,
    ):
        if cache_implementation == "dynamic":
            return TreeDynamicCache()
        
        elif cache_implementation == "static":
            return TreeStaticCache(
                max_cache_len=max_cache_len,
                max_batch_size=max_batch_size,
                config=config,
                device=device,
                dtype=dtype,
            )

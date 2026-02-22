"""SubSpec SD Draft Model with FlashInfer v3.

Clean implementation with CUDA graph support (currently disabled for testing).
"""

import torch
import nvtx
from copy import deepcopy

from ..utils.cpu_tree import Tree
from .base import DraftModelBase, TreeData, TreeMaskCache
from ..utils.flashinfer_v3.cache_manager import (
    RequestKvCache,
    KvCacheBatchPosition,
    getKvCacheBatchPosition,
)
from ..utils.flashinfer_v3.attention_wrapper import FlashinferAttentionWrapper


def _share_param_deepcopy(model):
    """Deep copy model while sharing parameters and buffers."""
    memo = {id(p): p for _, p in model.named_parameters()}
    memo.update({id(b): b for _, b in model.named_buffers()})
    return deepcopy(model, memo=memo)


class SubSpecSDDraftModel(DraftModelBase):
    """SubSpec SD Draft Model with FlashInfer v3."""

    def __init__(self, *args, cuda_graph_enabled: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._cuda_graph_enabled = cuda_graph_enabled
        self._wrapper: FlashinferAttentionWrapper = None
        self._pool = None
        self._count = 0

        # State preserved for postspec
        self._token_ids = None
        self._position_ids = None
        self._parent_probs = None
        self._request_kv_cache = None
        self._postspec_count = 0

        # CUDA graph state (initialized in init_cuda_graph_runner)
        self._graph = None
        self._output_buffer = None
        # Static buffers for CUDA graph
        self._input_ids_buf = None
        self._position_ids_buf = None
        self._batch_position_buf = None
        # KV position buffers
        self._seq_indptr_buf = None
        self._kv_page_indptr_buf = None
        self._kv_page_indices_buf = None
        self._kv_last_page_len_buf = None
        self._batch_indices_buf = None
        self._positions_buf = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, *, target_model=None,
                        torch_dtype=torch.float32, cuda_graph_enabled: bool = True, **kwargs):
        eos_token_id = kwargs.pop("eos_token_id", None)
        model = cls(
            base_model=_share_param_deepcopy(target_model),
            eos_token_id=eos_token_id,
            cuda_graph_enabled=cuda_graph_enabled,
            **kwargs
        )
        return model.to(dtype=torch_dtype)

    def forward(self, input_ids, with_softmax=False, *args, **kwargs):
        input_ids, kwargs = self._align_forward_inputs_to_model_device(input_ids, kwargs)
        logits = self.model(input_ids, *args, **kwargs).logits
        if with_softmax:
            logits = torch.softmax(logits / self.draft_params.temperature, dim=-1)
        return logits

    @torch.no_grad()
    def speculate(self, input_ids: torch.Tensor, request_kv_cache: RequestKvCache, **kwargs) -> Tree:
        """Generate draft tree."""
        device = input_ids.device
        dtype = self.model.lm_head.weight.dtype
        batch_size, input_len = input_ids.shape

        # Lazy init FlashInfer wrapper
        if self._wrapper is None:
            cfg = self.model.config
            self._wrapper = FlashinferAttentionWrapper(
                cfg.num_attention_heads, cfg.num_key_value_heads,
                cfg.hidden_size, request_kv_cache.kvCachePool.page_len
            )
        self._pool = request_kv_cache.kvCachePool
        self._request_kv_cache = request_kv_cache

        assert batch_size == 1, "Only support batch_size=1 for now."

        # Get initial kv_len
        with nvtx.annotate("kv_init"):
            kv_len = request_kv_cache.get_seq_length()
            if isinstance(kv_len, torch.Tensor):
                kv_len = kv_len.item()

        # First forward pass (decode mode)
        with nvtx.annotate("draft_prefill", color="red"):
            request_kv_cache.increment()
            batch_position = getKvCacheBatchPosition(
                [request_kv_cache], mode="decode", device=device,
            )
            self._wrapper.prepareAttention(
                "decode", batch_position, self._pool.page_len,
                "NONE", self._pool.cache_data[0].dtype,
            )
            position_ids = torch.full((batch_size, input_len), kv_len, device=device, dtype=torch.long)
            sampled_probs = self(
                input_ids, with_softmax=True, logits_to_keep=1,
                position_ids=position_ids,
                kvCachePool=self._pool,
                batch_position=batch_position,
                mode="decode",
                flashinferWrapper=self._wrapper,
            )
            kv_len += input_len

        # Init tree state
        with nvtx.annotate("draft_init_state"):
            parent_probs = torch.ones((1, 1), device=device, dtype=dtype)
            position_ids = torch.full((batch_size, self.draft_params.topk_len), kv_len, device=device, dtype=torch.long)

        # Create tree structures
        root_id = input_ids[0, -1]
        self.tree = Tree(root_id, dtype)
        self.tree_data = TreeData()
        self.tree_mask_cache = TreeMaskCache(
            prefix_len=kv_len,
            sample_len=self.draft_params.topk_len,
            max_cache_len=None,
            dtype=dtype,
            device=device,
        )

        # Main speculation loop
        for depth_i in range(self.draft_params.max_depth):
            with nvtx.annotate("draft_sample", color="green"):
                token_ids, child_probs, parent_indices = self.topk_sampling(
                    sampled_probs, parent_probs, self.draft_params.topk_len
                )
                parent_probs = child_probs

            with nvtx.annotate("tree_data/update", color="green"):
                self.tree_data.update(token_ids, child_probs, parent_indices)

            with nvtx.annotate("tree_mask/update"):
                tree_attention_mask = self.tree_mask_cache.update_tree_mask(
                    parent_indices, return_invert=False
                )

            num_tokens = int(self.draft_params.topk_len)
            if not self._has_postspec_headroom(
                step_tokens=num_tokens,
                request_kv_cache=request_kv_cache,
            ):
                break

            with nvtx.annotate("draft_forward", color="red"):
                request_kv_cache.increment(num_tokens)

                batch_position = getKvCacheBatchPosition(
                    request_kv_caches=[request_kv_cache],
                    mode="tree",
                    device=device,
                    treeTokens=num_tokens,
                )
                self._wrapper.prepareAttention(
                    "tree", batch_position, self._pool.page_len,
                    "NONE", self._pool.cache_data[0].dtype,
                    attention_mask=tree_attention_mask,
                )

                if self._has_cuda_graph:
                    # Use CUDA graph replay
                    sampled_probs = self._tree_step(token_ids, position_ids, batch_position)
                else:
                    # Direct forward pass
                    sampled_probs = self(
                        token_ids, with_softmax=True, past_key_values=None,
                        position_ids=position_ids,
                        kvCachePool=self._pool,
                        batch_position=batch_position,
                        mode="tree",
                        flashinferWrapper=self._wrapper,
                    )
                kv_len += num_tokens

            with nvtx.annotate("state_update"):
                position_ids += 1

        # Finalize tree
        with nvtx.annotate("tree_finalize"):
            self.tree.add_nodes(*self.tree_data.get_data())

        # Preserve state for postspec
        self._token_ids = token_ids
        self._position_ids = position_ids
        self._parent_probs = parent_probs
        self._count += 1

        return self.tree

    def init_postspec(self):
        """Initialize postspec state."""
        self.tree_data = TreeData()
        self._postspec_count = 0

    @torch.no_grad()
    def postspec(self):
        """Post-speculation step (called during target model forward)."""
        if self._count == 0 or self._postspec_count > self.draft_params.max_depth - 1:
            return False
        if not self._has_postspec_headroom(
            step_tokens=int(self.draft_params.topk_len),
            request_kv_cache=getattr(self, "_request_kv_cache", None),
        ):
            return False
        with nvtx.annotate("postspec_step", color="blue"):
            progressed = self._speculate_once()
        if not progressed:
            return False
        self._postspec_count += 1
        return True

    @torch.no_grad()
    def _speculate_once(self):
        """Single speculation step for postspec."""
        tree_attention_mask = self.tree_mask_cache.get_tree_mask(return_invert=False)
        token_ids = self._token_ids
        parent_probs = self._parent_probs
        position_ids = self._position_ids
        request_kv_cache = self._request_kv_cache
        num_tokens = int(self.draft_params.topk_len)
        if not self._has_postspec_headroom(
            step_tokens=num_tokens,
            request_kv_cache=request_kv_cache,
        ):
            return False

        with nvtx.annotate("draft_forward", color="red"):
            request_kv_cache.increment(num_tokens)

            batch_position = getKvCacheBatchPosition(
                request_kv_caches=[request_kv_cache],
                mode="tree",
                device=token_ids.device,
                treeTokens=num_tokens,
            )
            self._wrapper.prepareAttention(
                "tree", batch_position, self._pool.page_len,
                "NONE", self._pool.cache_data[0].dtype,
                attention_mask=tree_attention_mask,
            )
            sampled_probs = self(
                token_ids, with_softmax=True, past_key_values=None,
                position_ids=position_ids,
                kvCachePool=self._pool,
                batch_position=batch_position,
                mode="tree",
                flashinferWrapper=self._wrapper,
            )

        with nvtx.annotate("draft_sample", color="green"):
            token_ids, child_probs, parent_indices = self.topk_sampling(
                sampled_probs, parent_probs, self.draft_params.topk_len
            )

        with nvtx.annotate("tree_update", color="green"):
            self.tree_data.update(token_ids, child_probs, parent_indices)
            self.tree_mask_cache.update_tree_mask(parent_indices)

        # Update preserved state
        self._token_ids = token_ids
        self._parent_probs = child_probs
        self._position_ids += 1
        return True

    def update_tree_after_post(self) -> Tree:
        """Return finalized tree after post-speculation."""
        self.tree.add_nodes(*self.tree_data.get_data())
        return self.tree

    def init_cuda_graph_runner(self, device: torch.device):
        """Initialize and capture CUDA graph for tree forward.

        This captures the tree forward pass into a CUDA graph for faster replay.
        Must be called AFTER first speculate() so FlashInfer wrapper state is ready.
        """
        if not self._cuda_graph_enabled or self._graph is not None:
            return

        print("[v3] Capturing CUDA graph for tree forward...")
        self.model.eval()

        B = 1
        L = self.draft_params.topk_len

        # Allocate static input buffers
        self._input_ids_buf = torch.zeros((B, L), dtype=torch.long, device=device)
        self._position_ids_buf = torch.zeros((B, L), dtype=torch.long, device=device)

        # Allocate KV position buffers
        self._seq_indptr_buf = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        self._kv_page_indptr_buf = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        self._kv_page_indices_buf = torch.zeros((self._pool.max_pages,), dtype=torch.int32, device=device)
        self._kv_last_page_len_buf = torch.zeros((B,), dtype=torch.int32, device=device)
        self._batch_indices_buf = torch.zeros((L,), dtype=torch.int32, device=device)
        self._positions_buf = torch.zeros((L,), dtype=torch.int32, device=device)

        # Create static batch position
        self._batch_position_buf = KvCacheBatchPosition(
            seq_indptr=self._seq_indptr_buf,
            kv_page_indptr=self._kv_page_indptr_buf,
            kv_page_indices=self._kv_page_indices_buf,
            kv_last_page_len=self._kv_last_page_len_buf,
            batch_indices=self._batch_indices_buf,
            positions=self._positions_buf,
        )

        # Use a separate stream for capture
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            # Warmup runs (outside graph capture)
            # NOTE: Don't call prepareAttention here - rely on state from last speculate()
            for _ in range(2):
                _ = self(
                    self._input_ids_buf,
                    with_softmax=True,
                    position_ids=self._position_ids_buf,
                    kvCachePool=self._pool,
                    batch_position=self._batch_position_buf,
                    mode="tree",
                    flashinferWrapper=self._wrapper,
                )

            torch.cuda.current_stream().wait_stream(stream)

            # Capture graph
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=stream):
                self._output_buffer = self(
                    self._input_ids_buf,
                    with_softmax=True,
                    position_ids=self._position_ids_buf,
                    kvCachePool=self._pool,
                    batch_position=self._batch_position_buf,
                    mode="tree",
                    flashinferWrapper=self._wrapper,
                )

        print("[v3] CUDA graph captured successfully")

    def _tree_step(self, token_ids: torch.Tensor, position_ids: torch.Tensor,
                   batch_position: KvCacheBatchPosition) -> torch.Tensor:
        """Execute tree forward using CUDA graph replay.

        Copies fresh data to static buffers and replays the captured graph.
        """
        B, L = token_ids.shape
        if L > self.draft_params.topk_len:
            raise ValueError(f"token_ids length {L} exceeds topk_len {self.draft_params.topk_len}")

        # Copy input data to static buffers
        self._input_ids_buf[:, :L].copy_(token_ids)
        self._position_ids_buf[:, :L].copy_(position_ids)

        # Copy KV position data
        self._seq_indptr_buf.copy_(batch_position.seq_indptr)
        self._kv_page_indptr_buf.copy_(batch_position.kv_page_indptr)
        self._kv_last_page_len_buf.copy_(batch_position.kv_last_page_len)
        self._batch_indices_buf[:L].copy_(batch_position.batch_indices[:L])
        self._positions_buf[:L].copy_(batch_position.positions[:L])

        # Copy page indices (only the used portion)
        n_pages = batch_position.kv_page_indptr[1].item()
        self._kv_page_indices_buf[:n_pages].copy_(batch_position.kv_page_indices[:n_pages])

        # Replay graph
        self._graph.replay()
        return self._output_buffer

    @property
    def _has_cuda_graph(self) -> bool:
        """Check if CUDA graph is captured and ready."""
        return self._graph is not None

    @property
    def had_first_speculate(self):
        return self._count > 0

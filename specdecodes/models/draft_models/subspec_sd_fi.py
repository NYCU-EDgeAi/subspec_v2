import torch
import nvtx

from ..utils.cpu_tree import Tree
from .base import DraftModelBase, FIFirstStepContract, TreeData, TreeMaskCache
from copy import deepcopy

from ..utils.flashinfer.cache_manager import (
    KvCacheBatchPosition,
    getKvCacheBatchPosition,
)
from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper


def share_param_deepcopy(model):
    # Build the memo dictionary from the model's parameters (and optionally buffers)
    model_memo = {}
    for _, param in model.named_parameters():
        model_memo[id(param)] = param
    for _, buf in model.named_buffers():
        model_memo[id(buf)] = buf

    # Clone the model using the memo dictionary.
    share_model = deepcopy(model, memo=model_memo)
    return share_model

class SubSpecSDDraftModel(DraftModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.had_first_speculate = False
        self.postspec_count = 0

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path=None,
        *model_args,
        target_model = None,
        torch_dtype=torch.float32,
        **model_kwargs
    ):
        # Remove the following arguments from model_kwargs, cause AutoModelForCausalLM does not accept them
        eos_token_id = model_kwargs.pop("eos_token_id", None)
        
        base_model = share_param_deepcopy(target_model)
        model = cls(base_model=base_model, eos_token_id=eos_token_id, *model_args, **model_kwargs)
        
        # Convert the model to the desired dtype and return
        model.to(dtype=torch_dtype)
        return model
    
    def forward(self, input_ids, with_softmax=False, *model_args, **kwargs):
        input_ids, kwargs = self._align_forward_inputs_to_model_device(input_ids, kwargs)
        logits = self.model(input_ids, *model_args, **kwargs).logits
        if with_softmax:
            logits = torch.softmax(logits/self.draft_params.temperature, dim=-1)
            
        return logits

    def _first_step_contract(self) -> FIFirstStepContract:
        return FIFirstStepContract(
            attention_mode="decode",
            batch_position_mode="decode",
            cache_increment="one",
            position_ids_style="constant_kv_len",
        )
    
    def init_cuda_graph_runner(
        self,
        device: torch.device,
    ):
        if bool(getattr(self, "_cuda_graph_inited", False)):
            return

        self._fi_cuda_graph_reset_state(decode_chunk_size=int(self.draft_params.topk_len))
        self.model.eval()
        kvCachePool = self.kvCachePool

        B = 1
        tree_L = self.decode_chunk_size
        first_L = 1

        self.input_ids_buf = torch.zeros((B, tree_L), dtype=torch.long, device=device)
        self.position_ids_buf = torch.zeros((B, tree_L), dtype=torch.long, device=device)
        self._fi_alloc_batch_position_buffers(
            prefix="",
            batch_size=B,
            token_count=tree_L,
            max_pages=int(kvCachePool.max_pages),
            device=device,
        )

        self.first_input_ids_buf = torch.zeros((B, first_L), dtype=torch.long, device=device)
        self.first_position_ids_buf = torch.zeros((B, first_L), dtype=torch.long, device=device)
        self._fi_alloc_batch_position_buffers(
            prefix="first_",
            batch_size=B,
            token_count=first_L,
            max_pages=int(kvCachePool.max_pages),
            device=device,
        )

        if not hasattr(self, "flashinferWrapper"):
            raise ValueError("flashinferWrapper not found in draft model.")

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            dummy_tok = torch.zeros((B, tree_L), dtype=torch.long, device=device)
            dummy_pos = torch.zeros_like(dummy_tok)

            for _ in range(2):
                _ = self(
                    dummy_tok,
                    with_softmax=True,
                    position_ids=dummy_pos,
                    kvCachePool=kvCachePool,
                    batch_position=self.batch_position,
                    mode="tree",
                    flashinferWrapper=self.flashinferWrapper,
                )

            torch.cuda.current_stream().wait_stream(stream)
            tree_cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(tree_cg, stream=stream):
                self.output_buffer = self(
                    self.input_ids_buf,
                    with_softmax=True,
                    position_ids=self.position_ids_buf,
                    kvCachePool=kvCachePool,
                    batch_position=self.batch_position,
                    mode="tree",
                    flashinferWrapper=self.flashinferWrapper,
                )

            first_cg = None
            if bool(getattr(self.flashinferWrapper, "decode_use_cuda_graph", False)):
                dummy_first_tok = torch.zeros((B, first_L), dtype=torch.long, device=device)
                dummy_first_pos = torch.zeros_like(dummy_first_tok)
                for _ in range(2):
                    _ = self(
                        dummy_first_tok,
                        with_softmax=True,
                        position_ids=dummy_first_pos,
                        kvCachePool=kvCachePool,
                        batch_position=self.first_batch_position,
                        mode="decode",
                        flashinferWrapper=self.flashinferWrapper,
                    )

                first_cg = torch.cuda.CUDAGraph()
                with torch.cuda.graph(first_cg, stream=stream):
                    self.first_output_buffer = self(
                        self.first_input_ids_buf,
                        with_softmax=True,
                        position_ids=self.first_position_ids_buf,
                        kvCachePool=kvCachePool,
                        batch_position=self.first_batch_position,
                        mode="decode",
                        flashinferWrapper=self.flashinferWrapper,
                    )

        self.graph = tree_cg
        self.first_graph = first_cg
        self._cuda_graph_inited = True


    def tree_step(
        self,
        token_ids: torch.Tensor,           # [1, L]  – same L as topk_len
        position_ids: torch.Tensor,        # [1, L]
        batch_position: KvCacheBatchPosition,
    ):
        return self._fi_replay_graph_step(
            prefix="",
            graph_attr="graph",
            output_attr="output_buffer",
            token_ids=token_ids,
            position_ids=position_ids,
            batch_position=batch_position,
            max_tokens=int(self.decode_chunk_size),
            require_exact_tokens=False,
            context="subspec_sd_fi.tree_step",
        )

    def first_step(
        self,
        token_ids: torch.Tensor,  # [1, 1]
        position_ids: torch.Tensor,  # [1, 1]
        batch_position: KvCacheBatchPosition,
    ):
        return self._fi_replay_graph_step(
            prefix="first_",
            graph_attr="first_graph",
            output_attr="first_output_buffer",
            token_ids=token_ids,
            position_ids=position_ids,
            batch_position=batch_position,
            max_tokens=1,
            require_exact_tokens=True,
            context="subspec_sd_fi.first_step",
        )
    
    @torch.no_grad()
    def update_tree(self, tree_data):
        if not tree_data.has_data():
            return self.tree
        with nvtx.annotate("tree_finalize"):
            with nvtx.annotate("tree_data/get"):
                data = tree_data.get_data()
            with nvtx.annotate("tree/apply"):
                self.tree.add_nodes(*data)
        return self.tree
    
    @torch.no_grad()
    def speculate(self, input_ids, request_kv_cache, **kwargs):

        self.had_first_speculate = True

        # 1) Obtain necessary parameters
        device = input_ids.device
        dtype = self.model.lm_head.weight.dtype
        batch_size, input_len = input_ids.shape
        self.request_kv_cache = request_kv_cache
        
        max_cache_len = None
        if not hasattr(self, 'flashinferWrapper'):
            self.flashinferWrapper = FlashinferAttentionWrapper(
                self.model.config.num_attention_heads,
                self.model.config.num_key_value_heads,
                self.model.config.hidden_size,
                request_kv_cache.kvCachePool.page_len,
                decode_use_cuda_graph=True,
            )
        self.kvCachePool = request_kv_cache.kvCachePool

        assert (self.flashinferWrapper is not None)
        assert batch_size == 1, "Only support batch_size=1 for now."

        
        # 2) Initialize kv_len & cache_position
        with nvtx.annotate("kv_init"):
            kv_len = request_kv_cache.get_seq_length()
            # convert kv_len to int if it is a tensor
            if isinstance(kv_len, torch.Tensor):
                kv_len = kv_len.item()

        # 3) First forward pass
        with nvtx.annotate("draft_prefill", color="red"):
            first_step_contract = self._first_step_contract()
            batch_position, position_ids = self._fi_prepare_first_step(
                request_kv_cache=request_kv_cache,
                kv_len=int(kv_len),
                input_len=int(input_len),
                batch_size=int(batch_size),
                device=device,
                contract=first_step_contract,
                get_batch_position_fn=getKvCacheBatchPosition,
            )
            self.flashinferWrapper.prepareAttention(
                first_step_contract.attention_mode,
                batch_position,
                request_kv_cache.kvCachePool.page_len,
                "NONE", #POS_ENCODING_MODE.NONE
                request_kv_cache.kvCachePool.cache_data[0].dtype,
            )  
            if self._fi_graph_enabled("first_graph"):
                sampled_probs = self.first_step(
                    input_ids,
                    position_ids,
                    batch_position=batch_position,
                )
            else:
                sampled_probs = self(
                    input_ids,
                    with_softmax=True,
                    logits_to_keep=1,
                    position_ids = position_ids,
                    kvCachePool=request_kv_cache.kvCachePool,
                    batch_position=batch_position,
                    mode=first_step_contract.attention_mode,
                    flashinferWrapper=self.flashinferWrapper,
                )
           
            kv_len += input_len
            

        with nvtx.annotate("draft_sample", color="green"):
            parent_probs = torch.ones((1, 1), device=device, dtype=dtype)
            token_ids, child_probs, parent_indices = self.topk_sampling(
                sampled_probs,
                parent_probs,
                self.draft_params.topk_len,
            )
            parent_probs = child_probs
        
        # 4) Create TreeData & TreeMaskCache to manage tree structure and intermediate data.
        root_id = input_ids[0, -1]
        self.tree = Tree(root_id, dtype)
        self.tree_data = TreeData()
        self.tree_mask_cache = TreeMaskCache(
            prefix_len=kv_len,
            sample_len=self.draft_params.topk_len,
            max_cache_len=max_cache_len,
            dtype=dtype,
            device=device,
        )

        # 5) First update of tree_data and tree_mask_cache.
        with nvtx.annotate("tree_update", color="green"):
            self.tree_data.update(token_ids, child_probs, parent_indices)
            self.tree_mask_cache.update_tree_mask(parent_indices)

        # Set initial frontier state for postspec/speculate_once continuation.
        self.token_ids = token_ids
        self.parent_probs = parent_probs
        self.position_ids = torch.full(
            (batch_size, self.draft_params.topk_len),
            kv_len,
            device=device,
            dtype=torch.long,
        )

        # 6) Main loop.
        for _ in range(self.draft_params.max_depth - 1):
            if not self.speculate_once():
                break

        self.update_tree(self.tree_data)
        return self.tree
    
    def init_postspec(self, *, rebuild_frontier: bool = False):
        self.tree_data = TreeData()
        self.postspec_count = 0
        if bool(rebuild_frontier):
            self._rebuild_postspec_frontier_from_tree()

    def _resolve_postspec_prefix_len(self) -> int:
        tree = getattr(self, "tree", None)
        tree_mask_cache = getattr(self, "tree_mask_cache", None)
        request_kv_cache = getattr(self, "request_kv_cache", None)

        cached_prefix_len = None
        if tree_mask_cache is not None:
            cached_prefix_len = getattr(tree_mask_cache, "prefix_len", None)
            if cached_prefix_len is not None:
                cached_prefix_len = int(cached_prefix_len)

        # Prefer recomputing from live request-cache state when available;
        # cached prefix_len can be stale after post-verify prune/rewrite.
        if tree is not None and request_kv_cache is not None:
            tree_size = int(tree.size())
            if tree_size > 0:
                seq_len = int(request_kv_cache.get_seq_length())
                prefix_len = int(seq_len) - int(tree_size) + 1
                if prefix_len <= 0:
                    raise RuntimeError(
                        "Invalid postspec prefix length in subspec_sd_fi: "
                        f"seq_len={seq_len}, tree_size={tree_size}, computed_prefix_len={prefix_len}"
                    )
                return int(prefix_len)

        return int(cached_prefix_len or 0)

    def _rebuild_postspec_frontier_from_tree(self) -> None:
        tree = getattr(self, "tree", None)
        tree_mask_cache = getattr(self, "tree_mask_cache", None)
        if tree is None or tree_mask_cache is None:
            return
        if int(tree.size()) <= 0:
            return

        leaves = list(getattr(tree, "available_leaves", []))
        if len(leaves) == 0:
            leaves = [int(tree.size()) - 1]

        leaf_indices = self._resize_frontier_leaf_indices(
            torch.tensor(leaves, dtype=torch.long, device="cpu"),
            width=int(self.draft_params.topk_len),
        )

        node_data = tree.get_tree_data(skip_nodes=0)
        token_ids_all = node_data["token_ids"]
        probs_all = node_data["cumulative_probabilities"]
        depths_all = node_data["depths"]

        frontier_token_ids = token_ids_all.index_select(0, leaf_indices)
        frontier_probs = probs_all.index_select(0, leaf_indices)
        frontier_depths = depths_all.index_select(0, leaf_indices)

        prefix_len = self._resolve_postspec_prefix_len()
        position_offset = int(prefix_len) - 1
        frontier_position_ids = frontier_depths + int(position_offset)

        model_device = self.model.lm_head.weight.device
        prob_dtype = self.model.lm_head.weight.dtype
        self.token_ids = frontier_token_ids.unsqueeze(0).to(
            device=model_device,
            dtype=torch.long,
            non_blocking=True,
        )
        self.parent_probs = frontier_probs.unsqueeze(0).to(
            device=model_device,
            dtype=prob_dtype,
            non_blocking=True,
        )
        self.position_ids = frontier_position_ids.unsqueeze(0).to(
            device=model_device,
            dtype=torch.long,
            non_blocking=True,
        )

        frontier_mask = tree.create_attention_mask(
            prefix_length=int(position_offset),
            skip_nodes=0,
            device="cpu",
        )[:, :, leaf_indices, :]

        self._assign_frontier_mask_cache(
            tree_mask_cache=tree_mask_cache,
            frontier_mask=frontier_mask,
            model_device=model_device,
        )
        tree_mask_cache.prefix_len = int(prefix_len)

    @staticmethod
    def _resize_frontier_leaf_indices(leaf_indices: torch.Tensor, width: int) -> torch.Tensor:
        width = int(width)
        if int(leaf_indices.numel()) >= width:
            return leaf_indices[:width]
        repeats = width // int(leaf_indices.numel())
        remainder = width % int(leaf_indices.numel())
        repeated = leaf_indices.repeat(int(repeats))
        if remainder > 0:
            repeated = torch.cat([repeated, leaf_indices[:remainder]], dim=0)
        return repeated

    @staticmethod
    def _assign_frontier_mask_cache(
        *,
        tree_mask_cache: TreeMaskCache,
        frontier_mask: torch.Tensor,
        model_device: torch.device,
    ) -> None:
        mask_method = getattr(tree_mask_cache, "tree_mask_update_method", "dynamic")
        if mask_method == "static":
            static_cache = tree_mask_cache.tree_mask_cache
            mask_rows = int(frontier_mask.shape[2])
            mask_cols = int(frontier_mask.shape[3])
            if static_cache.shape[2] < mask_rows or static_cache.shape[3] < mask_cols:
                tree_mask_cache.tree_mask_update_method = "dynamic"
                tree_mask_cache.tree_mask_cache = frontier_mask.to(
                    device=model_device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
                return
            static_cache.zero_()
            static_cache[:, :, :mask_rows, :mask_cols] = frontier_mask.to(
                device=static_cache.device,
                dtype=torch.bool,
                non_blocking=True,
            )
            tree_mask_cache.current_len = int(mask_cols)
            return
        tree_mask_cache.tree_mask_cache = frontier_mask.to(
            device=model_device,
            dtype=torch.bool,
            non_blocking=True,
        )
        
    @torch.no_grad()
    def postspec(self):
        # Opt-in suspend hook: a caller can set `_suspend_postspec = True` to pause
        # postspec (returns False, does no work, leaves postspec_count untouched).
        # Defaults off, so no behavior change for the normal path.
        if getattr(self, "_suspend_postspec", False):
            return False
        if not self.had_first_speculate:
            return False
        if self.postspec_count > (self.draft_params.max_depth - 1):
            return False
        if not self._has_postspec_headroom(
            step_tokens=int(self.draft_params.topk_len),
            request_kv_cache=getattr(self, "request_kv_cache", None),
        ):
            return False
        with nvtx.annotate("postspec_step", color="blue"):
            progressed = self.speculate_once()
        if not progressed:
            return False
        self.postspec_count += 1
        return True

    @torch.no_grad()
    def speculate_once(self, **kwargs):
        tree_attention_mask = self.tree_mask_cache.get_tree_mask(return_invert=False)
        token_ids = self.token_ids
        parent_probs = self.parent_probs
        position_ids = self.position_ids

        request_kv_cache = self.request_kv_cache
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
                mode='tree',
                device=token_ids.device,                    
                treeTokens=num_tokens,
            )
            self.flashinferWrapper.prepareAttention(
                'tree',
                batch_position,
                request_kv_cache.kvCachePool.page_len,
                "NONE", #POS_ENCODING_MODE.NONE
                request_kv_cache.kvCachePool.cache_data[0].dtype,
                attention_mask=tree_attention_mask,
            )

            if self._fi_graph_enabled("graph"):
                # use CUDA graph
                sampled_probs = self.tree_step(
                    token_ids,
                    position_ids,
                    batch_position=batch_position,  
                )
            else:
                sampled_probs = self(
                    token_ids,
                    with_softmax=True,
                    past_key_values=None,
                    position_ids=position_ids,
                    kvCachePool=request_kv_cache.kvCachePool,
                    batch_position=batch_position,
                    mode='tree',
                    flashinferWrapper=self.flashinferWrapper,
                )

        with nvtx.annotate("draft_sample", color="green"):
            token_ids, child_probs, parent_indices = self.topk_sampling(
                sampled_probs,
                parent_probs,
                self.draft_params.topk_len
            )
            parent_probs = child_probs
            
        with nvtx.annotate("tree_update", color="green"):
            self.tree_data.update(token_ids, child_probs, parent_indices)
            self.tree_mask_cache.update_tree_mask(parent_indices)
            
        # Update internal state
        self.token_ids = token_ids
        self.parent_probs = parent_probs
        self.position_ids += 1
        return True


    def update_tree_after_post(self):
        """Return the finalized draft tree after post-speculation."""
        # Update the tree data and mask cache before returning
        self.update_tree(self.tree_data)
        return self.tree

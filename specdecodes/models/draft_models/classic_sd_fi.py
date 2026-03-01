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



class ClassicSDDraftModel(DraftModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.had_first_speculate = False
        self.postspec_count = 0

    def forward(self, input_ids, with_softmax=False, *model_args, **kwargs):
        input_ids, kwargs = self._align_forward_inputs_to_model_device(input_ids, kwargs)
        logits = self.model(input_ids, *model_args, **kwargs).logits
        if with_softmax:
            logits = torch.softmax(logits/self.draft_params.temperature, dim=-1)
            
        return logits

    def _first_step_contract(self) -> FIFirstStepContract:
        # Classic FI retains the existing first-step behavior:
        # append `input_len` tokens and run first forward in prefill mode.
        return FIFirstStepContract(
            attention_mode="prefill",
            batch_position_mode="tree",
            cache_increment="input_len",
            position_ids_style="arange",
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

        self.input_ids_buf = torch.zeros((B, tree_L), dtype=torch.long, device=device)
        self.position_ids_buf = torch.zeros((B, tree_L), dtype=torch.long, device=device)
        self._fi_alloc_batch_position_buffers(
            prefix="",
            batch_size=B,
            token_count=tree_L,
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

        self.graph = tree_cg
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
            context="classic_sd_fi.tree_step",
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
            org_kv_len = kv_len

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
                self.kvCachePool.page_len,
                "NONE", # POS_ENCODING_MODE.NONE,
                self.kvCachePool.cache_data[0].dtype,
            )  
            sampled_probs = self(
                input_ids,
                with_softmax=True,
                logits_to_keep=1,
                position_ids=position_ids,
                kvCachePool=request_kv_cache.kvCachePool,
                batch_position=batch_position,
                mode=first_step_contract.attention_mode,
                flashinferWrapper=self.flashinferWrapper,
            )
           
            kv_len += input_len
            
        org_kv_len = kv_len
        with nvtx.annotate("draft_init_state"):
            parent_probs = torch.ones((1, 1), device=device, dtype=dtype)
            position_ids = torch.full((batch_size, self.draft_params.topk_len), kv_len, device=device, dtype=torch.long)
        
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

        # 5) Main loop
        for depth_i in range(self.draft_params.max_depth):
            with nvtx.annotate("draft_sample", color="green"):
                token_ids, child_probs, parent_indices = self.topk_sampling(
                    sampled_probs,
                    parent_probs,
                    self.draft_params.topk_len
                )
                
                parent_probs = child_probs
                
            with nvtx.annotate("tree_data/update", color="green"):
                self.tree_data.update(token_ids, child_probs, parent_indices)
                
            with nvtx.annotate("tree_mask/update"):
                tree_attention_mask = self.tree_mask_cache.update_tree_mask(parent_indices,return_invert=False)

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
                    mode='tree',
                    device=input_ids.device,
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
                kv_len += self.draft_params.topk_len
                
            with nvtx.annotate("state_update"):
                position_ids += 1

        request_kv_cache.decrement(kv_len - org_kv_len)
        self.update_tree(self.tree_data)
        self.token_ids = token_ids
        self.position_ids = position_ids
        self.parent_probs = parent_probs

        return self.tree
    
    def init_postspec(self):
        self.tree_data = TreeData()
        self.postspec_count = 0
        
    @torch.no_grad()
    def postspec(self):
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
        tree_attention_mask = self.tree_mask_cache.get_tree_mask()
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

import torch
from typing import Any, Optional, Tuple

from .lossy_tree_verify import lossy_bottom_up_verify
from .traversal_verification import traversal_verification_tree

@torch.no_grad()
def verify_tree(
    *,
    tree,
    root_ind: int,
    logits: torch.Tensor,
    sample_token_fn,
    verify_step_fn,
    eos_token_id: Optional[int],
    logits_processor,
    do_sample: bool,
    skip_nodes: int = 0,
    verify_method: str = "exact",
    verify_kwargs: Optional[dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Verify a speculative draft tree against target logits.

    This is a shared abstraction used by tree-based SD methods (classic/subspec/eagle/etc).

    Args:
      tree: Draft tree object (CPU tree in this repo).
      root_ind: Root node index in the *original tree indexing*.
      logits: Target model logits for the decoded tree slice.
      sample_token_fn: Callable like GeneratorBase._sample_token.
      verify_step_fn: Callable like ClassicSDGeneratorBase._verify_step.
      eos_token_id: EOS token id.
      logits_processor: HF LogitsProcessorList (or None if do_sample=False).
      do_sample: Whether to sample target token.
      skip_nodes: Number of leading nodes skipped for this verify call (SubSpec v2 post-spec).
    verify_method: Verification method. Supported: "exact", "lossy", "traversal".
        verify_kwargs: Method-specific kwargs. For lossy:
            {"threshold": float, "window_size": int, "threshold_method": "entropy"|"prob"}.

    Returns:
      sampled_tokens: (1, L)
      hidden_indices: (L,) (indices into the original tree indexing)
      (total_len, accept_len): metrics (accept_len excludes bonus token)
    """
    if verify_method == "traversal":
        return traversal_verification_tree(
            tree = tree,
            root_ind = root_ind,
            logits = logits,
            sample_token_fn = sample_token_fn,
            verify_step_fn = verify_step_fn,
            eos_token_id = eos_token_id,
            logits_processor = logits_processor,
            do_sample = do_sample,
            skip_nodes = skip_nodes,
        )

    method = str(verify_method or "exact").strip().lower()
    vk: dict[str, Any] = dict(verify_kwargs or {})

    if method == "exact":
        # ---- Exact verifier (existing behavior) ----
        global_p = sample_token_fn(logits, logits_processor, do_sample, return_probs=True)
        global_p = global_p.squeeze(0).cpu()

        sampled_tokens = torch.empty(0, dtype=torch.long, device="cpu")
        hidden_indices = torch.empty(0, dtype=torch.long, device="cpu")
        total_len = 0
        accept_len = 0

        node_data = tree.get_tree_data(skip_nodes=skip_nodes)
        token_ids = node_data["token_ids"]
        base = int(skip_nodes)
        visible_rows = int(min(global_p.shape[0], token_ids.shape[0]))

        if visible_rows <= 0:
            # No visible tree rows from logits; emit one token from the first row.
            dist0 = global_p[0].squeeze(0)
            bonus0 = dist0.multinomial(num_samples=1).squeeze(-1) if do_sample else dist0.argmax()
            return bonus0.view(1, 1), torch.tensor([int(root_ind)], dtype=torch.long), (1, 0)

        empty_long = torch.empty(0, dtype=torch.long, device="cpu")

        def _is_visible(idx: torch.Tensor) -> bool:
            if idx.numel() == 0:
                return False
            local = idx - base
            return bool(torch.all((local >= 0) & (local < visible_rows)))

        def _filter_visible_children(children: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if children.numel() == 0:
                return empty_long, empty_long
            local = children - base
            valid = (local >= 0) & (local < visible_rows)
            if not torch.any(valid):
                return empty_long, empty_long
            children = children[valid]
            return children, token_ids[(children - base).long()]

        cur_ind = torch.tensor([int(root_ind)], dtype=torch.long, device="cpu")
        if not _is_visible(cur_ind):
            cur_ind = torch.tensor([base], dtype=torch.long, device="cpu")
        last_valid_cur_ind = cur_ind.clone()
        children_inds, children_token_ids = _filter_visible_children(
            tree.get_children_indices(cur_ind)
        )

        bonus_token = None

        # Preserve existing synchronization behavior.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        while children_inds.numel() > 0:
            total_len += 1
            if not _is_visible(cur_ind):
                break
            dist = global_p[(cur_ind - base).long()].squeeze(0)
            accept_token_id, bonus_token = verify_step_fn(dist, children_token_ids, logits_processor, do_sample)

            if accept_token_id is not None:
                accept_len += 1
                sampled_tokens = torch.cat([sampled_tokens, accept_token_id[None]])
                hidden_indices = torch.cat([hidden_indices, cur_ind])
                last_valid_cur_ind = cur_ind.clone()

                if eos_token_id is not None and int(accept_token_id) == int(eos_token_id):
                    break

                matched_children = children_inds[children_token_ids == accept_token_id]
                if matched_children.numel() == 0:
                    break
                cur_ind = matched_children[:1]

                children_inds, children_token_ids = _filter_visible_children(
                    tree.get_children_indices(cur_ind)
                )
            else:
                break

        # Bonus token, unless EOS already emitted.
        if sampled_tokens.numel() == 0 or (eos_token_id is None) or (int(sampled_tokens[-1].item()) != int(eos_token_id)):
            bonus_ref_ind = cur_ind if _is_visible(cur_ind) else last_valid_cur_ind
            if bonus_token is None:
                dist = global_p[(bonus_ref_ind - base).long()].squeeze(0)
                bonus_token = dist.multinomial(num_samples=1).squeeze(-1) if do_sample else dist.argmax()

            if bonus_token is not None:
                sampled_tokens = torch.cat([sampled_tokens, bonus_token.unsqueeze(0)])
                hidden_indices = torch.cat([hidden_indices, bonus_ref_ind])

        return sampled_tokens.unsqueeze(0), hidden_indices, (int(total_len), int(accept_len))

    if method != "lossy":
        raise ValueError(f"Unsupported verify_method: {verify_method!r} (supported: 'exact', 'lossy')")

    # ---- Lossy bottom-up verifier ----
    if do_sample and logits_processor is None:
        from transformers.generation.logits_process import LogitsProcessorList

        logits_processor = LogitsProcessorList()

    probs = sample_token_fn(logits, logits_processor, do_sample, return_probs=True)
    probs = probs.squeeze(0).detach().cpu()  # (num_nodes_from_logits, vocab)

    num_nodes_from_logits = int(probs.shape[0])
    node_data = tree.get_tree_data(skip_nodes=skip_nodes)
    token_ids = node_data["token_ids"].cpu()[:num_nodes_from_logits]
    parent_indices = node_data["parent_indices"].cpu()[:num_nodes_from_logits]

    num_nodes_local = int(token_ids.numel())
    if num_nodes_local == 0:
        # Degenerate case: no nodes decoded; emit a single bonus token from root context if possible.
        # This should be rare; keep behavior conservative.
        global_p = probs
        bonus = global_p[0].multinomial(1).squeeze(-1) if do_sample else global_p[0].argmax()
        return bonus.view(1, 1), torch.tensor([int(root_ind)], dtype=torch.long), (1, 0)

    root_local = int(root_ind) - int(skip_nodes)
    if root_local < 0 or root_local >= num_nodes_local:
        root_local = 0

    children_lists: list[list[int]] = [[] for _ in range(num_nodes_local)]
    base = int(skip_nodes)
    end = base + num_nodes_local
    for v_orig in range(base, end):
        v_local = v_orig - base
        for c_orig in tree.nodes[v_orig].children:
            c_local = int(c_orig) - base
            if 0 <= c_local < num_nodes_local:
                children_lists[v_local].append(c_local)

    sampled_1d, hidden_1d_local, accept_len = lossy_bottom_up_verify(
        probs=probs,
        token_ids=token_ids,
        parent_indices=parent_indices - int(skip_nodes),
        children_lists=children_lists,
        root_index=root_local,
        eos_token_id=eos_token_id,
        do_sample=do_sample,
        threshold=float(vk.get("threshold", 0.0)),
        window_size=int(vk.get("window_size", 1)),
        threshold_method=str(vk.get("threshold_method", "prob") or "prob").strip().lower(),
    )

    hidden_indices = hidden_1d_local + int(skip_nodes)
    sampled_tokens = sampled_1d.unsqueeze(0)
    total_len = int(sampled_1d.numel())
    return sampled_tokens, hidden_indices, (total_len, int(accept_len))

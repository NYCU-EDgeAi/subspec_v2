import math
import torch
from typing import Tuple, Optional


@torch.no_grad()
def lossy_bottom_up_verify(
    *,
    probs: torch.Tensor,
    token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    children_lists: list[list[int]],
    root_index: int,
    eos_token_id: Optional[int],
    do_sample: bool,
    threshold: float,
    window_size: int,
    threshold_method: str = "prob",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Bottom-up lossy verification over a speculative *tree chunk*.

    Inputs are in *local chunk indexing* (0..num_nodes-1). `probs[i]` is the
    target distribution produced *at node i* (i.e., predicting the next token
    after token_ids[i]).

        Verification rule (as clarified):
            - If target's generated token matches a draft child token, accept it (same
                behavior as the classic verifier).
            - Otherwise, we may accept a *non-matching* draft child token c only if:
                          1) the lookahead constraint is satisfied (can accept at least
                              `window_size` exact-match draft tokens after the chosen child), and
                    2) the threshold gate passes:
                        * threshold_method="entropy": normalized entropy h_j < threshold
                        * threshold_method="prob": probs[parent, token_ids[c]] >= threshold

        This is implemented via a bottom-up DP `best_len[u]` which counts the maximum
        number of draft tokens we can accept starting from context node u.

    Returns:
      sampled_tokens: 1D (accept_len + 1,) tensor (accepted draft tokens + bonus)
      hidden_indices: 1D indices aligned with sampled_tokens semantics used in
        this repo (indices of context nodes whose logits were used to emit each
        sampled token).
      accept_len: number of accepted draft tokens (excluding bonus).
    """
    num_nodes = int(probs.shape[0])

    # Move to CPU to avoid repeated CUDA syncs from scalar `.item()` calls.
    if probs.device.type != "cpu":
        probs = probs.cpu()

    threshold_f = float(threshold)
    required_exact_after = max(0, int(window_size))
    threshold_method_str = str(threshold_method or "entropy").strip().lower()
    if threshold_method_str not in {"entropy", "prob"}:
        raise ValueError(
            f"Unsupported threshold_method: {threshold_method!r} (supported: 'entropy', 'prob')"
        )
    use_entropy = threshold_method_str == "entropy"

    def _compute_entropy_list(prob_rows: torch.Tensor) -> list[float]:
        vocab_size = int(prob_rows.shape[1])
        denom = math.log(vocab_size) if vocab_size > 1 else 1.0
        entropy = torch.special.entr(prob_rows).sum(dim=-1) / denom
        entropy_list = entropy.tolist()
        del entropy
        return entropy_list

    # Target token per node (one per context).
    target_tokens = (
        torch.multinomial(probs, num_samples=1).squeeze(-1) if do_sample else probs.argmax(dim=-1)
    )

    # Cache ids as Python ints to avoid repeated tensor -> Python conversions.
    token_ids_i = token_ids.tolist()
    target_tokens_i = target_tokens.tolist()
    del target_tokens

    def _max_exact_after(children: list[int], target_tok: int, exact_after: list[int]) -> tuple[bool, int]:
        max_exact_after = 0
        has_exact = False
        for c in children:
            c = int(c)
            if token_ids_i[c] == target_tok:
                has_exact = True
                if exact_after[c] > max_exact_after:
                    max_exact_after = exact_after[c]
        return has_exact, max_exact_after

    entropy_i: Optional[list[float]] = None
    if use_entropy:
        entropy_i = _compute_entropy_list(probs)

    # Bottom-up DP over accepted length.
    best_len: list[int] = [0] * num_nodes
    best_next: list[int] = [-1] * num_nodes
    exact_after_len: list[int] = [0] * num_nodes

    for u in range(num_nodes - 1, -1, -1):
        children = children_lists[u]
        if not children:
            continue

        tgt = target_tokens_i[u]
        has_exact, max_exact_after = _max_exact_after(children, tgt, exact_after_len)
        if has_exact:
            exact_after_len[u] = 1 + max_exact_after

        # Choose the child that yields the longest acceptable path.
        best_c = -1
        best_total_len = 0
        best_is_exact = False
        best_p = -1.0

        entropy_u = float(entropy_i[u]) if use_entropy and entropy_i is not None else 0.0
        probs_u = probs[u]

        for c in children:
            c = int(c)
            tok = token_ids_i[c]

            is_exact = (tok == tgt)
            if is_exact:
                p = -1.0  # not used for exact-match unless tie-breaking falls through
            else:
                # Lossy acceptability: window constraint + gate.
                if exact_after_len[c] < required_exact_after:
                    continue
                if use_entropy and entropy_u >= threshold_f:
                    continue
                p = float(probs_u[tok].item())
                if (not use_entropy) and p < threshold_f:
                    continue

            total_len = 1 + best_len[c]
            if total_len > best_total_len:
                best_c = c
                best_total_len = total_len
                best_is_exact = is_exact
                best_p = p
                continue

            if total_len == best_total_len:
                # Tie-break: prefer exact match, else higher prob.
                if is_exact and not best_is_exact:
                    best_c = c
                    best_is_exact = True
                    best_p = p
                elif (not is_exact) and (not best_is_exact) and p > best_p:
                    best_c = c
                    best_p = p

        if best_c >= 0 and best_total_len > 0:
            best_next[u] = best_c
            best_len[u] = best_total_len

    # Top-down extraction for the chosen path.
    sampled_tokens: list[int] = []
    hidden_indices: list[int] = []
    context = root_index
    accept_len = 0

    while True:
        nxt = best_next[context]
        if nxt < 0:
            break

        tok = token_ids_i[nxt]
        sampled_tokens.append(tok)
        hidden_indices.append(context)
        accept_len += 1

        if eos_token_id is not None and tok == int(eos_token_id):
            break

        context = nxt

    # Bonus token from the final context (or root if none accepted).
    if not sampled_tokens or (eos_token_id is None) or (sampled_tokens[-1] != int(eos_token_id)):
        sampled_tokens.append(target_tokens_i[context])
        hidden_indices.append(context)

    return (
        torch.tensor(sampled_tokens, dtype=torch.long),
        torch.tensor(hidden_indices, dtype=torch.long),
        accept_len,
    )

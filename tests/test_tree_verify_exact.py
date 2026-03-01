import torch

from specdecodes.models.utils.tree_verify import verify_tree


class _Node:
    def __init__(self, token_id: int, children: list[int] | None = None):
        self.token_id = torch.tensor(int(token_id), dtype=torch.long)
        self.children = list(children or [])


class _DummyTree:
    def __init__(self):
        # 0(root)->1 ; 1->(4,5). Index 5 is intentionally outside visible logits rows.
        self.nodes = [
            _Node(0, [1]),
            _Node(10, [4, 5]),
            _Node(11, []),
            _Node(12, []),
            _Node(20, []),
            _Node(30, []),
        ]

    def get_tree_data(self, skip_nodes: int = 0):
        token_ids = [int(n.token_id.item()) for n in self.nodes[int(skip_nodes) :]]
        return {"token_ids": torch.tensor(token_ids, dtype=torch.long)}

    def get_children_indices(self, cur_ind: torch.Tensor):
        idx = int(cur_ind.reshape(-1)[0].item())
        return torch.tensor(self.nodes[idx].children, dtype=torch.long)


def _sample_token(logits, logits_processor, do_sample, return_probs=False):
    probs = torch.softmax(logits, dim=-1)
    if return_probs:
        return probs
    return probs.argmax(dim=-1)


def _verify_step(p, token_ids, logits_processor, do_sample):
    sampled_token_id = p.argmax()
    if torch.any(sampled_token_id == token_ids):
        return sampled_token_id, None
    return None, sampled_token_id


def test_exact_verify_filters_children_outside_visible_logits():
    tree = _DummyTree()

    # Visible logits rows are [0..4]. Node index 5 exists in tree but is invisible.
    logits = torch.full((1, 5, 64), -100.0, dtype=torch.float32)
    logits[0, 0, 10] = 10.0  # root accepts token for child index 1
    logits[0, 1, 30] = 10.0  # would choose child index 5 (invisible row)
    logits[0, 4, 20] = 10.0

    sampled_tokens, hidden_indices, _ = verify_tree(
        tree=tree,
        root_ind=0,
        logits=logits,
        sample_token_fn=_sample_token,
        verify_step_fn=_verify_step,
        eos_token_id=None,
        logits_processor=None,
        do_sample=False,
        skip_nodes=0,
        verify_method="exact",
        verify_kwargs={},
    )

    # Should not crash; hidden indices must stay within visible rows.
    assert sampled_tokens.shape[1] >= 2
    assert sampled_tokens[0, 0].item() == 10
    assert sampled_tokens[0, 1].item() == 30
    assert int(hidden_indices.max().item()) < 5

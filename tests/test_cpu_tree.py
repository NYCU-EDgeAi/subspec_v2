import torch

from specdecodes.models.utils.cpu_tree import Tree


def test_add_nodes_rebases_parent_indices_after_leaf_shrink():
    tree = Tree(torch.tensor(1), prob_dtype=torch.float32)
    assert tree.available_leaves == [0]

    token_ids = torch.tensor([[[10, 11, 12, 13, 14, 15]]], dtype=torch.long)
    token_probs = torch.tensor([[[0.5, 0.2, 0.1, 0.1, 0.06, 0.04]]], dtype=torch.float32)
    # Indices > 0 are stale when only one leaf is available.
    parent_indices = torch.tensor([[[0, 1, 2, 3, 0, 4]]], dtype=torch.long)

    tree.add_nodes(token_ids, token_probs, parent_indices)

    assert tree.current_size == 7
    assert len(tree.available_leaves) == 6
    assert all(node.parent == 0 for node in tree.nodes[1:])

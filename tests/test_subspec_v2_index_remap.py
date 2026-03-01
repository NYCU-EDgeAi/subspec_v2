from types import SimpleNamespace

import pytest
import torch

from specdecodes.models.generators.base import GeneratorBase


class _DummyTargetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=4096)
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def forward(self, *_args, **_kwargs):  # pragma: no cover
        return None


class _DummyTokenizer:
    eos_token_id = 2


class _DummyGenerator(GeneratorBase):
    def _generate(self, *_args, **_kwargs):  # pragma: no cover
        raise NotImplementedError


def test_remap_hidden_indices_after_tree_prune():
    generator = _DummyGenerator(
        target_model=_DummyTargetModel(),
        tokenizer=_DummyTokenizer(),
    )

    hidden = torch.tensor([0, 4, 6], dtype=torch.long)
    kept_old = torch.tensor([0, 2, 4, 6], dtype=torch.long)

    remapped = generator._remap_hidden_indices_after_tree_prune(
        hidden,
        kept_old,
        method_name="subspec_sd_v2_fi",
    )
    assert remapped.tolist() == [0, 2, 3]


def test_remap_hidden_indices_raises_when_dropped():
    generator = _DummyGenerator(
        target_model=_DummyTargetModel(),
        tokenizer=_DummyTokenizer(),
    )

    hidden = torch.tensor([0, 5], dtype=torch.long)
    kept_old = torch.tensor([0, 2, 4, 6], dtype=torch.long)

    with pytest.raises(RuntimeError, match="dropped"):
        generator._remap_hidden_indices_after_tree_prune(
            hidden,
            kept_old,
            method_name="subspec_sd_v2_fi",
        )

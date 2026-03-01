from types import SimpleNamespace

import torch

from specdecodes.models.generators.classic_sd import ClassicSDGeneratorBase


class _DummyTargetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=4096)
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.model = self

    def forward(self, *_args, **_kwargs):  # pragma: no cover - not used
        return None


class _DummyTokenizer:
    eos_token_id = 2


class _DummyTree:
    def __init__(self, token_ids):
        self._token_ids = torch.tensor(token_ids, dtype=torch.long)

    def get_tree_data(self, _skip_nodes):
        return {"token_ids": self._token_ids}


def _make_generator(
    step_trace_enabled: bool,
    *,
    step_trace_debug_verify: bool = False,
) -> ClassicSDGeneratorBase:
    return ClassicSDGeneratorBase(
        {
            "step_trace": bool(step_trace_enabled),
            "step_trace_debug_verify": bool(step_trace_debug_verify),
        },
        target_model=_DummyTargetModel(),
        tokenizer=_DummyTokenizer(),
        draft_model=SimpleNamespace(),
        draft_params=SimpleNamespace(),
        cache_implementation="dynamic",
    )


def test_step_trace_disabled_returns_none():
    generator = _make_generator(False)
    generator._init_step_trace()
    generator._append_step_trace(
        is_prev_accepted=False,
        skip_nodes=0,
        tree_size_before_cap=10,
        tree_size_after_cap=9,
        decoded_tree_size=9,
        root_ind_in=0,
        root_ind_out=3,
        accept_len=8,
        hidden_indices_len=9,
        post_verify_used=False,
    )
    assert generator._export_step_trace() is None


def test_is_prev_accepted_stats_are_tracked_even_without_step_trace():
    generator = _make_generator(False)
    generator._init_step_trace()
    generator._append_step_trace(
        is_prev_accepted=False,
        skip_nodes=0,
        tree_size_before_cap=10,
        tree_size_after_cap=9,
        decoded_tree_size=9,
        root_ind_in=0,
        root_ind_out=3,
        accept_len=8,
        hidden_indices_len=9,
        post_verify_used=False,
    )
    generator._append_step_trace(
        is_prev_accepted=True,
        skip_nodes=0,
        tree_size_before_cap=10,
        tree_size_after_cap=9,
        decoded_tree_size=9,
        root_ind_in=0,
        root_ind_out=3,
        accept_len=8,
        hidden_indices_len=9,
        post_verify_used=False,
    )
    generator._append_step_trace(
        is_prev_accepted=True,
        skip_nodes=0,
        tree_size_before_cap=10,
        tree_size_after_cap=9,
        decoded_tree_size=9,
        root_ind_in=0,
        root_ind_out=3,
        accept_len=8,
        hidden_indices_len=9,
        post_verify_used=False,
    )

    stats = generator._export_is_prev_accepted_stats()
    assert stats["is_prev_accepted_count"] == 2
    assert stats["is_prev_accepted_steps"] == 3
    assert abs(stats["is_prev_accepted_rate"] - (2.0 / 3.0)) < 1e-12


def test_step_trace_enabled_records_monotonic_steps():
    generator = _make_generator(True)
    generator._init_step_trace()
    generator._append_step_trace(
        is_prev_accepted=False,
        skip_nodes=0,
        tree_size_before_cap=12,
        tree_size_after_cap=12,
        decoded_tree_size=12,
        root_ind_in=0,
        root_ind_out=5,
        accept_len=11,
        hidden_indices_len=12,
        post_verify_used=False,
    )
    generator._append_step_trace(
        is_prev_accepted=True,
        skip_nodes=12,
        tree_size_before_cap=20,
        tree_size_after_cap=18,
        decoded_tree_size=6,
        root_ind_in=5,
        root_ind_out=2,
        accept_len=4,
        hidden_indices_len=5,
        post_verify_used=True,
    )

    trace = generator._export_step_trace()
    assert isinstance(trace, list)
    assert len(trace) == 2
    assert trace[0]["step"] == 0
    assert trace[1]["step"] == 1
    assert trace[0]["is_prev_accepted"] is False
    assert trace[1]["is_prev_accepted"] is True
    assert trace[1]["post_verify_used"] is True
    assert trace[1]["skip_nodes"] == 12


def test_step_trace_enabled_records_extra_fields():
    generator = _make_generator(True)
    generator._init_step_trace()
    generator._append_step_trace(
        is_prev_accepted=False,
        skip_nodes=0,
        tree_size_before_cap=12,
        tree_size_after_cap=12,
        decoded_tree_size=12,
        root_ind_in=0,
        root_ind_out=5,
        accept_len=11,
        hidden_indices_len=12,
        post_verify_used=False,
        extra_fields={"verify_tree_token_hash": 14, "debug_flag": True},
    )

    trace = generator._export_step_trace()
    assert trace[0]["verify_tree_token_hash"] == 14
    assert trace[0]["debug_flag"] is True


def test_verify_debug_trace_builds_weighted_hashes():
    generator = _make_generator(True, step_trace_debug_verify=True)
    tree = _DummyTree([1, 2, 3])
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],  # argmax=4
                [5.0, 6.0, 2.0, 1.0, 0.0],  # argmax=1
                [9.0, 8.0, 7.0, 6.0, 5.0],  # argmax=0
            ]
        ],
        dtype=torch.float32,
    )

    debug = generator._build_verify_debug_trace(
        tree=tree,
        next_token_logits=logits,
        skip_nodes=0,
    )
    assert debug["verify_tree_token_count"] == 3
    assert debug["verify_tree_token_hash"] == 14
    assert debug["verify_argmax_len"] == 3
    assert debug["verify_argmax_hash"] == 6
    assert debug["verify_argmax_last"] == 0

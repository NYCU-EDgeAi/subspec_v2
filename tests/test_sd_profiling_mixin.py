import torch

from specdecodes.models.utils.mixin import SDProfilingMixin
from specdecodes.models.utils.wandb_logger import wandb_logger


class _DummyCudaEvent:
    def record(self):
        return None

    def elapsed_time(self, _other):
        return 1.0


class _NoVerifyBase:
    def _generate(self, input_ids: torch.LongTensor, *_args, **_kwargs):
        self.post_verify_count = 0
        self.speculate_count = 0
        sampled = torch.tensor([[42]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, sampled], dim=-1)


class _NoVerifyGenerator(SDProfilingMixin, _NoVerifyBase):
    pass


def test_sd_profiling_handles_zero_verify_rounds(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", lambda enable_timing=True: _DummyCudaEvent())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    wandb_logger.clear_log_data()
    generator = _NoVerifyGenerator(profiling=True, profiling_verbose=False)
    output = generator._generate(torch.tensor([[1, 2]], dtype=torch.long))

    assert output.tolist() == [[1, 2, 42]]
    assert generator.profile_data["total_iterations"] == 1
    assert generator.profile_data["total_sampled"] == 1
    assert wandb_logger.log_data["n_tokens"] == 1
    assert wandb_logger.log_data["n_iter"] == 1

from types import SimpleNamespace

import pytest

from run.pipelines import run_benchmark


class _DummyBuilder:
    def __init__(self, tmp_path):
        self._args = SimpleNamespace(
            out_dir=None,
            log_dir=str(tmp_path),
            settings_snapshot={"method": "dummy"},
        )
        self.generator_profiling = False
        self.profiling_verbose = False

    @property
    def args(self):
        return self._args

    def build(self):
        return object(), object(), object(), object()


def test_main_rejects_non_lane_benchmark_before_build(tmp_path):
    class _FailIfBuilt(_DummyBuilder):
        def build(self):  # pragma: no cover - should not be called
            raise AssertionError("builder.build() should not run on pre-validation failure")

    builder = _FailIfBuilt(tmp_path)
    with pytest.raises(ValueError, match="with_answers=True"):
        run_benchmark.main(builder, benchmarks="alpaca", max_samples=1)


@pytest.mark.parametrize("bench_name", ["gsm8k", "mt-bench"])
def test_main_dispatches_with_answers_and_lane(monkeypatch, tmp_path, bench_name):
    builder = _DummyBuilder(tmp_path)

    monkeypatch.setattr(run_benchmark, "prepare_output_dir", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_benchmark, "cleanup_gpu", lambda: None)
    monkeypatch.setattr(run_benchmark, "setup_benchmark_dir", lambda base, bench, snapshot=None: str(tmp_path / bench))

    load_calls = []

    def _fake_load_dataset(bench_name, max_samples=None, seed=0, shuffle=True, with_answers=False):
        load_calls.append((bench_name, with_answers))
        return ["sample"]

    monkeypatch.setattr(run_benchmark, "load_dataset", _fake_load_dataset)

    def _fake_eval(*_args, **_kwargs):
        return {"accuracy": 1.0}

    monkeypatch.setattr(run_benchmark, "get_lane_evaluator", lambda *_args, **_kwargs: _fake_eval)
    monkeypatch.setattr(run_benchmark, "resolve_lane_for_task", lambda *_args, **_kwargs: "behavior")

    results = []

    def _fake_append(log_dir, bench_name, metrics, digits=3):
        results.append((bench_name, metrics))

    monkeypatch.setattr(run_benchmark, "append_benchmark_result", _fake_append)

    run_benchmark.main(builder, benchmarks=bench_name, max_samples=1, lane="behavior")

    assert load_calls == [(bench_name, True)]
    assert [name for name, _ in results] == [bench_name]
    assert results[0][1]["lane"] == "behavior"
